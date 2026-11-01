import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  fetchTerminalInfo,
  runAction,
  terminateAction,
  type TerminalCommandInfo,
  type TerminalFlagInfo,
  type TerminalProgress,
} from '../../api/terminal';

/** Only these actions are offered in this app (the backend also filters them). */
const ALLOWED_ACTIONS = new Set(['randomized_builder', 'history']);

interface Props {
  jobId: string;
  /** Catalog dictionary being reviewed; lets the backend resolve the GPL segmentation. */
  dictionaryId?: number | null;
  /** Kept mounted so progress + results survive closing and reopening. */
  open: boolean;
  /** Template the reviewer is viewing; default target for `split`. */
  currentTemplate?: number;
  /** Called after an action registers a new dictionary the job switched to. */
  onDictionaryChanged?: (newDictionaryId: number) => void;
  onClose: () => void;
}

const NONE_NOTE = 'if you don\'t want to use this parameter, write "none"';

function defaultValue(flag: TerminalFlagInfo, currentTemplate?: number): string {
  if (flag.name === 'template' && (flag.default === '' || flag.default == null)) {
    return currentTemplate != null ? String(currentTemplate) : '';
  }
  return flag.default ?? '';
}

function fieldValid(flag: TerminalFlagInfo, value: string): boolean {
  const text = value.trim();
  if (text === '') return false;
  if (flag.allowNone && text.toLowerCase() === 'none') return true;
  if (flag.choices && flag.choices.length > 0) {
    return flag.choices.includes(text);
  }
  if (flag.kind === 'int') return /^-?\d+$/.test(text);
  if (flag.kind === 'float') {
    const n = Number(text);
    return Number.isFinite(n);
  }
  return true;
}

function ProgressReadout({
  progress,
  actionName,
}: {
  progress: TerminalProgress | null;
  actionName?: string | null;
}) {
  const pct = progress?.fraction != null ? Math.round(progress.fraction * 100) : null;
  const timing = [
    progress?.elapsed ? `Elapsed: ${progress.elapsed}` : null,
    progress?.eta ? `~${progress.eta} remaining` : null,
  ]
    .filter(Boolean)
    .join(' · ');

  return (
    <div
      style={{
        marginTop: 10,
        padding: '14px 18px',
        background: '#f8fafc',
        border: '1px solid #e2e8f0',
        borderRadius: 8,
        color: '#0f172a',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 15, fontWeight: 500, flexWrap: 'wrap' }}>
        <span style={{ color: '#b45309', display: 'inline-flex', alignItems: 'center', gap: 6, fontWeight: 600 }}>
          <span style={{ display: 'inline-block', width: 9, height: 9, borderRadius: '50%', background: '#d97706' }} />
          Running{actionName ? ` ${actionName}` : ''}:
        </span>
        <span style={{ color: '#0f172a', fontWeight: 500 }}>{progress?.message || 'processing…'}</span>
        {pct != null ? <span style={{ color: '#6d28d9', fontWeight: 700 }}>({pct}%)</span> : ''}
      </div>
      {pct != null && (
        <div
          style={{
            height: 8,
            marginTop: 10,
            borderRadius: 4,
            background: '#e2e8f0',
            overflow: 'hidden',
          }}
        >
          <div
            style={{
              height: '100%',
              width: `${pct}%`,
              background: 'linear-gradient(90deg, #7c3aed 0%, #a78bfa 100%)',
              transition: 'width 0.2s ease',
            }}
          />
        </div>
      )}
      {timing && <div style={{ marginTop: 8, fontSize: 13.5, color: '#64748b' }}>{timing}</div>}
    </div>
  );
}

export function RepeatActionForm({
  jobId,
  dictionaryId,
  open,
  currentTemplate,
  onDictionaryChanged,
  onClose,
}: Props) {
  const [commands, setCommands] = useState<TerminalCommandInfo[]>([]);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [action, setAction] = useState('');
  const [values, setValues] = useState<Record<string, string>>({});
  const [helpOpen, setHelpOpen] = useState(false);
  /** Parameter whose "Read more" popup is open. */
  const [detailsFlag, setDetailsFlag] = useState<TerminalFlagInfo | null>(null);
  const [running, setRunning] = useState(false);
  const [runningAction, setRunningAction] = useState<string | null>(null);
  const [terminating, setTerminating] = useState(false);
  const [terminateError, setTerminateError] = useState<string | null>(null);
  const [progress, setProgress] = useState<TerminalProgress | null>(null);
  const [result, setResult] = useState<{ ok: boolean; lines: string[] } | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const activeRunId = useRef<string | null>(null);

  const selected = useMemo(
    () => commands.find(command => command.name === action) ?? null,
    [commands, action],
  );

  useEffect(() => {
    let cancelled = false;
    fetchTerminalInfo(jobId)
      .then(info => {
        if (!cancelled) {
          setCommands(info.commands.filter(command => ALLOWED_ACTIONS.has(command.name)));
          setLoadError(null);
        }
      })
      .catch(error => {
        if (!cancelled) setLoadError(String(error));
      });
    return () => {
      cancelled = true;
    };
  }, [jobId]);

  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [result, progress, running]);

  const applyDefaults = useCallback(
    (command: TerminalCommandInfo) => {
      const next: Record<string, string> = {};
      for (const flag of command.flags) {
        next[flag.name] = defaultValue(flag, currentTemplate);
      }
      setValues(next);
    },
    [currentTemplate],
  );

  const onSelectAction = (name: string) => {
    setAction(name);
    setResult(null);
    setProgress(null);
    setDetailsFlag(null);
    const command = commands.find(item => item.name === name);
    if (command) applyDefaults(command);
    else setValues({});
  };

  const canSubmit = Boolean(
    selected
    && selected.runnable
    && selected.available
    && !running
    && selected.flags.every(flag => fieldValid(flag, values[flag.name] ?? '')),
  );

  const onTerminate = async () => {
    if (!activeRunId.current || terminating) return;
    setTerminating(true);
    setTerminateError(null);
    try {
      const stopped = await terminateAction(jobId, activeRunId.current);
      // The poll loop clears `terminating` once the run reports done. Only an
      // unsuccessful call has to reset it here, or the button stays stuck on
      // "Terminating…" for the rest of the run.
      if (!stopped) {
        setTerminateError('The run had already finished on the server.');
        setTerminating(false);
      }
    } catch (error) {
      setTerminateError(error instanceof Error ? error.message : String(error));
      setTerminating(false);
    }
  };

  const submit = async () => {
    if (!selected || !canSubmit) return;
    const actionToRun = selected.name;
    const valuesToRun = { ...values };

    setRunning(true);
    setRunningAction(actionToRun);
    setTerminating(false);
    setTerminateError(null);
    setProgress(null);
    setResult(null);

    // Reset form immediately after submission while keeping progress report and output
    setAction('');
    setValues({});

    try {
      const outcome = await runAction(
        jobId,
        actionToRun,
        valuesToRun,
        currentTemplate,
        setProgress,
        (runId) => {
          activeRunId.current = runId;
        },
        dictionaryId,
      );
      setResult({
        ok: outcome.ok,
        lines: outcome.lines.length > 0 ? outcome.lines : ['(no output)'],
      });
      if (outcome.ok && outcome.newDictionaryId != null) {
        onDictionaryChanged?.(outcome.newDictionaryId);
      }
    } catch (error) {
      setResult({ ok: false, lines: [String(error)] });
    } finally {
      setRunning(false);
      setRunningAction(null);
      setTerminating(false);
      setProgress(null);
      activeRunId.current = null;
    }
  };

  if (!open) return null;

  const inputStyle = {
    width: '100%' as const,
    boxSizing: 'border-box' as const,
    background: '#ffffff',
    color: '#0f172a',
    border: '1.5px solid #cbd5e1',
    borderRadius: 6,
    padding: '10px 14px',
    fontSize: 15,
    outline: 'none',
  };

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label="Dictionary actions"
      onClick={onClose}
      style={{
        position: 'fixed',
        inset: 0,
        background: 'rgba(15, 23, 42, 0.45)',
        backdropFilter: 'blur(4px)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 1000,
        padding: 24,
      }}
    >
      <div
        onClick={event => event.stopPropagation()}
        onKeyDown={event => event.stopPropagation()}
        style={{
          width: 'min(720px, 100%)',
          maxHeight: 'min(760px, 92vh)',
          background: '#ffffff',
          border: '1px solid #e2e8f0',
          borderRadius: 12,
          boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.25), 0 0 0 1px rgba(0, 0, 0, 0.05)',
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
          fontFamily: 'ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
        }}
      >
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 10,
            padding: '14px 20px',
            background: '#f8fafc',
            borderBottom: '1px solid #e2e8f0',
          }}
        >
          <button
            type="button"
            onClick={() => setHelpOpen(true)}
            aria-label="How to use this form"
            title="How to use this form"
            style={{
              width: 28,
              height: 28,
              borderRadius: '50%',
              border: '1px solid #cbd5e1',
              background: '#ffffff',
              color: '#475569',
              fontSize: 14,
              fontWeight: 700,
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            ?
          </button>
          <span style={{ color: '#1e293b', fontSize: 16, fontWeight: 700, letterSpacing: '0.04em', flex: 1 }}>
            Dictionary Actions
          </span>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close actions"
            style={{
              background: 'transparent',
              border: 'none',
              color: '#64748b',
              fontSize: 24,
              lineHeight: 1,
              cursor: 'pointer',
              padding: '2px 8px',
              borderRadius: 4,
            }}
          >
            ×
          </button>
        </div>

        <div
          ref={scrollRef}
          style={{
            flex: 1,
            minHeight: 0,
            overflowY: 'auto',
            padding: '20px 24px',
            fontSize: 15,
            color: '#0f172a',
          }}
        >
          {loadError && (
            <p role="alert" style={{ color: '#dc2626', fontSize: 14, margin: '0 0 16px' }}>
              {loadError}
            </p>
          )}

          <label style={{ display: 'block', marginBottom: 18 }}>
            <div style={{ fontSize: 15, fontWeight: 600, color: '#0f172a', marginBottom: 8 }}>Action</div>
            <select
              value={action}
              disabled={running}
              onChange={event => onSelectAction(event.target.value)}
              aria-label="Choose action"
              style={inputStyle}
            >
              <option value="">Select an action…</option>
              {commands.map(command => (
                <option key={command.name} value={command.name}>
                  {command.name}
                  {!command.runnable ? ' (not implemented yet)' : ''}
                </option>
              ))}
            </select>
          </label>

          {selected && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
              <div
                style={{
                  padding: '14px 16px',
                  background: '#f5f3ff',
                  border: '1px solid #ddd6fe',
                  borderLeft: '4px solid #7c3aed',
                  borderRadius: 6,
                  fontSize: 15.5,
                  lineHeight: 1.55,
                  color: '#1e1b4b',
                  fontWeight: 500,
                }}
              >
                {selected.summary}
              </div>
              {!selected.runnable && (
                <div
                  role="status"
                  style={{
                    padding: '12px 16px',
                    background: '#fffbeb',
                    border: '1px solid #fde68a',
                    borderLeft: '4px solid #f59e0b',
                    borderRadius: 6,
                    fontSize: 14.5,
                    color: '#92400e',
                  }}
                >
                  This action is listed for later use; its pipeline is not wired yet, so Submit stays disabled.
                </div>
              )}
              {selected.flags.map(flag => (
                <label key={flag.name} style={{ display: 'block' }}>
                  <div style={{ fontSize: 15, fontWeight: 600, color: '#0f172a', marginBottom: 8 }}>
                    {flag.name}
                    {!flag.allowNone && (
                      <span style={{ fontSize: 13, color: '#dc2626', fontWeight: 500, marginLeft: 6 }}>
                        (required)
                      </span>
                    )}
                  </div>
                  {flag.choices && flag.choices.length > 0 ? (
                    <select
                      value={values[flag.name] ?? ''}
                      disabled={running}
                      onChange={event =>
                        setValues(current => ({ ...current, [flag.name]: event.target.value }))
                      }
                      aria-label={flag.name}
                      style={inputStyle}
                    >
                      {flag.choices.map(choice => (
                        <option key={choice} value={choice}>
                          {choice}
                        </option>
                      ))}
                    </select>
                  ) : (
                    <input
                      value={values[flag.name] ?? ''}
                      disabled={running}
                      onChange={event =>
                        setValues(current => ({ ...current, [flag.name]: event.target.value }))
                      }
                      aria-label={flag.name}
                      spellCheck={false}
                      style={inputStyle}
                    />
                  )}
                  <div style={{ fontSize: 13.5, color: '#475569', marginTop: 6, lineHeight: 1.45 }}>
                    {flag.help}
                    {flag.allowNone ? ` — ${NONE_NOTE}` : ''}
                    {flag.details && (
                      <>
                        {' '}
                        <button
                          type="button"
                          onClick={event => {
                            // Inside a <label>: stop the click from focusing the input.
                            event.preventDefault();
                            setDetailsFlag(flag);
                          }}
                          aria-label={`Read more about ${flag.name}`}
                          style={{
                            background: 'transparent',
                            border: 'none',
                            padding: 0,
                            font: 'inherit',
                            color: '#7c3aed',
                            fontWeight: 600,
                            cursor: 'pointer',
                            textDecoration: 'underline',
                          }}
                        >
                          Read more
                        </button>
                      </>
                    )}
                  </div>
                </label>
              ))}
            </div>
          )}

          {running && (
            <div style={{ marginTop: 16 }}>
              <ProgressReadout progress={progress} actionName={runningAction} />
            </div>
          )}
          {result && (
            <div
              style={{
                marginTop: 16,
                padding: '14px 16px',
                background: '#f8fafc',
                border: '1px solid #e2e8f0',
                borderRadius: 6,
                fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Consolas, monospace',
                fontSize: 14,
                maxHeight: 220,
                overflowY: 'auto',
              }}
            >
              {result.lines.map((line, index) => (
                <div
                  key={index}
                  style={{
                    color: result.ok ? '#15803d' : '#b91c1c',
                    whiteSpace: 'pre-wrap',
                    wordBreak: 'break-word',
                    lineHeight: 1.5,
                  }}
                >
                  {line}
                </div>
              ))}
            </div>
          )}
        </div>

        {terminateError && (
          <div
            role="alert"
            style={{
              padding: '12px 18px',
              borderTop: '1px solid #fecaca',
              background: '#fef2f2',
              color: '#991b1b',
              fontSize: 14,
            }}
          >
            {terminateError}
          </div>
        )}

        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            gap: 12,
            padding: '14px 20px',
            borderTop: '1px solid #e2e8f0',
            background: '#f8fafc',
          }}
        >
          {running ? (
            <button
              type="button"
              disabled={terminating}
              onClick={() => void onTerminate()}
              aria-label="Terminate running action"
              style={{
                background: terminating ? '#991b1b' : '#dc2626',
                color: '#ffffff',
                border: 'none',
                borderRadius: 6,
                padding: '10px 18px',
                fontSize: 14.5,
                fontWeight: 600,
                cursor: terminating ? 'wait' : 'pointer',
                transition: 'background 0.15s ease',
                boxShadow: '0 2px 6px rgba(220, 38, 38, 0.3)',
              }}
            >
              {terminating ? 'Terminating…' : 'Terminate Action'}
            </button>
          ) : (
            <div />
          )}
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <button
              type="button"
              onClick={onClose}
              style={{
                background: '#ffffff',
                color: '#334155',
                border: '1px solid #cbd5e1',
                borderRadius: 6,
                padding: '10px 18px',
                fontWeight: 500,
                fontSize: 14.5,
                cursor: 'pointer',
              }}
            >
              Cancel
            </button>
            <button
              type="button"
              disabled={!canSubmit}
              onClick={() => void submit()}
              style={{
                background: canSubmit ? '#7c3aed' : '#cbd5e1',
                color: canSubmit ? '#ffffff' : '#64748b',
                border: 'none',
                borderRadius: 6,
                padding: '10px 22px',
                fontWeight: 600,
                fontSize: 15,
                cursor: canSubmit ? 'pointer' : 'not-allowed',
                boxShadow: canSubmit ? '0 2px 10px rgba(124, 58, 237, 0.3)' : undefined,
                transition: 'all 0.15s ease',
              }}
            >
              {running ? 'Running…' : 'Submit'}
            </button>
          </div>
        </div>
      </div>

      {detailsFlag && (
        <div
          role="dialog"
          aria-modal="true"
          aria-label={`About ${detailsFlag.name}`}
          onClick={event => {
            event.stopPropagation();
            setDetailsFlag(null);
          }}
          style={{
            position: 'fixed',
            inset: 0,
            background: 'rgba(15, 23, 42, 0.45)',
            backdropFilter: 'blur(3px)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 1001,
            padding: 24,
          }}
        >
          <div
            onClick={event => event.stopPropagation()}
            style={{
              width: 'min(560px, 100%)',
              maxHeight: '85%',
              overflowY: 'auto',
              background: '#ffffff',
              border: '1px solid #e2e8f0',
              borderRadius: 12,
              boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.25)',
              padding: 24,
              color: '#0f172a',
              fontSize: 14.5,
              lineHeight: 1.6,
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
              <strong style={{ fontSize: 16, color: '#1e293b' }}>{detailsFlag.name}</strong>
              <button
                type="button"
                onClick={() => setDetailsFlag(null)}
                aria-label="Close parameter description"
                style={{ background: 'transparent', border: 'none', color: '#64748b', fontSize: 22, cursor: 'pointer' }}
              >
                ×
              </button>
            </div>
            <div style={{ fontSize: 13, color: '#64748b', marginBottom: 14 }}>
              {detailsFlag.help}
              {detailsFlag.default ? ` · default ${detailsFlag.default}` : ''}
            </div>
            {detailsFlag.details.split(/\n\s*\n/).map((paragraph, index) => (
              <p key={index} style={{ margin: '0 0 12px', color: '#334155' }}>
                {paragraph}
              </p>
            ))}
          </div>
        </div>
      )}

      {helpOpen && (
        <div
          role="dialog"
          aria-modal="true"
          aria-label="How to use dictionary actions"
          onClick={event => {
            event.stopPropagation();
            setHelpOpen(false);
          }}
          style={{
            position: 'fixed',
            inset: 0,
            background: 'rgba(15, 23, 42, 0.45)',
            backdropFilter: 'blur(3px)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 1001,
            padding: 24,
          }}
        >
          <div
            onClick={event => event.stopPropagation()}
            style={{
              width: 'min(600px, 100%)',
              maxHeight: '85%',
              overflowY: 'auto',
              background: '#ffffff',
              border: '1px solid #e2e8f0',
              borderRadius: 12,
              boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.25)',
              padding: 24,
              color: '#0f172a',
              fontSize: 14.5,
              lineHeight: 1.6,
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 14 }}>
              <strong style={{ fontSize: 16, color: '#1e293b' }}>How to use this form</strong>
              <button
                type="button"
                onClick={() => setHelpOpen(false)}
                aria-label="Close help"
                style={{ background: 'transparent', border: 'none', color: '#64748b', fontSize: 22, cursor: 'pointer' }}
              >
                ×
              </button>
            </div>
            <p style={{ margin: '0 0 14px', color: '#334155' }}>
              Choose an action, review the default parameters, change any you need, then click Submit.
              Fields that accept <code>none</code> can be left as <code>none</code> to skip that parameter.
              Submit is enabled as soon as the action is runnable and every required field has a valid value
              (defaults count).
            </p>
            <ul style={{ margin: 0, paddingLeft: 22, color: '#0f172a' }}>
              {commands.map(command => (
                <li key={command.name} style={{ marginBottom: 10 }}>
                  <strong style={{ color: '#7c3aed' }}>{command.name}</strong>
                  {!command.runnable ? ' — not wired yet. ' : ' — '}
                  <span style={{ color: '#334155' }}>{command.summary}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      )}
    </div>
  );
}
