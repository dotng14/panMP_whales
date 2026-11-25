import { useCallback, useEffect, useRef, useState, type CSSProperties } from 'react';
import {
  matchRecordings,
  type MatchEntry,
  type MatchResponse,
  type RecordingFilters,
} from '../api/library';
import {
  DEFAULT_SITE,
  WHEN_EXAMPLE,
  WHEN_HELP,
  WALL_CLOCK_FIELDS,
  WALL_CLOCK_LABELS,
  WALL_CLOCK_MAX_LEN,
  WALL_CLOCK_PLACEHOLDERS,
  parseRecordingWhen,
  partsAreEmpty,
  partsFromStemDateTime,
  partsToStartPattern,
  partsToStartPrefix,
  startToParts,
  type WallClockField,
  type WallClockParts,
} from '../lib/recordingWhen';
import { parseWavName } from '../lib/socalWavName';

const PAGE_SIZE = 50;

function fmtSize(bytes: number): string {
  if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(1)} GB`;
  if (bytes >= 1e6) return `${(bytes / 1e6).toFixed(0)} MB`;
  return `${(bytes / 1e3).toFixed(0)} KB`;
}

interface Props {
  onSelectFile: (file: { fileId: string; name: string } | null) => void;
  selectedFileId: string | null;
  disabled?: boolean;
  libraryCacheRefresh?: { fileId: string; nonce: number } | null;
}

function toFilters(whenText: string, parts: WallClockParts): RecordingFilters {
  const filters: RecordingFilters = { site: DEFAULT_SITE, family: 'SOCAL' };
  const fromParts = partsToStartPattern(parts);
  const fromWhen = parseRecordingWhen(whenText, DEFAULT_SITE).start;

  // Prefer typed boxes when any are filled (supports any subset, e.g. Hour alone).
  if (fromParts) {
    filters.start = fromParts;
  } else if (fromWhen) {
    filters.start = fromWhen;
  } else if (whenText.trim() && partsAreEmpty(parts)) {
    filters.q = whenText.trim();
  }
  return filters;
}

function digitsOnly(raw: string, maxLen: number): string {
  return raw.replace(/\D/g, '').slice(0, maxLen);
}

export function RecordingPicker({ onSelectFile, selectedFileId, disabled, libraryCacheRefresh }: Props) {
  const [when, setWhen] = useState('');
  const [parts, setParts] = useState<WallClockParts>({});
  const [match, setMatch] = useState<MatchResponse | null>(null);
  const [offset, setOffset] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [helpOpen, setHelpOpen] = useState(false);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const lastCacheRefreshNonceRef = useRef<number | null>(null);
  const listScrollRef = useRef<HTMLDivElement>(null);
  const onSelectFileRef = useRef(onSelectFile);
  useEffect(() => {
    onSelectFileRef.current = onSelectFile;
  });

  const load = useCallback(
    async (
      whenText: string,
      nextParts: WallClockParts,
      off: number,
      opts?: { preserveSelection?: boolean; silent?: boolean },
    ) => {
      const scrollTop =
        opts?.silent && listScrollRef.current ? listScrollRef.current.scrollTop : null;
      if (!opts?.silent) setLoading(true);
      setError(null);
      const filters = toFilters(whenText, nextParts);
      try {
        const matchRes = await matchRecordings({ ...filters, offset: off, limit: PAGE_SIZE });
        setMatch(matchRes);
        if (!opts?.preserveSelection) {
          onSelectFileRef.current(null);
        }
        if (scrollTop !== null) {
          requestAnimationFrame(() => {
            if (listScrollRef.current) listScrollRef.current.scrollTop = scrollTop;
          });
        }
      } catch (e) {
        setError(String(e));
      } finally {
        if (!opts?.silent) setLoading(false);
      }
    },
    [],
  );

  useEffect(() => {
    load('', {}, 0);
  }, [load]);

  useEffect(() => {
    if (!libraryCacheRefresh) return;
    if (libraryCacheRefresh.nonce === lastCacheRefreshNonceRef.current) return;
    lastCacheRefreshNonceRef.current = libraryCacheRefresh.nonce;
    setMatch(prev =>
      prev
        ? {
            ...prev,
            entries: prev.entries.map(entry =>
              entry.fileId === libraryCacheRefresh.fileId ? { ...entry, cached: true } : entry,
            ),
          }
        : prev,
    );
    load(when, parts, offset, { preserveSelection: true, silent: true });
  }, [libraryCacheRefresh]); // eslint-disable-line react-hooks/exhaustive-deps

  function scheduleLoad(whenText: string, nextParts: WallClockParts) {
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => load(whenText, nextParts, 0), 250);
  }

  function applyWhen(raw: string) {
    setWhen(raw);
    setOffset(0);

    const whenParsed = parseRecordingWhen(raw, DEFAULT_SITE);
    const stemParsed = parseWavName(raw);
    let next: WallClockParts = {};
    if (stemParsed?.date) {
      next = partsFromStemDateTime(stemParsed.date, stemParsed.time);
    } else if (whenParsed.start) {
      next = startToParts(whenParsed.start);
    }

    setParts(next);
    scheduleLoad(raw, next);
  }

  function setPart(field: WallClockField, raw: string) {
    setOffset(0);
    const cleaned = digitsOnly(raw, WALL_CLOCK_MAX_LEN[field]);
    const next: WallClockParts = { ...parts };
    if (!cleaned) delete next[field];
    else next[field] = cleaned;

    const prefix = partsToStartPrefix(next);
    const nextWhen = prefix ?? '';
    setParts(next);
    setWhen(nextWhen);
    scheduleLoad(nextWhen, next);
  }

  function clearAll() {
    setWhen('');
    setParts({});
    setOffset(0);
    load('', {}, 0);
  }

  const totalPages = match ? Math.ceil(match.total / PAGE_SIZE) : 0;
  const currentPage = Math.floor(offset / PAGE_SIZE) + 1;
  const hasFilters = Boolean(when.trim()) || !partsAreEmpty(parts);

  const boxStyle: CSSProperties = {
    width: '100%',
    boxSizing: 'border-box',
    padding: '6px 8px',
    borderRadius: 6,
    border: '1px solid #d1d5db',
    fontSize: 13,
    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Consolas, monospace',
    fontVariantNumeric: 'tabular-nums',
    textAlign: 'center',
  };

  return (
    <div
      style={{
        border: '1px solid #d1d5db',
        borderRadius: 8,
        overflow: 'hidden',
        fontSize: 13,
        maxWidth: 720,
        background: '#fff',
      }}
    >
      <div style={{ padding: '10px 12px', background: '#f9fafb', borderBottom: '1px solid #d1d5db' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
          <label htmlFor="recording-when" style={{ flex: 1, fontSize: 12, color: '#6b7280' }}>
            When / paste name
          </label>
          <button
            type="button"
            aria-label="How to find recordings"
            aria-expanded={helpOpen}
            title="How to find recordings"
            onClick={() => setHelpOpen(v => !v)}
            style={{
              width: 22,
              height: 22,
              borderRadius: '50%',
              border: helpOpen ? '1px solid #2563eb' : '1px solid #9ca3af',
              background: helpOpen ? '#dbeafe' : '#fff',
              color: helpOpen ? '#1d4ed8' : '#4b5563',
              fontSize: 13,
              fontWeight: 600,
              lineHeight: '20px',
              padding: 0,
              cursor: 'pointer',
            }}
          >
            ?
          </button>
        </div>
        <div style={{ display: 'flex', gap: 8, marginBottom: 8 }}>
          <div
            aria-label="Site"
            title="Site (fixed)"
            style={{
              minWidth: 48,
              padding: '6px 10px',
              borderRadius: 6,
              border: '1px solid #d1d5db',
              background: '#fff',
              fontWeight: 600,
              textAlign: 'center',
              alignSelf: 'stretch',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            {DEFAULT_SITE}
          </div>
          <input
            id="recording-when"
            type="text"
            value={when}
            disabled={disabled}
            placeholder={WHEN_EXAMPLE}
            onChange={e => applyWhen(e.target.value)}
            style={{
              flex: 1,
              padding: '6px 10px',
              borderRadius: 6,
              border: '1px solid #d1d5db',
              fontSize: 13,
              fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Consolas, monospace',
            }}
          />
          {hasFilters && (
            <button
              type="button"
              onClick={clearAll}
              disabled={disabled}
              style={{
                padding: '6px 10px',
                borderRadius: 6,
                border: '1px solid #d1d5db',
                background: '#fff',
                cursor: 'pointer',
                fontSize: 13,
              }}
            >
              Clear
            </button>
          )}
        </div>
        <div style={{ fontSize: 12, color: '#6b7280' }}>
          Type <code>date &apos;+%Y-%m-%d %H:%M:%S&apos;</code> parts below, or paste into When.
        </div>
        {helpOpen && (
          <div
            role="region"
            aria-label="Recording picker help"
            style={{
              marginTop: 10,
              padding: 12,
              borderRadius: 8,
              border: '1px solid #dbeafe',
              background: '#eff6ff',
              color: '#1e3a5f',
              fontSize: 12,
              lineHeight: 1.45,
            }}
          >
            <div style={{ fontWeight: 600, marginBottom: 6 }}>Site + wall-clock boxes</div>
            <ul style={{ margin: 0, paddingLeft: 18 }}>
              {WHEN_HELP.map(line => (
                <li key={line} style={{ marginBottom: 4 }}>
                  {line}
                </li>
              ))}
              <li>Year alone lists that year; Year+Month+Day lists that day; fill time to narrow further.</li>
            </ul>
          </div>
        )}
      </div>

      <div style={{ padding: '10px 12px', borderBottom: '1px solid #e5e7eb' }}>
        <div style={{ fontSize: 12, color: '#6b7280', marginBottom: 8 }}>
          Components <span style={{ color: '#9ca3af' }}>(%Y-%m-%d %H:%M:%S)</span>
        </div>
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(6, minmax(0, 1fr))',
            gap: 8,
            alignItems: 'end',
          }}
        >
          {WALL_CLOCK_FIELDS.map(field => (
            <div key={field}>
              <label
                htmlFor={`recording-${field}`}
                style={{ display: 'block', fontSize: 11, color: '#6b7280', marginBottom: 4 }}
              >
                {WALL_CLOCK_LABELS[field]}
              </label>
              <input
                id={`recording-${field}`}
                type="text"
                inputMode="numeric"
                pattern="[0-9]*"
                maxLength={WALL_CLOCK_MAX_LEN[field]}
                placeholder={WALL_CLOCK_PLACEHOLDERS[field]}
                value={parts[field] ?? ''}
                disabled={disabled}
                onChange={e => setPart(field, e.target.value)}
                style={boxStyle}
              />
            </div>
          ))}
        </div>
      </div>

      <div style={{ padding: '8px 12px', fontSize: 12, color: '#6b7280', borderBottom: '1px solid #f3f4f6' }}>
        {match ? `${match.total.toLocaleString()} matching recording${match.total === 1 ? '' : 's'}` : '…'}
      </div>

      <div ref={listScrollRef} style={{ maxHeight: 320, overflowY: 'auto' }}>
        {loading && (
          <div style={{ padding: '20px 12px', color: '#9ca3af', textAlign: 'center' }}>Loading…</div>
        )}
        {error && <div style={{ padding: 12, color: '#dc2626' }}>{error}</div>}
        {!loading && match && match.entries.length === 0 && (
          <div style={{ padding: '20px 12px', color: '#9ca3af', textAlign: 'center' }}>
            No recordings match these filters.
          </div>
        )}
        {!loading &&
          match?.entries.map((entry: MatchEntry) => {
            const isSelected = entry.fileId === selectedFileId;
            return (
              <div
                key={entry.fileId}
                onClick={() => {
                  if (disabled) return;
                  onSelectFile({ fileId: entry.fileId, name: entry.name });
                }}
                style={{
                  display: 'flex',
                  flexDirection: 'column',
                  padding: '8px 12px',
                  gap: 2,
                  cursor: 'pointer',
                  background: isSelected ? '#dbeafe' : 'transparent',
                  borderBottom: '1px solid #f3f4f6',
                }}
                onMouseEnter={e => {
                  if (!isSelected) (e.currentTarget as HTMLDivElement).style.background = '#f9fafb';
                }}
                onMouseLeave={e => {
                  if (!isSelected) (e.currentTarget as HTMLDivElement).style.background = 'transparent';
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <span
                    style={{
                      flex: 1,
                      fontVariantNumeric: 'tabular-nums',
                      fontWeight: 500,
                      color: '#111827',
                    }}
                  >
                    {entry.startTime || '—'}
                  </span>
                  {entry.cached && (
                    <span
                      title="Pre-processed — loads instantly"
                      style={{
                        fontSize: 11,
                        fontWeight: 600,
                        color: '#15803d',
                        background: '#dcfce7',
                        border: '1px solid #86efac',
                        borderRadius: 4,
                        padding: '1px 6px',
                      }}
                    >
                      ready
                    </span>
                  )}
                  {entry.sizeBytes != null && (
                    <span style={{ color: '#9ca3af', minWidth: 60, textAlign: 'right' }}>
                      {fmtSize(entry.sizeBytes)}
                    </span>
                  )}
                </div>
                <div
                  style={{
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    whiteSpace: 'nowrap',
                    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Consolas, monospace',
                    fontSize: 11,
                    color: '#6b7280',
                  }}
                >
                  {entry.name}
                </div>
              </div>
            );
          })}
      </div>

      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          padding: '8px 12px',
          borderTop: '1px solid #d1d5db',
          background: '#f9fafb',
        }}
      >
        <button
          type="button"
          onClick={() => {
            const next = Math.max(0, offset - PAGE_SIZE);
            setOffset(next);
            load(when, parts, next, { preserveSelection: true });
          }}
          disabled={offset === 0 || loading}
          style={{
            padding: '3px 10px',
            borderRadius: 4,
            border: '1px solid #d1d5db',
            background: '#fff',
            cursor: offset === 0 ? 'default' : 'pointer',
            fontSize: 13,
          }}
        >
          ◀
        </button>
        <span style={{ flex: 1, textAlign: 'center', color: '#6b7280' }}>
          {match
            ? `Page ${currentPage} of ${Math.max(1, totalPages)} (${match.total.toLocaleString()} items)`
            : ''}
        </span>
        <button
          type="button"
          onClick={() => {
            const next = offset + PAGE_SIZE;
            setOffset(next);
            load(when, parts, next, { preserveSelection: true });
          }}
          disabled={!match || offset + PAGE_SIZE >= match.total || loading}
          style={{
            padding: '3px 10px',
            borderRadius: 4,
            border: '1px solid #d1d5db',
            background: '#fff',
            cursor: 'pointer',
            fontSize: 13,
          }}
        >
          ▶
        </button>
      </div>
    </div>
  );
}
