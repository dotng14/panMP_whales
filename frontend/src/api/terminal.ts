import { API_BASE } from './base';

export interface TerminalFlagInfo {
  name: string;
  kwarg: string;
  kind: 'int' | 'float' | 'str';
  metavar: string;
  help: string;
  /** What the analysis uses when the field is omitted (e.g. "0.9", "none"). */
  default: string;
  choices: string[] | null;
  allowNone: boolean;
  /** Long description behind "Read more" (paragraphs split by blank lines); "" = none. */
  details: string;
}

export interface TerminalCommandInfo {
  name: string;
  summary: string;
  example: string;
  available: boolean;
  runnable: boolean;
  flags: TerminalFlagInfo[];
}

export interface TerminalInfo {
  commands: TerminalCommandInfo[];
}

export interface TerminalRunResult {
  ok: boolean;
  lines: string[];
  /** Set when the action persisted a new dictionary the job switched to. */
  newDictionaryId?: number | null;
}

/** Live progress for a long-running action (currently `extend`). */
export interface TerminalProgress {
  /** Human status, e.g. "grew atom 2/5 · coverage 0.71". */
  message: string;
  /** Completion in [0, 1] once estimable, else null. */
  fraction?: number | null;
  /** Wall-clock so far, preformatted (e.g. "1m 20s"). */
  elapsed?: string | null;
  /** Estimated time remaining, preformatted, or null while still unknown. */
  eta?: string | null;
}

/** Action catalog + wiring status for the repeat action form. */
export async function fetchTerminalInfo(jobId: string): Promise<TerminalInfo> {
  const response = await fetch(`${API_BASE}/api/jobs/${jobId}/terminal`);
  if (!response.ok) throw new Error(`Action catalog fetch failed: ${response.status}`);
  return response.json() as Promise<TerminalInfo>;
}

const POLL_INTERVAL_MS = 700;

const sleep = (ms: number) => new Promise<void>(resolve => setTimeout(resolve, ms));

export async function terminateAction(jobId: string, runId: string): Promise<boolean> {
  const response = await fetch(`${API_BASE}/api/jobs/${jobId}/terminal/runs/${runId}`, {
    method: 'DELETE',
  });
  if (!response.ok) {
    // 405 means the path matched but DELETE did not: the backend predates the
    // terminate endpoint. In practice that is a deployed frontend talking to a
    // backend that was never restarted after the pull.
    if (response.status === 405) {
      throw new Error(
        'The backend on this server has no terminate endpoint — it is running an older build '
        + 'than the frontend. Restart the backend service.',
      );
    }
    const detail = (await response.text().catch(() => '')).slice(0, 200);
    throw new Error(`Terminate failed (HTTP ${response.status})${detail ? `: ${detail}` : ''}`);
  }
  const data = (await response.json().catch(() => null)) as { ok?: boolean; terminated?: boolean } | null;
  return data?.terminated ?? true;
}

/**
 * Run one dictionary action. The backend starts it on a background thread
 * (extend can run for minutes) and returns a run id; we poll until it
 * finishes. Resolves with the output, or rejects if the run is lost.
 */
export async function runAction(
  jobId: string,
  action: string,
  params: Record<string, string>,
  currentTemplate?: number,
  onProgress?: (progress: TerminalProgress) => void,
  onRunId?: (runId: string) => void,
  dictionaryId?: number | null,
): Promise<TerminalRunResult> {
  const response = await fetch(`${API_BASE}/api/jobs/${jobId}/terminal`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action, params, currentTemplate, dictionaryId: dictionaryId ?? null }),
  });
  if (!response.ok) {
    const detail = await response.json().catch(() => null);
    throw new Error(detail?.detail ?? `Action failed: ${response.status}`);
  }
  const { runId } = (await response.json()) as { runId: string };
  if (onRunId) onRunId(runId);

  for (;;) {
    await sleep(POLL_INTERVAL_MS);
    const poll = await fetch(`${API_BASE}/api/jobs/${jobId}/terminal/runs/${runId}`);
    if (!poll.ok) {
      const detail = await poll.json().catch(() => null);
      throw new Error(detail?.detail ?? `Action failed: ${poll.status}`);
    }
    const data = (await poll.json()) as {
      status: string;
      ok: boolean;
      lines: string[];
      new_dictionary_id?: number | null;
      progress?: TerminalProgress | null;
    };
    if (data.status === 'done') {
      return { ok: data.ok, lines: data.lines, newDictionaryId: data.new_dictionary_id };
    }
    if (data.progress && onProgress) onProgress(data.progress);
  }
}
