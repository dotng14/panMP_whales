export interface ViewHistoryEntry {
  id: string;
  fileId: string;
  fileName: string;
  timeSec: number;
  /** Short label for the interaction that recorded this point (e.g. "Seek", "Song 3", "Match"). */
  context?: string;
  visitedAt: number;
}

const STORAGE_KEY = 'panmp.view-history.v1';
const MAX_ENTRIES = 60;
const COALESCE_SEC = 5;

export function formatHistoryTimeSec(sec: number): string {
  const s = Math.max(0, Math.floor(sec));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const rem = s % 60;
  const pad = (n: number) => String(n).padStart(2, '0');
  return h > 0 ? `${pad(h)}:${pad(m)}:${pad(rem)}` : `${pad(m)}:${pad(rem)}`;
}

export function loadViewHistory(): ViewHistoryEntry[] {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(
      (item): item is ViewHistoryEntry =>
        typeof item === 'object'
        && item !== null
        && typeof item.id === 'string'
        && typeof item.fileId === 'string'
        && typeof item.fileName === 'string'
        && typeof item.timeSec === 'number'
        && Number.isFinite(item.timeSec)
        && typeof item.visitedAt === 'number',
    );
  } catch {
    return [];
  }
}

function saveViewHistory(entries: ViewHistoryEntry[]): void {
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(entries));
  } catch {
    // Best-effort persistence.
  }
}

export function recordViewHistory(
  current: ViewHistoryEntry[],
  input: { fileId: string; fileName: string; timeSec: number; context?: string },
): ViewHistoryEntry[] {
  const timeSec = Math.max(0, Math.round(input.timeSec * 10) / 10);
  const now = Date.now();
  // Coalesce with any existing entry for the same file within COALESCE_SEC seconds.
  const filtered = current.filter(
    e => !(e.fileId === input.fileId && Math.abs(e.timeSec - timeSec) < COALESCE_SEC),
  );
  const entry: ViewHistoryEntry = {
    id: `${input.fileId}@${timeSec}@${now}`,
    fileId: input.fileId,
    fileName: input.fileName,
    timeSec,
    context: input.context,
    visitedAt: now,
  };
  const next = [entry, ...filtered].slice(0, MAX_ENTRIES);
  saveViewHistory(next);
  return next;
}

export function removeViewHistoryEntry(
  current: ViewHistoryEntry[],
  id: string,
): ViewHistoryEntry[] {
  const next = current.filter(e => e.id !== id);
  saveViewHistory(next);
  return next;
}

export function clearViewHistory(): ViewHistoryEntry[] {
  saveViewHistory([]);
  return [];
}
