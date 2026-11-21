/**
 * Universal recording key: site letter + wall-clock start
 * (``YYYY-MM-DD HH:MM:SS`` or a prefix). Mirrors ``database.recording_when``.
 */

import { parseWavName } from './socalWavName';

export const DEFAULT_SITE = 'H';

export interface RecordingWhen {
  site: string | null;
  /** Normalized prefix or full ``YYYY-MM-DD[ HH:MM[:SS]]``. */
  start: string | null;
}

const ISO_FULL =
  /^(?<date>\d{4}-\d{2}-\d{2})(?:[ T](?<time>\d{2}(?::\d{2}(?::\d{2})?)?)?)?$/;
const SITE_PREFIX = /^(?:site\s*)?(?<site>[A-Za-z])\s+(?<rest>.+)$/i;

/** Map stem ``YYMMDD`` / ``HHMMSS`` chips to wall-clock When text. */
export function yymmddHhmmssToStartDisplay(date?: string | null, time?: string | null): string | null {
  return yymmddHhmmssToStart(date, time);
}

/** Stem ``YYMMDD`` → ``YYYY-MM-DD`` (``date '+%Y-%m-%d'``). */
export function displayDateYmd(date?: string | null): string | null {
  if (!date) return null;
  if (/^\d{4}-\d{2}-\d{2}$/.test(date)) return date;
  const full = yymmddHhmmssToStart(date, null);
  return full;
}

/** Stem ``HHMMSS`` → ``HH:MM:SS`` (``date '+%H:%M:%S'``). */
export function displayTimeHms(time?: string | null): string | null {
  if (!time) return null;
  if (/^\d{2}:\d{2}:\d{2}$/.test(time)) return time;
  if (!/^\d{6}$/.test(time)) return time;
  return `${time.slice(0, 2)}:${time.slice(2, 4)}:${time.slice(4, 6)}`;
}

/** ``YYYY-MM-DD`` → stem ``YYMMDD`` for API filters. */
export function isoDateToYymmdd(iso: string): string | null {
  const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(iso.trim());
  if (!m) return null;
  return `${m[1].slice(2)}${m[2]}${m[3]}`;
}

/** ``HH:MM:SS`` → stem ``HHMMSS`` for API filters. */
export function isoTimeToHhmmss(iso: string): string | null {
  const m = /^(\d{2}):(\d{2}):(\d{2})$/.exec(iso.trim());
  if (!m) return null;
  return `${m[1]}${m[2]}${m[3]}`;
}

function yymmddHhmmssToStart(date?: string | null, time?: string | null): string | null {
  if (!date || !/^\d{6}$/.test(date)) return null;
  const y = 2000 + parseInt(date.slice(0, 2), 10);
  const mo = parseInt(date.slice(2, 4), 10);
  const d = parseInt(date.slice(4, 6), 10);
  if (!(mo >= 1 && mo <= 12 && d >= 1 && d <= 31)) return null;
  const base = `${String(y).padStart(4, '0')}-${String(mo).padStart(2, '0')}-${String(d).padStart(2, '0')}`;
  if (!time || !/^\d{6}$/.test(time)) return base;
  const hh = parseInt(time.slice(0, 2), 10);
  const mm = parseInt(time.slice(2, 4), 10);
  const ss = parseInt(time.slice(4, 6), 10);
  if (!(hh >= 0 && hh <= 23 && mm >= 0 && mm <= 59 && ss >= 0 && ss <= 59)) return base;
  return `${base} ${String(hh).padStart(2, '0')}:${String(mm).padStart(2, '0')}:${String(ss).padStart(2, '0')}`;
}

export function normalizeStartPrefix(raw: string): string | null {
  let text = raw.trim().replace('T', ' ');
  text = text.replace(/\s+/g, ' ');
  if (!text) return null;
  const m = ISO_FULL.exec(text);
  if (!m?.groups) return null;
  const date = m.groups.date;
  const time = m.groups.time;
  if (!time) return date;
  const parts = time.split(':');
  if (parts.length === 1) return `${date} ${parts[0]}`;
  if (parts.length === 2) return `${date} ${parts[0]}:${parts[1]}`;
  return `${date} ${parts[0]}:${parts[1]}:${parts[2]}`;
}

/** Parse When field or paste (ISO datetime, ``H …``, or legacy SOCAL stem). */
export function parseRecordingWhen(
  text: string,
  defaultSite: string = DEFAULT_SITE,
): RecordingWhen {
  const raw = text.trim();
  if (!raw) return { site: defaultSite, start: null };

  let site = defaultSite;
  let rest = raw;
  const sm = SITE_PREFIX.exec(raw);
  if (sm?.groups && normalizeStartPrefix(sm.groups.rest) != null) {
    site = sm.groups.site.toUpperCase();
    rest = sm.groups.rest.trim();
  }

  const start = normalizeStartPrefix(rest);
  if (start != null) return { site, start };

  const parsed = parseWavName(raw);
  if (parsed) {
    return {
      site: parsed.site || defaultSite,
      start: yymmddHhmmssToStart(parsed.date, parsed.time),
    };
  }

  return { site: defaultSite, start: null };
}

export const WHEN_EXAMPLE = '2023-04-26 19:35:00';
export const WHEN_HELP = [
  'Primary key is site + wall-clock start time (UTC), not folder layout.',
  'Site is fixed as H for now.',
  'Type Year / Month / Day / Hour / Minute / Second (same order as date \'+%Y-%m-%d %H:%M:%S\').',
  'Or paste 2023-04-26 19:35:00 / a SOCAL stem into When — boxes fill automatically.',
];

/** Boxes in ``date '+%Y-%m-%d %H:%M:%S'`` order. */
export type WallClockField = 'year' | 'month' | 'day' | 'hour' | 'minute' | 'second';

export const WALL_CLOCK_FIELDS: WallClockField[] = [
  'year',
  'month',
  'day',
  'hour',
  'minute',
  'second',
];

export const WALL_CLOCK_LABELS: Record<WallClockField, string> = {
  year: 'Year',
  month: 'Month',
  day: 'Day',
  hour: 'Hour',
  minute: 'Minute',
  second: 'Second',
};

export const WALL_CLOCK_PLACEHOLDERS: Record<WallClockField, string> = {
  year: 'YYYY',
  month: 'MM',
  day: 'DD',
  hour: 'HH',
  minute: 'MM',
  second: 'SS',
};

export const WALL_CLOCK_MAX_LEN: Record<WallClockField, number> = {
  year: 4,
  month: 2,
  day: 2,
  hour: 2,
  minute: 2,
  second: 2,
};

export type WallClockParts = Partial<Record<WallClockField, string>>;

/** Build a ``start`` prefix from typed boxes (left-to-right; empty stops the chain). */
export function partsToStartPrefix(parts: WallClockParts): string | null {
  const y = (parts.year ?? '').trim();
  const mo = (parts.month ?? '').trim();
  const d = (parts.day ?? '').trim();
  const h = (parts.hour ?? '').trim();
  const mi = (parts.minute ?? '').trim();
  const s = (parts.second ?? '').trim();

  if (!y || !/^\d{1,4}$/.test(y)) return null;
  // Partial year (e.g. "202") → LIKE '202%'; need full year before month/day.
  if (y.length < 4) return y;

  let out = y;
  if (!mo || !/^\d{1,2}$/.test(mo)) return out;
  out = `${y}-${mo.padStart(2, '0')}`;

  if (!d || !/^\d{1,2}$/.test(d)) return out;
  out = `${out.slice(0, 7)}-${d.padStart(2, '0')}`;

  if (!h || !/^\d{1,2}$/.test(h)) return out;
  out = `${out} ${h.padStart(2, '0')}`;

  if (!mi || !/^\d{1,2}$/.test(mi)) return out;
  out = `${out}:${mi.padStart(2, '0')}`;

  if (!s || !/^\d{1,2}$/.test(s)) return out;
  return `${out}:${s.padStart(2, '0')}`;
}

/** Split a normalized start / stem into typed boxes. */
export function startToParts(start: string | null | undefined): WallClockParts {
  if (!start) return {};
  const m = /^(\d{4})(?:-(\d{2})(?:-(\d{2})(?:[ T](\d{2})(?::(\d{2})(?::(\d{2}))?)?)?)?)?$/.exec(
    start.trim(),
  );
  if (!m) return {};
  const parts: WallClockParts = { year: m[1] };
  if (m[2]) parts.month = m[2];
  if (m[3]) parts.day = m[3];
  if (m[4]) parts.hour = m[4];
  if (m[5]) parts.minute = m[5];
  if (m[6]) parts.second = m[6];
  return parts;
}

export function partsFromStemDateTime(date?: string | null, time?: string | null): WallClockParts {
  return startToParts(yymmddHhmmssToStart(date, time) ?? undefined);
}

export function partsAreEmpty(parts: WallClockParts): boolean {
  return WALL_CLOCK_FIELDS.every(k => !parts[k]?.trim());
}
