/**
 * Mirror of ``database.wav_name`` SOCAL parsing for instant paste feedback.
 * Server facets/match remain the source of truth for inventory.
 */

export const COMPONENT_KEYS = [
  'family',
  'site',
  'deployment',
  'disk',
  'date',
  'time',
  'df',
  'channel',
  'resample',
] as const;

export type ComponentKey = (typeof COMPONENT_KEYS)[number];

export interface RecordingComponents {
  family?: string | null;
  site?: string | null;
  deployment?: string | null;
  disk?: string | null;
  date?: string | null;
  time?: string | null;
  df?: string | null;
  channel?: string | null;
  resample?: string | null;
}

/** Chip fields shown in the SOCAL picker (family is implicit). */
export const SOCAL_CHIP_KEYS: ComponentKey[] = [
  'site',
  'deployment',
  'disk',
  'date',
  'time',
  'df',
  'channel',
  'resample',
];

export const SOCAL_CHIP_LABELS: Record<string, string> = {
  site: 'Site',
  deployment: 'Deployment',
  disk: 'Disk',
  date: 'Date',
  time: 'Time',
  df: 'df',
  channel: 'Channel',
  resample: 'Resample',
};

export function normalizeWavBasename(name: string): string {
  const base = name.trim().split(/[/\\]/).pop() ?? name.trim();
  return base.toLowerCase().endsWith('.wav') ? base.slice(0, -4) : base;
}

const MODERN_HEAD = /^SOCAL_H_(\d+)(?:_HE)?/i;
const OLD_HEAD = /^SOCAL(\d+)H/i;
const DISK_EXPLICIT = /(?:^|_)(?:DL(\d+)|disk(\d+)|d(\d+))(?=_|\.|$)/i;
const DISK_BARE = /^_(\d{1,2})_(?=\d{6}_\d{6})/;
const DATETIME = /(?<!\d)(\d{6})_(\d{6})(?!\d)/;
const DF = /(?:^|[._])df(\d+)(?=[._]|$)/i;
const DF_SHORT = /\.d(\d+)(?=\.|$)/i;
const CHANNEL = /(?<![A-Za-z0-9])(x(?:\.x)?)(?=(?:_\d+kHz)?$)/i;
const RESAMPLE = /_(\d+kHz)$/i;

function parseSocal(stem: string): RecordingComponents | null {
  if (!stem.toUpperCase().startsWith('SOCAL')) return null;

  let site: string | undefined;
  let deployment: string | undefined;
  let rest = stem;

  const modern = stem.match(MODERN_HEAD);
  if (modern) {
    site = 'H';
    deployment = modern[1];
    rest = stem.slice(modern[0].length);
  } else {
    const old = stem.match(OLD_HEAD);
    if (old) {
      site = 'H';
      deployment = old[1];
      rest = stem.slice(old[0].length);
    } else if (/^SOCAL_H$/i.test(stem)) {
      return { family: 'SOCAL', site: 'H' };
    } else if (/^SOCAL$/i.test(stem)) {
      return { family: 'SOCAL' };
    } else {
      return { family: 'SOCAL' };
    }
  }

  let disk: string | undefined;
  const diskExpl = rest.match(DISK_EXPLICIT);
  if (diskExpl) {
    if (diskExpl[1] != null) disk = `DL${diskExpl[1]}`;
    else if (diskExpl[2] != null) disk = `disk${diskExpl[2]}`;
    else disk = `d${diskExpl[3]}`;
  } else {
    const bare = rest.match(DISK_BARE);
    if (bare) disk = bare[1];
  }

  const dt = stem.match(DATETIME);
  let date = dt?.[1];
  const time = dt?.[2];
  if (!dt) {
    const alone = rest.match(/(?<!\d)(\d{6})(?!\d)/);
    if (alone) date = alone[1];
  }

  const ch = stem.match(CHANNEL);
  const channel = ch ? ch[1].toLowerCase() : undefined;

  const rs = stem.match(RESAMPLE);
  const resample = rs?.[1];

  const dfm = stem.match(DF) ?? stem.match(DF_SHORT);
  let df: string | undefined;
  if (dfm) df = dfm[1];
  else if (channel && date && time) df = 'native';

  return {
    family: 'SOCAL',
    site,
    deployment,
    disk,
    date,
    time,
    df,
    channel,
    resample,
  };
}

/** Parse a pasted name; SOCAL only in v1 (registry seam for later families). */
export function parseWavName(name: string): RecordingComponents | null {
  const stem = normalizeWavBasename(name);
  if (!stem) return null;
  return parseSocal(stem);
}

export function filledComponents(c: RecordingComponents): Partial<Record<ComponentKey, string>> {
  const out: Partial<Record<ComponentKey, string>> = {};
  for (const key of COMPONENT_KEYS) {
    const v = c[key];
    if (v != null && v !== '') out[key] = v;
  }
  return out;
}
