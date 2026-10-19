/** Stable colors for song-structure annotation lanes. */

const MOTIF_PALETTE = [
  '#f97316', '#22c55e', '#3b82f6', '#eab308', '#a855f7',
  '#ef4444', '#14b8a6', '#f472b6', '#84cc16', '#06b6d4',
  '#fb923c', '#4ade80', '#60a5fa', '#facc15', '#c084fc',
];

const THEME_PALETTE = [
  '#0ea5e9', '#f59e0b', '#10b981', '#8b5cf6', '#f43f5e',
  '#64748b', '#d946ef', '#0891b2',
];

const PHRASE_PALETTE = [
  '#94a3b8', '#a78bfa', '#34d399', '#fb7185', '#38bdf8',
  '#fbbf24', '#2dd4bf', '#e879f9',
];

const SONG_PALETTE = [
  '#1d4ed8', '#b45309', '#047857', '#7c3aed', '#be123c',
  '#0369a1', '#a16207', '#15803d',
];

function hashLabel(label: string): number {
  let h = 0;
  for (let i = 0; i < label.length; i++) {
    h = (h * 31 + label.charCodeAt(i)) >>> 0;
  }
  return h;
}

export function motifColor(motif: string): string {
  return MOTIF_PALETTE[hashLabel(motif) % MOTIF_PALETTE.length];
}

export function themeColor(theme: number): string {
  return THEME_PALETTE[Math.abs(theme) % THEME_PALETTE.length];
}

export function phraseColor(phraseId: number): string {
  return PHRASE_PALETTE[Math.abs(phraseId) % PHRASE_PALETTE.length];
}

export function songColor(song: number): string {
  return SONG_PALETTE[Math.abs(song) % SONG_PALETTE.length];
}
