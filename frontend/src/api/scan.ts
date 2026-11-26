import { API_BASE } from './base';

export interface ScanMonth {
  month: string;
  imageUrl: string;
  recordings?: number;
  hours?: number;
  minutes?: number;
  meanEnergyRatio?: number;
  meanFftFraction?: number;
}

export interface ScanOverview {
  id: string;
  label: string;
  imageUrl: string;
}

export interface ScanSummary {
  available: boolean;
  months: ScanMonth[];
  overviews: ScanOverview[];
}

export async function getScanSummary(): Promise<ScanSummary> {
  const res = await fetch(`${API_BASE}/api/scan/summary`);
  if (!res.ok) throw new Error(`Scan summary fetch failed: ${res.status}`);
  return res.json() as Promise<ScanSummary>;
}

export function scanImageUrl(path: string): string {
  return `${API_BASE}${path}`;
}

export interface HybridMotif {
  id: number;
  widthCols: number;
  durationSec: number;
  imageUrl: string;
  zoomImageUrl: string;
}

export interface HybridDictionary {
  available: boolean;
  model?: string;
  nRows?: number;
  freqLo?: number;
  freqHi?: number;
  motifs: HybridMotif[];
}

export async function getHybridDictionary(): Promise<HybridDictionary> {
  const res = await fetch(`${API_BASE}/api/scan/hybrid-dictionary`);
  if (!res.ok) throw new Error(`Hybrid dictionary fetch failed: ${res.status}`);
  return res.json() as Promise<HybridDictionary>;
}

