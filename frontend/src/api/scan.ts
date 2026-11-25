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
