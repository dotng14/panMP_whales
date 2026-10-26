import { API_BASE } from './base';

interface CatalogItemBase {
  id: number;
  name: string;
  /** User alias; null when unnamed. */
  label: string | null;
  /** True only when `label` is non-blank (the automatic `name` does not count). */
  named: boolean;
  displayName: string;
  createdAt: string | null;
}

/** A randomized-build (`seed`) or matching-pursuit (`matching`) run in the catalog. */
export interface CatalogDictionary extends CatalogItemBase {
  kind: 'dictionary';
  description: string | null;
  source: 'seed' | 'refinement' | 'matching' | 'edit';
  parentId: number | null;
  nTemplates: number;
  nWindows: number;
  nMatchings: number;
  timeCoverage: number | null;
}

/** A song-structure CSV (`*.motif_segments.csv`) in the catalog. */
export interface CatalogCsv extends CatalogItemBase {
  kind: 'csv';
  nRows: number;
  nSongs: number;
}

export type CatalogItem = CatalogDictionary | CatalogCsv;

/** What the viewer overlays: a catalog item, or nothing (spectrogram only). */
export type AnalysisChoice = { kind: CatalogItem['kind']; id: number; label: string } | null;

export interface FileAnalysesResponse {
  stem: string;
  items: CatalogItem[];
}

/** Dictionaries and CSVs in the catalog for one library WAV. */
export async function getFileAnalyses(fileId: string): Promise<FileAnalysesResponse> {
  const res = await fetch(`${API_BASE}/api/library/analyses?fileId=${encodeURIComponent(fileId)}`);
  if (!res.ok) throw new Error(`Analysis list failed: ${res.status}`);
  return res.json() as Promise<FileAnalysesResponse>;
}

/** Set (or, with null, clear) the user label of a dictionary or CSV. */
export async function patchCatalogLabel(
  kind: CatalogItem['kind'],
  id: number,
  label: string | null,
): Promise<CatalogItem> {
  const path = kind === 'dictionary' ? 'dictionaries' : 'csvs';
  const res = await fetch(`${API_BASE}/api/catalog/${path}/${id}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ label }),
  });
  if (!res.ok) throw new Error(`Rename failed: ${res.status}`);
  return res.json() as Promise<CatalogItem>;
}

export interface Placement {
  matchingId: number;
  templateId: number;
  /** User alias of the atom within this dictionary. */
  label: string | null;
  startSec: number;
  endSec: number;
  amplitude: number;
  feedback: 'good' | 'bad' | null;
}

export interface DictionaryPlacements {
  dictionary: CatalogDictionary;
  placements: Placement[];
  /** Placements whose GPL window was not found, so they have no time. */
  nWithoutTime: number;
}

/** A dictionary's MP placements in seconds from the WAV start. */
export async function fetchDictionaryPlacements(dictionaryId: number): Promise<DictionaryPlacements> {
  const res = await fetch(`${API_BASE}/api/catalog/dictionaries/${dictionaryId}/placements`);
  if (!res.ok) throw new Error(`Placements fetch failed: ${res.status}`);
  return res.json() as Promise<DictionaryPlacements>;
}
