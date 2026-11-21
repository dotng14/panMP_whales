import { API_BASE } from './base';
import type { JobCreated } from './types';

export interface LibraryRoot {
  id: number;
  label: string;
}

export interface LibraryEntry {
  type: 'dir' | 'file';
  name: string;
  // dirs have path, files have fileId
  path?: string;
  fileId?: string;
  sizeBytes?: number;
  mtime?: string;
  // files: true when a pre-processed .gpl cache exists (loads instantly)
  cached?: boolean;
}

export interface BrowseResponse {
  root: number;
  path: string;
  parentPath: string;
  total: number;
  offset: number;
  limit: number;
  entries: LibraryEntry[];
}

export interface BrowseParams {
  root?: number;
  path?: string;
  q?: string;
  sort?: 'name' | 'mtime' | 'size';
  order?: 'asc' | 'desc';
  offset?: number;
  limit?: number;
  recursive?: boolean;
}

export async function getLibraryRoots(): Promise<LibraryRoot[]> {
  const res = await fetch(`${API_BASE}/api/library/roots`);
  if (!res.ok) throw new Error(`Roots fetch failed: ${res.status}`);
  const data = await res.json();
  return data.roots as LibraryRoot[];
}

export async function browseLibrary(params: BrowseParams = {}): Promise<BrowseResponse> {
  const q = new URLSearchParams();
  if (params.root !== undefined) q.set('root', String(params.root));
  if (params.path !== undefined) q.set('path', params.path);
  if (params.q) q.set('q', params.q);
  if (params.sort) q.set('sort', params.sort);
  if (params.order) q.set('order', params.order);
  if (params.offset !== undefined) q.set('offset', String(params.offset));
  if (params.limit !== undefined) q.set('limit', String(params.limit));
  if (params.recursive) q.set('recursive', 'true');

  const res = await fetch(`${API_BASE}/api/library/browse?${q}`);
  if (!res.ok) throw new Error(`Browse failed: ${res.status}`);
  return res.json() as Promise<BrowseResponse>;
}

export async function createJobFromLibrary(fileId: string): Promise<JobCreated> {
  const res = await fetch(`${API_BASE}/api/jobs/from-library`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ fileId }),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Job create failed (${res.status}): ${text}`);
  }
  return res.json() as Promise<JobCreated>;
}

/** Component filters for facets / match (SOCAL fields; omit empty). */
export type RecordingFilters = Partial<{
  family: string;
  site: string;
  deployment: string;
  disk: string;
  date: string;
  time: string;
  df: string;
  channel: string;
  resample: string;
  q: string;
}>;

export interface FacetValue {
  value: string;
  count: number;
}

export interface FacetsResponse {
  facets: Record<string, FacetValue[]>;
}

export interface MatchEntry {
  type: 'file';
  name: string;
  fileId: string;
  sizeBytes?: number;
  mtime?: string;
  cached?: boolean;
  stem?: string;
  components?: Record<string, string | null>;
}

export interface MatchResponse {
  total: number;
  offset: number;
  limit: number;
  entries: MatchEntry[];
}

function filtersToParams(filters: RecordingFilters): URLSearchParams {
  const q = new URLSearchParams();
  for (const [key, value] of Object.entries(filters)) {
    if (value != null && value !== '') q.set(key, value);
  }
  return q;
}

async function raiseIfBad(res: Response, label: string): Promise<void> {
  if (res.ok) return;
  let detail = '';
  try {
    const body = await res.json();
    detail = typeof body?.detail === 'string' ? body.detail : JSON.stringify(body?.detail ?? body);
  } catch {
    try { detail = await res.text(); } catch { /* ignore */ }
  }
  throw new Error(detail ? `${label} failed: ${res.status} ${detail}` : `${label} failed: ${res.status}`);
}

export async function getFacets(filters: RecordingFilters = {}): Promise<FacetsResponse> {
  const q = filtersToParams(filters);
  const res = await fetch(`${API_BASE}/api/library/facets?${q}`);
  await raiseIfBad(res, 'Facets fetch');
  return res.json() as Promise<FacetsResponse>;
}

export async function matchRecordings(
  filters: RecordingFilters & { offset?: number; limit?: number } = {},
): Promise<MatchResponse> {
  const { offset, limit, ...rest } = filters;
  const q = filtersToParams(rest);
  if (offset !== undefined) q.set('offset', String(offset));
  if (limit !== undefined) q.set('limit', String(limit));
  const res = await fetch(`${API_BASE}/api/library/match?${q}`);
  await raiseIfBad(res, 'Match fetch');
  return res.json() as Promise<MatchResponse>;
}
