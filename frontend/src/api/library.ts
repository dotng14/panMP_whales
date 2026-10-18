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
