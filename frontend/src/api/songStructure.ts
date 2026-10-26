import { API_BASE } from './base';

export interface SongStructureUnit {
  motif: string;
  song: number;
  theme: number;
  phraseId: number;
  phraseCount: number;
  startSec: number;
  endSec: number;
}

export interface SongStructureSong {
  song: number;
  startSec: number;
  endSec: number;
}

export interface SongStructure {
  wavStartUnix: number;
  units: SongStructureUnit[];
  songs: SongStructureSong[];
}

/** Motif / song / theme / phrase lanes for one job; null when no CSV exists.
 *  `csvId` picks a catalog CSV; without it the CSV named after the WAV is used. */
export async function fetchSongStructure(jobId: string, csvId?: number): Promise<SongStructure | null> {
  const query = csvId === undefined ? '' : `?csvId=${csvId}`;
  const res = await fetch(`${API_BASE}/api/jobs/${jobId}/song-structure${query}`);
  if (res.status === 404) return null;
  if (!res.ok) throw new Error(`Song structure fetch failed: ${res.status}`);
  return res.json() as Promise<SongStructure>;
}
