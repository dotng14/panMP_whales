import { API_BASE } from './base';

export interface RepeatChunk {
  id: string;
  sourceStartCol: number;
  sourceEndCol: number;
  sourceStartSample: number;
  sourceEndSample: number;
  windowId: string;
  windowStartCol: number;
  amplitude: number;
  originalTemplateNumber: number;
  stagedTemplateNumber: number;
  /** Global pre-stride bin index of matching t_start (for width remapping). */
  timeBinStart?: number;
  /** Exclusive end of the window in pre-stride bins. */
  windowBinEnd?: number;
  /** Exclusive-ish clamp for sample end within the window. */
  windowEndSample?: number;
  overlap?: number;
  stride?: number;
  displayCols?: number;
  /** Unexplained energy fraction vs catalog atom in [0, 1]; omitted if undefined. */
  distanceNorm?: number;
}

export interface RepeatTemplateAverage {
  rows: number;
  cols: number;
  durationSec: number;
  normMax: number;
  /** Base64-encoded uint8 magnitude, row-major, length rows*cols. */
  magnitude: string;
}

export interface RepeatTemplate {
  id: string;
  templateNumber: number;
  /** User alias; null/undefined when unnamed. */
  name?: string | null;
  sourceSegmentId: string | null;
  /** Atom width in STFT time bins; null for synthetic buckets. */
  widthBins?: number | null;
  chunks: RepeatChunk[];
  average: RepeatTemplateAverage | null;
}

export interface RepeatMatches {
  version: number;
  templates: RepeatTemplate[];
}

export async function fetchRepeats(jobId: string, dictionaryId: number): Promise<RepeatMatches> {
  const response = await fetch(
    `${API_BASE}/api/jobs/${jobId}/repeats?dictionaryId=${encodeURIComponent(dictionaryId)}`,
  );
  if (!response.ok) throw new Error(`Repeat matches fetch failed: ${response.status}`);
  return response.json() as Promise<RepeatMatches>;
}

export async function patchJobTemplateLabel(
  jobId: string,
  templateNumber: number,
  label: string | null,
  dictionaryId: number,
): Promise<{ label: string | null; named: boolean }> {
  const response = await fetch(
    `${API_BASE}/api/jobs/${jobId}/templates/${templateNumber}?dictionaryId=${encodeURIComponent(dictionaryId)}`,
    {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ label }),
    },
  );
  if (!response.ok) {
    const detail = await response.json().catch(() => null);
    throw new Error(detail?.detail ?? `Template rename failed: ${response.status}`);
  }
  return response.json() as Promise<{ label: string | null; named: boolean }>;
}

export async function saveRepeatReview(
  jobId: string,
  chunkId: string,
  targetTemplateId: number,
  dictionaryId: number,
): Promise<void> {
  const response = await fetch(
    `${API_BASE}/api/jobs/${jobId}/repeat-reviews?dictionaryId=${encodeURIComponent(dictionaryId)}`,
    {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ chunkId, targetTemplateId }),
    },
  );
  if (!response.ok) throw new Error(`Repeat review save failed: ${response.status}`);
}

export interface ReviewSubmitResult {
  skipped?: boolean;
  reason?: string;
  currentDictionaryId?: number | null;
  newDictionaryId: number;
  outputDir: string;
  excludedNonMatchingCount: number;
  submittedPositiveCount: number;
  rebuiltLabelWidths: Record<string, number>;
  rebuiltMatchingsPerLabel: Record<string, number>;
  iterationsCompleted: number;
  energyBefore: number | null;
  energyAfter: number | null;
}

/** Submit-rebuild is not ported yet: the backend answers 501 with a detail message. */
export async function submitRepeatReview(jobId: string): Promise<ReviewSubmitResult> {
  const response = await fetch(`${API_BASE}/api/jobs/${jobId}/review-submit`, {
    method: 'POST',
  });
  if (!response.ok) {
    const detail = await response.json().catch(() => null);
    throw new Error(detail?.detail ?? `Review submit failed: ${response.status}`);
  }
  return response.json() as Promise<ReviewSubmitResult>;
}
