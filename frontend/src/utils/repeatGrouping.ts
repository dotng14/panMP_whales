import type { RepeatChunk, RepeatTemplate, RepeatTemplateAverage } from '../api/repeats';

const TEMPLATE_ID_RE = /^template-(\d+)$/;

function parseTemplateNumber(templateId: string): number {
  const match = TEMPLATE_ID_RE.exec(templateId);
  if (!match) throw new Error(`invalid template id: ${templateId}`);
  return Number(match[1]);
}

function effectiveBucket(
  chunk: RepeatChunk,
  overrides: Record<string, number>,
): number {
  if (chunk.id in overrides) return overrides[chunk.id];
  return chunk.stagedTemplateNumber ?? chunk.originalTemplateNumber;
}

function resolveWidthBins(
  meta: Pick<RepeatTemplate, 'widthBins' | 'average'> | undefined,
): number | null {
  if (!meta) return null;
  if (meta.widthBins != null && meta.widthBins >= 1) return meta.widthBins;
  const cols = meta.average?.cols;
  if (cols != null && cols >= 1) return cols;
  return null;
}

/**
 * Resize chunk span to destination template widthBins (same start anchors).
 * Must match backend/app/services/repeat_reviews.py::remap_chunk_to_template_width.
 */
export function remapChunkToTemplateWidth(
  chunk: RepeatChunk,
  widthBins: number | null,
): RepeatChunk {
  if (widthBins == null || widthBins < 1) return chunk;
  const timeBinStart = chunk.timeBinStart;
  const windowBinEnd = chunk.windowBinEnd;
  const overlap = chunk.overlap;
  const strideRaw = chunk.stride;
  if (
    timeBinStart == null
    || windowBinEnd == null
    || overlap == null
    || strideRaw == null
  ) {
    return chunk;
  }
  const stride = Math.max(1, strideRaw);
  const newEndBin = Math.min(windowBinEnd, timeBinStart + widthBins);
  if (newEndBin <= timeBinStart) return chunk;

  const startCol = chunk.sourceStartCol;
  let endCol = Math.max(startCol + 1, Math.ceil(newEndBin / stride));
  if (chunk.displayCols != null) {
    endCol = Math.min(chunk.displayCols, endCol);
  }

  const spanBins = newEndBin - timeBinStart;
  let sourceEndSample = chunk.sourceStartSample + spanBins * overlap - 1;
  if (chunk.windowEndSample != null) {
    sourceEndSample = Math.min(chunk.windowEndSample, sourceEndSample);
  }

  return {
    ...chunk,
    sourceEndCol: endCol,
    sourceEndSample,
  };
}

type CatalogMeta = {
  sourceSegmentId: string | null;
  average: RepeatTemplateAverage | null;
  widthBins: number | null;
};

/**
 * Regroup templates by effective bucket.
 * Width remapping matches backend stamp_reviews_onto_templates behavior.
 * Empty original template numbers stay as navigable slots.
 */
export function groupChunksByEffectiveTemplate(
  templates: RepeatTemplate[],
  overrides: Record<string, number>,
  pinnedTemplateNumbers: number[] = [],
): RepeatTemplate[] {
  const catalogMeta = new Map<number, CatalogMeta>();
  const allChunks: RepeatChunk[] = [];

  for (const template of templates) {
    const tid = template.templateNumber ?? parseTemplateNumber(template.id);
    const widthBins = template.widthBins ?? resolveWidthBins(template);
    catalogMeta.set(tid, {
      sourceSegmentId: template.sourceSegmentId,
      average: template.average,
      widthBins,
    });
    for (const chunk of template.chunks) {
      allChunks.push({
        ...chunk,
        originalTemplateNumber: chunk.originalTemplateNumber ?? tid,
      });
    }
  }

  // Keep every catalog template number as a navigable slot — including empties
  // after all of its chunks are reassigned (e.g. move all of 2 → 1, keep empty 2).
  // pinnedTemplateNumbers survives even if the input list temporarily drops a row.
  const originalNumbers = [...new Set([
    ...templates.map(t => t.templateNumber ?? parseTemplateNumber(t.id)),
    ...pinnedTemplateNumbers,
  ])].sort((a, b) => a - b);
  const originalSet = new Set(originalNumbers);

  for (const n of originalNumbers) {
    if (!catalogMeta.has(n)) {
      catalogMeta.set(n, { sourceSegmentId: null, average: null, widthBins: null });
    }
  }

  const buckets = new Map<number, RepeatChunk[]>();
  for (const n of originalNumbers) buckets.set(n, []);

  for (const chunk of allChunks) {
    const bucket = effectiveBucket(chunk, overrides);
    const width = resolveWidthBins(catalogMeta.get(bucket));
    const remapped = remapChunkToTemplateWidth(chunk, width);
    if (!buckets.has(bucket)) buckets.set(bucket, []);
    buckets.get(bucket)!.push(remapped);
  }

  const syntheticNumbers = [...buckets.keys()]
    .filter(n => !originalSet.has(n))
    .sort((a, b) => a - b);
  const leadingZero = syntheticNumbers.includes(0) ? [0] : [];
  const trailingSynthetic = syntheticNumbers.filter(n => n !== 0);
  const navigable = [...leadingZero, ...originalNumbers, ...trailingSynthetic];

  return navigable.map(number => {
    const meta = catalogMeta.get(number);
    return {
      id: `template-${number}`,
      templateNumber: number,
      sourceSegmentId: meta?.sourceSegmentId ?? null,
      widthBins: meta?.widthBins ?? null,
      chunks: buckets.get(number) ?? [],
      average: meta?.average ?? null,
    };
  });
}

/** Resolve destination widthBins for a staged template number from catalog templates. */
export function widthBinsForTemplateNumber(
  templates: RepeatTemplate[],
  templateNumber: number,
): number | null {
  const match = templates.find(t => t.templateNumber === templateNumber);
  return resolveWidthBins(match);
}
