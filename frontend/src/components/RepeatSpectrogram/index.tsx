import { useCallback, useEffect, useId, useMemo, useRef, useState, type CSSProperties } from 'react';
import {
  fetchRepeats,
  patchJobTemplateLabel,
  saveRepeatReview,
  type RepeatMatches,
  submitRepeatReview,
  type ReviewSubmitResult,
  type RepeatTemplate,
} from '../../api/repeats';
import { API_BASE } from '../../api/base';
import { remapChunkToTemplateWidth, widthBinsForTemplateNumber } from '../../utils/repeatGrouping';
import { announcePlayback, subscribeToPlayback } from '../../utils/audioCoordinator';
import { ColorScaleBar } from '../SpectrogramViewer/ColorScaleBar';
import { PlayButton, SpectrogramButton } from '../SpectrogramViewer/PlayButton';
import type { SpectrogramScale } from '../SpectrogramViewer/useSpectrogramScale';
import { SPEC_TARGET_H, SpectrogramCanvas } from '../SpectrogramViewer/SpectrogramCanvas';
import type { SpectrogramViewerProps } from '../SpectrogramViewer/types';
import { DistanceCdfPlot, DISTANCE_CDF_PAD, DISTANCE_CDF_PLOT_W } from './DistanceCdfPlot';
import { InlineRename } from '../InlineRename';
import { RepeatActionForm } from './RepeatActionForm';
import { YAxis, YAXIS_W } from '../SpectrogramViewer/YAxis';

const PLAYHEAD_LINE_WIDTH = 3;
const MIN_CHUNK_BOX_PX = 40;
const STORAGE_PREFIX = 'panmp.repeat-session';
const AVG_CANVAS_W = 60;
const SIDE_TITLE_H = 28;
const CDF_INNER_W = DISTANCE_CDF_PLOT_W - DISTANCE_CDF_PAD.left - DISTANCE_CDF_PAD.right;

const sideTitleStyle: CSSProperties = {
  margin: 0,
  fontSize: 11,
  color: '#7c3aed',
  fontWeight: 600,
  textTransform: 'uppercase',
  letterSpacing: '0.06em',
  fontFamily: 'monospace',
  whiteSpace: 'nowrap',
  lineHeight: 1.2,
};
interface Props {
  jobId: string;
  /** Catalog dictionary whose matchings are reviewed. */
  dictionaryId: number;
  source: SpectrogramViewerProps;
  matches: RepeatMatches;
  onShowOriginal: (range: { startCol: number; endCol: number } | null) => void;
  /** Zoom state shared with the original spectrogram above. */
  scale: SpectrogramScale;
  /** Notify parent after submit-rebuild so overlays use the new dictionary. */
  onMatchesChange?: (matches: RepeatMatches) => void;
  /** An Actions run registered a new dictionary: the parent should switch to it and refetch repeats. */
  onDictionaryChanged?: (newDictionaryId: number) => void;
  /** The catalog gained a dictionary: the parent may refresh any pickers listing it. */
  onCatalogRefresh?: () => void;
}

interface ComposedTemplate {
  magnitude: Float32Array | Uint8Array;
  timeCols: number;
  cutoffs: number[];
  /** Distance-to-template for the chunk immediately to the left of each cutoff (index-aligned with `cutoffs`); null when undefined. */
  cutoffDistances: Array<number | null>;
  ranges: Array<{ start: number; end: number }>;
}

function composeTemplate(
  source: SpectrogramViewerProps,
  template: RepeatTemplate | undefined,
): ComposedTemplate {
  if (!template) {
    return {
      magnitude: source.magnitudeIsUint8 ? new Uint8Array() : new Float32Array(),
      timeCols: 0,
      cutoffs: [],
      cutoffDistances: [],
      ranges: [],
    };
  }
  // Per-chunk widths are independent (mixed widths are valid after remapping).
  // Clamp each crop to the source spectrogram so TypedArray.set never throws.
  const widths = template.chunks.map(chunk => {
    const start = Math.max(0, Math.min(source.timeCols, chunk.sourceStartCol));
    const end = Math.max(start, Math.min(source.timeCols, chunk.sourceEndCol));
    return Math.max(1, end - start);
  });
  const starts = template.chunks.map(chunk =>
    Math.max(0, Math.min(source.timeCols - 1, chunk.sourceStartCol)),
  );
  const timeCols = widths.reduce((sum, width) => sum + width, 0);
  const magnitude = source.magnitudeIsUint8
    ? new Uint8Array(source.freqBins * timeCols)
    : new Float32Array(source.freqBins * timeCols);
  const cutoffs: number[] = [];
  const cutoffDistances: Array<number | null> = [];
  const ranges: Array<{ start: number; end: number }> = [];

  let destinationCol = 0;
  template.chunks.forEach((chunk, chunkIndex) => {
    const width = widths[chunkIndex];
    const startCol = starts[chunkIndex];
    const copyWidth = Math.min(width, Math.max(0, source.timeCols - startCol));
    ranges.push({ start: destinationCol, end: destinationCol + width });
    for (let row = 0; row < source.freqBins; row++) {
      const sourceOffset = row * source.timeCols + startCol;
      const destinationOffset = row * timeCols + destinationCol;
      if (copyWidth > 0) {
        magnitude.set(
          source.magnitude.subarray(sourceOffset, sourceOffset + copyWidth),
          destinationOffset,
        );
      }
      // Remaining columns (if any) stay zero — quiet pad when span overhangs.
    }
    destinationCol += width;
    // A line closes out every chunk, including the last (right edge of the strip).
    cutoffs.push(destinationCol);
    cutoffDistances.push(typeof chunk.distanceNorm === 'number' ? chunk.distanceNorm : null);
  });

  return { magnitude, timeCols, cutoffs, cutoffDistances, ranges };
}

/** Exported for unit tests: mixed/clamped widths must not throw. */
export { composeTemplate };

function buildInitialStagedLabels(matches: RepeatMatches): Record<string, number> {
  const initial: Record<string, number> = {};
  for (const template of matches.templates) {
    for (const chunk of template.chunks) {
      if (chunk.stagedTemplateNumber !== chunk.originalTemplateNumber) {
        initial[chunk.id] = chunk.stagedTemplateNumber;
      }
    }
  }
  return initial;
}

function isEditableTarget(target: EventTarget | null): boolean {
  const element = target as HTMLElement | null;
  if (!element) return false;
  if (element.isContentEditable) return true;
  return (
    element.tagName === 'INPUT'
    || element.tagName === 'TEXTAREA'
    || element.tagName === 'SELECT'
  );
}

export function RepeatSpectrogram({
  jobId,
  dictionaryId,
  source,
  matches,
  onShowOriginal,
  scale,
  onMatchesChange,
  onDictionaryChanged,
  onCatalogRefresh,
}: Props) {
  const [activeMatches, setActiveMatches] = useState<RepeatMatches>(matches);
  const [actionsOpen, setActionsOpen] = useState(false);
  const [templateIndex, setTemplateIndex] = useState(0);
  const [selectedChunkIndex, setSelectedChunkIndex] = useState(0);
  const [hasSelectedChunk, setHasSelectedChunk] = useState(false);
  const [linkToOriginal, setLinkToOriginal] = useState(false);
  const [stagedLabels, setStagedLabels] = useState<Record<string, number>>(
    () => buildInitialStagedLabels(matches),
  );
  const [editingChunkId, setEditingChunkId] = useState<string | null>(null);
  const [editingValue, setEditingValue] = useState('');
  const [saveError, setSaveError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [submitResult, setSubmitResult] = useState<ReviewSubmitResult | null>(null);
  const [playing, setPlaying] = useState(false);
  const [volume, setVolume] = useState(8);
  const { pxPerSec, maxFreqHz, hasFreqRange, freqCropFraction } = scale;
  const rootRef = useRef<HTMLDivElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const lineRef = useRef<HTMLDivElement>(null);
  const rafRef = useRef<number | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const gainRef = useRef<GainNode | null>(null);
  const playingRef = useRef(false);
  const nudgeInFlightRef = useRef(false);
  const playerId = `repeats-${useId()}`;
  const storageKey = `${STORAGE_PREFIX}:${jobId}:${dictionaryId}`;
  const setPlaybackActive = useCallback((active: boolean) => {
    playingRef.current = active;
    setPlaying(active);
  }, []);
  const stopPlayheadLoop = useCallback(() => {
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
  }, []);
  // Keep chunks under their original template until submit-rebuild. Only remap
  // width live for the staged destination label.
  const template = activeMatches.templates[templateIndex];
  const displayTemplate = useMemo(() => {
    if (!template) return template;
    return {
      ...template,
      chunks: template.chunks.map(chunk => {
        const staged = stagedLabels[chunk.id] ?? chunk.stagedTemplateNumber;
        const width = widthBinsForTemplateNumber(activeMatches.templates, staged);
        return remapChunkToTemplateWidth(chunk, width);
      }),
    };
  }, [activeMatches.templates, stagedLabels, template]);
  const composed = useMemo(
    () => composeTemplate(source, displayTemplate),
    [source, displayTemplate],
  );
  const averageMagnitude = useMemo(() => {
    const average = template?.average;
    if (!average) return null;
    const binary = atob(average.magnitude);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    return bytes;
  }, [template]);
  const distanceNorms = useMemo(() => {
    if (!displayTemplate) return [];
    return displayTemplate.chunks
      .map(chunk => chunk.distanceNorm)
      .filter((d): d is number => typeof d === 'number' && Number.isFinite(d));
  }, [displayTemplate]);
  const repeatDurationSec = source.audioDurationSec
    ? (composed.timeCols / source.timeCols) * source.audioDurationSec
    : 0;
  const displayWidth = repeatDurationSec
    ? Math.max(1, Math.round(repeatDurationSec * pxPerSec))
    : composed.timeCols;
  const canvasDisplayHeight = SPEC_TARGET_H / freqCropFraction;
  const selectedChunk = displayTemplate?.chunks[selectedChunkIndex];
  const selectedRange = composed.ranges[selectedChunkIndex] ?? null;
  const selectedAudioBounds = useMemo(() => {
    if (!selectedChunk || !source.audioDurationSec) return null;
    return {
      start: (selectedChunk.sourceStartCol / source.timeCols) * source.audioDurationSec,
      end: (selectedChunk.sourceEndCol / source.timeCols) * source.audioDurationSec,
    };
  }, [selectedChunk, source.audioDurationSec, source.timeCols]);

  // Maps audio.currentTime (in the original recording's timeline) onto the
  // selected chunk's span inside the *composed* (stitched) canvas. The line
  // is a PLAYHEAD_LINE_WIDTH-wide, left-anchored border box, so its own
  // rendered edge would otherwise poke past the chunk's right boundary at
  // progress=1 — cap it so it never renders past the highlighted chunk.
  const positionLine = useCallback(() => {
    const audio = audioRef.current;
    const line = lineRef.current;
    if (!audio || !line || !selectedAudioBounds || !selectedRange) return;
    const span = selectedAudioBounds.end - selectedAudioBounds.start;
    const progress = span > 0
      ? Math.max(0, Math.min(1, (audio.currentTime - selectedAudioBounds.start) / span))
      : 0;
    const composedCol = selectedRange.start + progress * (selectedRange.end - selectedRange.start);
    const rawX = (composedCol / composed.timeCols) * displayWidth;
    const maxX = (selectedRange.end / composed.timeCols) * displayWidth - PLAYHEAD_LINE_WIDTH;
    const canvasX = Math.min(rawX, Math.max(0, maxX));
    line.style.transform = `translateX(${canvasX}px)`;
  }, [selectedAudioBounds, selectedRange, composed.timeCols, displayWidth]);

  const syncLine = useCallback(() => {
    positionLine();
    const audio = audioRef.current;
    if (audio && !audio.paused) rafRef.current = requestAnimationFrame(syncLine);
  }, [positionLine]);

  useEffect(() => {
    const audio = new Audio(`${API_BASE}/api/jobs/${jobId}/audio`);
    audio.preload = 'metadata';
    audio.crossOrigin = 'anonymous';
    audioRef.current = audio;

    const context = new AudioContext();
    const sourceNode = context.createMediaElementSource(audio);
    const gain = context.createGain();
    gain.gain.value = 8;
    sourceNode.connect(gain);
    gain.connect(context.destination);
    gainRef.current = gain;

    const handleEnded = () => setPlaybackActive(false);
    audio.addEventListener('ended', handleEnded);
    return () => {
      audio.removeEventListener('ended', handleEnded);
      audio.pause();
      stopPlayheadLoop();
      void context.close();
      audioRef.current = null;
      gainRef.current = null;
    };
  }, [jobId, stopPlayheadLoop, setPlaybackActive]);

  // Keep the playhead aligned with the selected chunk's start whenever the
  // selection or zoom changes, even while paused. Declared after the audio
  // element is created above so audioRef.current is already set on mount.
  useEffect(() => {
    positionLine();
  }, [positionLine]);

  useEffect(() => {
    if (gainRef.current) gainRef.current.gain.value = volume;
  }, [volume]);

  useEffect(() => subscribeToPlayback(activePlayerId => {
    if (activePlayerId === playerId) return;
    audioRef.current?.pause();
    stopPlayheadLoop();
    setPlaybackActive(false);
  }), [playerId, stopPlayheadLoop, setPlaybackActive]);

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio || !selectedAudioBounds) return;
    audio.pause();
    audio.currentTime = selectedAudioBounds.start;
    stopPlayheadLoop();
    setPlaybackActive(false);
  }, [selectedAudioBounds, stopPlayheadLoop, setPlaybackActive]);

  // Always stop and reset to this chunk's start when it finishes — never
  // auto-advance into the next chunk. Space then replays from the start.
  useEffect(() => {
    const audio = audioRef.current;
    if (!audio || !selectedAudioBounds) return;
    const stopAtChunkEnd = () => {
      if (audio.currentTime >= selectedAudioBounds.end) {
        audio.pause();
        audio.currentTime = selectedAudioBounds.start;
        stopPlayheadLoop();
        positionLine();
        setPlaybackActive(false);
      }
    };
    audio.addEventListener('timeupdate', stopAtChunkEnd);
    return () => audio.removeEventListener('timeupdate', stopAtChunkEnd);
  }, [selectedAudioBounds, stopPlayheadLoop, positionLine, setPlaybackActive]);

  useEffect(() => {
    const el = scrollRef.current;
    if (!el || !selectedRange || composed.timeCols <= 0) return;
    const center = ((selectedRange.start + selectedRange.end) / 2 / composed.timeCols) * displayWidth;
    el.scrollTo({ left: center - el.clientWidth / 2, behavior: 'smooth' });
  }, [selectedRange, composed.timeCols, displayWidth]);

  const handlePlayPause = useCallback(() => {
    const audio = audioRef.current;
    if (!audio || !selectedAudioBounds) return;
    if (!playingRef.current) {
      announcePlayback(playerId);
      if (
        audio.currentTime < selectedAudioBounds.start
        || audio.currentTime >= selectedAudioBounds.end
      ) {
        audio.currentTime = selectedAudioBounds.start;
      }
      setPlaybackActive(true);
      void audio.play();
      stopPlayheadLoop();
      rafRef.current = requestAnimationFrame(syncLine);
    } else {
      stopPlayheadLoop();
      audio.pause();
      setPlaybackActive(false);
    }
  }, [selectedAudioBounds, playerId, stopPlayheadLoop, syncLine, setPlaybackActive]);

  useEffect(() => {
    setActiveMatches(matches);
    setStagedLabels(buildInitialStagedLabels(matches));
  }, [matches, jobId]);

  useEffect(() => {
    try {
      const raw = window.localStorage.getItem(storageKey);
      if (!raw) return;
      const parsed = JSON.parse(raw) as { templateIndex?: number };
      if (typeof parsed.templateIndex !== 'number') return;
      setTemplateIndex(Math.max(0, Math.min(matches.templates.length - 1, parsed.templateIndex)));
    } catch {
      // Best-effort persistence only.
    }
  }, [matches.templates.length, storageKey]);

  useEffect(() => {
    try {
      window.localStorage.setItem(storageKey, JSON.stringify({ templateIndex }));
    } catch {
      // Best-effort persistence only.
    }
  }, [storageKey, templateIndex]);

  useEffect(() => {
    const chunkCount = displayTemplate?.chunks.length ?? 0;
    if (chunkCount === 0) {
      setSelectedChunkIndex(0);
      return;
    }
    setSelectedChunkIndex(index => Math.min(chunkCount - 1, Math.max(0, index)));
  }, [displayTemplate?.chunks.length]);

  useEffect(() => {
    setTemplateIndex(index => Math.min(activeMatches.templates.length - 1, Math.max(0, index)));
  }, [activeMatches.templates.length]);

  const chunkLookup = useMemo(() => {
    const lookup = new Map<string, RepeatTemplate['chunks'][number]>();
    for (const item of activeMatches.templates) {
      for (const chunk of item.chunks) lookup.set(chunk.id, chunk);
    }
    return lookup;
  }, [activeMatches.templates]);

  const resolveStagedLabel = useCallback((chunk: RepeatTemplate['chunks'][number]) => (
    stagedLabels[chunk.id] ?? chunk.stagedTemplateNumber
  ), [stagedLabels]);

  const chunkCount = displayTemplate?.chunks.length ?? 0;
  const templateNamed = Boolean(template?.name && template.name.trim());
  const templateProgressLabel = template
    ? `${templateNamed ? template.name : `Template ${template.templateNumber}`} (${templateIndex + 1}/${activeMatches.templates.length})`
    : '';

  const saveTemplateName = useCallback(async (next: string) => {
    if (!template) return;
    const result = await patchJobTemplateLabel(
      jobId,
      template.templateNumber,
      next === '' ? null : next,
      dictionaryId,
    );
    setActiveMatches(prev => ({
      ...prev,
      templates: prev.templates.map(item => (
        item.templateNumber === template.templateNumber
          ? { ...item, name: result.label }
          : item
      )),
    }));
  }, [jobId, dictionaryId, template]);

  const commitReassign = useCallback(async (
    chunk: RepeatTemplate['chunks'][number],
    rawValue: string,
  ) => {
    const parsed = Number(rawValue);
    if (rawValue.trim() === '' || !Number.isInteger(parsed) || parsed < 0) {
      setSaveError('Enter a non-negative whole number.');
      return false;
    }
    const currentLabel = resolveStagedLabel(chunk);
    setEditingChunkId(null);
    if (parsed === currentLabel) return true;

    const previousOverride = stagedLabels[chunk.id];
    const previousStaged = chunk.stagedTemplateNumber;
    setSaveError(null);
    setSubmitError(null);
    // Always keep an explicit local value. Deleting the override when parsed
    // equals originalTemplateNumber used to fall back to a stale
    // chunk.stagedTemplateNumber and looked like the edit "reverted" until refresh.
    setStagedLabels(current => ({ ...current, [chunk.id]: parsed }));
    setActiveMatches(prev => ({
      ...prev,
      templates: prev.templates.map(template => ({
        ...template,
        chunks: template.chunks.map(item => (
          item.id === chunk.id
            ? { ...item, stagedTemplateNumber: parsed }
            : item
        )),
      })),
    }));
    try {
      await saveRepeatReview(jobId, chunk.id, parsed, dictionaryId);
      return true;
    } catch (error) {
      setStagedLabels(current => {
        const currentEffective = current[chunk.id] ?? chunk.originalTemplateNumber;
        if (currentEffective != parsed) return current;
        const next = { ...current };
        if (previousOverride === undefined) {
          delete next[chunk.id];
        } else {
          next[chunk.id] = previousOverride;
        }
        return next;
      });
      setActiveMatches(prev => ({
        ...prev,
        templates: prev.templates.map(template => ({
          ...template,
          chunks: template.chunks.map(item => (
            item.id === chunk.id
              ? { ...item, stagedTemplateNumber: previousStaged }
              : item
          )),
        })),
      }));
      setSaveError(error instanceof Error ? error.message : String(error));
      return false;
    }
  }, [jobId, dictionaryId, resolveStagedLabel, stagedLabels]);

  const flushPendingEdit = useCallback(async () => {
    if (!editingChunkId) return true;
    const chunk = chunkLookup.get(editingChunkId);
    if (!chunk) {
      setEditingChunkId(null);
      return true;
    }
    return commitReassign(chunk, editingValue);
  }, [chunkLookup, commitReassign, editingChunkId, editingValue]);

  const handleSubmitReview = async () => {
    const flushed = await flushPendingEdit();
    if (!flushed) return;
    setSubmitting(true);
    setSubmitError(null);
    try {
      const result = await submitRepeatReview(jobId);
      setSubmitResult(result);
      if (result.skipped) {
        setEditingChunkId(null);
        setEditingValue('');
        return;
      }
      const refreshed = await fetchRepeats(jobId, dictionaryId);
      // Drop pre-rebuild overrides. Chunk ids reuse template-{tid}:samples, so a
      // stale stagedLabels entry from an earlier edit can paint the wrong number
      // until a full page reload rebuilds labels from the new payload.
      setActiveMatches(refreshed);
      setStagedLabels(buildInitialStagedLabels(refreshed));
      onMatchesChange?.(refreshed);
      setTemplateIndex(0);
      setSelectedChunkIndex(0);
      setHasSelectedChunk(false);
      setEditingChunkId(null);
      setEditingValue('');
    } catch (error) {
      setSubmitError(error instanceof Error ? error.message : String(error));
    } finally {
      setSubmitting(false);
    }
  };

  const moveChunk = useCallback(async (offset: number) => {
    const currentTemplate = activeMatches.templates[templateIndex];
    const currentChunkCount = currentTemplate?.chunks.length ?? 0;
    if (currentChunkCount === 0) return;
    const flushed = await flushPendingEdit();
    if (!flushed) return;
    stopPlayheadLoop();
    audioRef.current?.pause();
    setPlaybackActive(false);
    setHasSelectedChunk(true);
    setSelectedChunkIndex(index => Math.min(currentChunkCount - 1, Math.max(0, index + offset)));
  }, [
    flushPendingEdit,
    activeMatches.templates,
    setPlaybackActive,
    stopPlayheadLoop,
    templateIndex,
  ]);

  const moveTemplate = useCallback(async (offset: number) => {
    const flushed = await flushPendingEdit();
    if (!flushed) return;
    setTemplateIndex(index => Math.min(activeMatches.templates.length - 1, Math.max(0, index + offset)));
    setSelectedChunkIndex(0);
    setSubmitError(null);
  }, [activeMatches.templates.length, flushPendingEdit]);

  const appendDigitToSelected = useCallback((digit: string) => {
    const chunk = displayTemplate?.chunks[selectedChunkIndex];
    if (!chunk) return;
    setSubmitResult(null);
    setSaveError(null);
    if (editingChunkId === chunk.id) {
      setEditingValue(current => `${current}${digit}`);
      return;
    }
    setEditingChunkId(chunk.id);
    setEditingValue(digit);
  }, [editingChunkId, selectedChunkIndex, displayTemplate?.chunks]);

  const selectChunk = useCallback(async (chunkIndex: number) => {
    if (editingChunkId !== null) {
      const flushed = await flushPendingEdit();
      if (!flushed) return false;
    }
    setSelectedChunkIndex(chunkIndex);
    setHasSelectedChunk(true);
    return true;
  }, [editingChunkId, flushPendingEdit]);

  const bumpSelectedLabel = useCallback(async (delta: number) => {
    if (nudgeInFlightRef.current) return;
    nudgeInFlightRef.current = true;
    try {
    const chunk = displayTemplate?.chunks[selectedChunkIndex];
    if (!chunk) return;
    const flushed = await flushPendingEdit();
    if (!flushed) return;
    setHasSelectedChunk(true);
    const current = resolveStagedLabel(chunk);
    const next = Math.max(0, current + delta);
    setEditingChunkId(chunk.id);
    setEditingValue(String(next));
    await commitReassign(chunk, String(next));
    } finally {
      nudgeInFlightRef.current = false;
    }
  }, [
    commitReassign,
    flushPendingEdit,
    resolveStagedLabel,
    selectedChunkIndex,
    displayTemplate?.chunks,
  ]);

  useEffect(() => {
    const onWindowKeyDown = (event: KeyboardEvent) => {
      if (event.defaultPrevented) return;
      if (event.metaKey || event.ctrlKey || event.altKey) return;
      if (isEditableTarget(event.target)) return;
      if (/^\d$/.test(event.key)) {
        event.preventDefault();
        appendDigitToSelected(event.key);
        return;
      }
      if (event.key === 'Enter' && editingChunkId !== null) {
        event.preventDefault();
        void flushPendingEdit();
      }
    };
    window.addEventListener('keydown', onWindowKeyDown);
    return () => {
      window.removeEventListener('keydown', onWindowKeyDown);
    };
  }, [appendDigitToSelected, editingChunkId, flushPendingEdit]);

  const handleColumnClick = (column: number) => {
    rootRef.current?.focus({ preventScroll: true });
    const chunkIndex = composed.ranges.findIndex(
      range => column >= range.start && column < range.end,
    );
    if (chunkIndex < 0) return;
    void (async () => {
      const selected = await selectChunk(chunkIndex);
      if (!selected || !linkToOriginal) return;
      const chunk = displayTemplate?.chunks[chunkIndex];
      if (!chunk) return;
      onShowOriginal({ startCol: chunk.sourceStartCol, endCol: chunk.sourceEndCol });
    })();
  };

  const handleActionDictionaryChanged = useCallback((newDictionaryId: number) => {
    onCatalogRefresh?.();
    onDictionaryChanged?.(newDictionaryId);
  }, [onCatalogRefresh, onDictionaryChanged]);

  // Rendered as a sibling of the review root so its keystrokes never reach the
  // review's chunk/playback shortcuts. Kept mounted so progress survives close/reopen.
  const actionForm = (
    <RepeatActionForm
      jobId={jobId}
      dictionaryId={dictionaryId}
      open={actionsOpen}
      currentTemplate={activeMatches.templates[templateIndex]?.templateNumber}
      onDictionaryChanged={handleActionDictionaryChanged}
      onClose={() => setActionsOpen(false)}
    />
  );

  if (activeMatches.templates.length === 0) {
    return (
      <>
        <div className="spectrogram-outer" style={{ textAlign: 'center', color: '#6b7280' }}>
          <p style={{ color: '#7c3aed', fontWeight: 600 }}>REPEATED MATCHES</p>
          <p>No repeat templates were found for this recording.</p>
          <SpectrogramButton onClick={() => setActionsOpen(true)} style={{ padding: '6px 10px' }}>
            Actions
          </SpectrogramButton>
        </div>
        {actionForm}
      </>
    );
  }

  if (!template || !displayTemplate) {
    return null;
  }

  return (
    <>
    <div
      ref={rootRef}
      className="spectrogram-outer"
      tabIndex={0}
      onKeyDown={event => {
        if (isEditableTarget(event.target)) return;
        if (event.key === 'ArrowLeft') {
          event.preventDefault();
          void moveChunk(-1);
        } else if (event.key === 'ArrowRight') {
          event.preventDefault();
          void moveChunk(1);
        } else if (event.key === 'ArrowUp') {
          event.preventDefault();
          void bumpSelectedLabel(1);
        } else if (event.key === 'ArrowDown') {
          event.preventDefault();
          void bumpSelectedLabel(-1);
        } else if (event.code === 'Space') {
          event.preventDefault();
          handlePlayPause();
        } else if (/^\d$/.test(event.key) && !event.metaKey && !event.ctrlKey && !event.altKey) {
          event.preventDefault();
          appendDigitToSelected(event.key);
        } else if (event.key === 'Enter') {
          event.preventDefault();
          void flushPendingEdit();
        }
      }}
      style={{ outline: 'none' }}
    >
      {/* Toolbar centered over the main repeats column (not Average/CDF). */}
      <div style={{ display: 'flex', alignItems: 'flex-start', gap: 10, marginBottom: 10 }}>
        <div
          style={{
            flex: 1,
            minWidth: 0,
            display: 'flex',
            justifyContent: 'center',
          }}
        >
          <div
            style={{
              display: 'flex',
              flexWrap: 'wrap',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '8px 10px',
              maxWidth: '100%',
            }}
          >
            <PlayButton playing={playing} onClick={handlePlayPause} />
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 11, color: '#6b7280' }}>
              <span>🔈</span>
              <input
                type="range"
                min={1}
                max={20}
                step={1}
                value={volume}
                onChange={event => setVolume(Number(event.target.value))}
                style={{ width: 72, accentColor: '#f59e0b' }}
                aria-label="Repeat playback volume"
              />
              <span>🔊</span>
            </div>
            <SpectrogramButton
              disabled={templateIndex === 0}
              onClick={() => {
                void moveTemplate(-1);
              }}
              style={{ padding: '6px 10px' }}
            >
              Prev
            </SpectrogramButton>
            <SpectrogramButton
              disabled={templateIndex === activeMatches.templates.length - 1}
              onClick={() => {
                void moveTemplate(1);
              }}
              style={{ padding: '6px 10px' }}
            >
              Next
            </SpectrogramButton>
            <SpectrogramButton
              aria-pressed={linkToOriginal}
              onClick={() => {
                if (linkToOriginal) onShowOriginal(null);
                setLinkToOriginal(value => !value);
              }}
              style={{
                padding: '6px 10px',
                boxShadow: linkToOriginal ? '0 0 0 3px #7c3aed inset' : undefined,
              }}
            >
              {linkToOriginal ? 'Original: on' : 'Original'}
            </SpectrogramButton>
            {/* Submit-rebuild is not ported yet (backend answers 501). Relabels are
                still saved as you edit; they just do not rebuild the dictionary. */}
            <SpectrogramButton
              disabled
              title="Submit-rebuild is not available yet. Relabels are saved automatically."
              onClick={() => void handleSubmitReview()}
              style={{ padding: '6px 10px' }}
            >
              {submitting ? 'Submitting…' : 'Submit'}
            </SpectrogramButton>
            <SpectrogramButton
              onClick={() => setActionsOpen(true)}
              title="Dictionary actions: randomized builder and action history"
              style={{ padding: '6px 10px' }}
            >
              Actions
            </SpectrogramButton>
            <span style={{ fontSize: 11, color: '#6b7280', display: 'inline-flex', alignItems: 'center', gap: 4 }}>
              {templateNamed ? template.name : `T ${templateIndex + 1}/${activeMatches.templates.length}`}
              {template && (
                <InlineRename
                  value={templateNamed ? (template.name ?? '') : ''}
                  untitled={!templateNamed}
                  ariaLabel="Rename template"
                  onSave={saveTemplateName}
                />
              )}
              {' · '}
              {chunkCount === 0
                ? 'No chunks'
                : `Chunk ${selectedChunkIndex + 1}/${chunkCount}`}
            </span>
            {submitError && (
              <span role="alert" style={{ fontSize: 11, color: '#b91c1c' }}>
                {submitError}
              </span>
            )}
            {submitResult && (
              <span style={{ fontSize: 11, color: '#166534', whiteSpace: 'pre-line' }}>
                {submitResult.skipped
                  ? (submitResult.reason ?? 'No changes to rebuild.')
                  : `Rebuilt ${Object.keys(submitResult.rebuiltLabelWidths).length} label(s)
${submitResult.submittedPositiveCount} positive chunk(s)
${submitResult.excludedNonMatchingCount} non-matching chunk(s) excluded
switched to dictionary ${submitResult.newDictionaryId}`}
              </span>
            )}
          </div>
        </div>
        {template.average && averageMagnitude ? (
          <div
            style={{
              flexShrink: 0,
              width: YAXIS_W + AVG_CANVAS_W + 10 + DISTANCE_CDF_PLOT_W,
            }}
            aria-hidden
          />
        ) : null}
      </div>

      <div style={{ display: 'flex', alignItems: 'flex-start', gap: 10 }}>
        {/* Main repeats spectrogram */}
        <div style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column' }}>
          <div
            style={{
              height: SIDE_TITLE_H,
              display: 'flex',
              alignItems: 'flex-end',
              justifyContent: 'center',
              marginBottom: 6,
            }}
          >
            <p style={{
              margin: 0,
              textAlign: 'center',
              fontSize: 13,
              color: '#7c3aed',
              fontWeight: 600,
              textTransform: 'uppercase',
              letterSpacing: '0.08em',
              fontFamily: 'monospace',
              lineHeight: 1.2,
              display: 'inline-flex',
              alignItems: 'center',
              gap: 6,
            }}
            >
              Repeated matches — {templateProgressLabel}
              {template && (
                <InlineRename
                  value={templateNamed ? (template.name ?? '') : ''}
                  untitled={!templateNamed}
                  ariaLabel="Rename template"
                  onSave={saveTemplateName}
                />
              )}
            </p>
          </div>
          <div style={{ display: 'flex', alignItems: 'flex-start' }}>
            <YAxis
              freqBins={source.freqBins}
              height={SPEC_TARGET_H}
              freqLo={source.freqLo}
              freqHi={hasFreqRange ? maxFreqHz : source.freqHi}
            />
            <div ref={scrollRef} style={{ flex: 1, minWidth: 0, overflowX: 'auto', overflowY: 'hidden', position: 'relative' }}>
              {chunkCount === 0 ? (
                <div
                  style={{
                    height: SPEC_TARGET_H,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    color: '#6b7280',
                    fontSize: 13,
                    border: '1px dashed #d1d5db',
                  }}
                >
                  {`No chunks available in template ${template.templateNumber}`}
                </div>
              ) : (
                <>
                  <SpectrogramCanvas
                    magnitude={composed.magnitude}
                    magnitudeIsUint8={source.magnitudeIsUint8 ?? false}
                    freqBins={source.freqBins}
                    timeCols={composed.timeCols}
                    cutoffs={composed.cutoffs}
                    cutoffDistances={composed.cutoffDistances}
                    scrollRef={scrollRef}
                    brightness={source.brightness ?? 1}
                    normMax={source.normMax}
                    onColClick={handleColumnClick}
                    displayWidth={displayWidth}
                    containerHeight={SPEC_TARGET_H}
                    displayHeight={canvasDisplayHeight}
                    highlightedRange={hasSelectedChunk && selectedRange ? {
                      startCol: selectedRange.start,
                      endCol: selectedRange.end,
                    } : null}
                  />
                  <div
                    ref={lineRef}
                    aria-hidden
                    style={{
                      position: 'absolute',
                      top: 0,
                      left: 0,
                      width: PLAYHEAD_LINE_WIDTH,
                      height: SPEC_TARGET_H,
                      borderLeft: `${PLAYHEAD_LINE_WIDTH}px dashed #86efac`,
                      pointerEvents: 'none',
                      zIndex: 10,
                      willChange: 'transform',
                    }}
                  />
                  <div
                    aria-label="Repeat chunk review bar"
                    style={{ display: 'flex', width: displayWidth, height: 24, marginTop: 4 }}
                  >
                    {displayTemplate.chunks.map((chunk, index) => {
                      const sourceWidth = composed.ranges[index].end - composed.ranges[index].start;
                      const proportionalWidth = (sourceWidth / composed.timeCols) * displayWidth;
                      const width = Math.max(MIN_CHUNK_BOX_PX, proportionalWidth);
                      const displayValue = editingChunkId === chunk.id
                        ? editingValue
                        : String(resolveStagedLabel(chunk));
                      return (
                        <button
                          type="button"
                          key={chunk.id}
                          title={`${chunk.id}: staged label ${resolveStagedLabel(chunk)}`}
                          aria-label={`Chunk ${index + 1}, staged template ${displayValue}`}
                          onClick={() => {
                            void (async () => {
                              const selected = await selectChunk(index);
                              if (!selected) return;
                              setSaveError(null);
                              setSubmitError(null);
                              rootRef.current?.focus({ preventScroll: true });
                            })();
                          }}
                          style={{
                            width,
                            minWidth: MIN_CHUNK_BOX_PX,
                            height: 24,
                            padding: '0 2px',
                            border: '1px solid #d1d5db',
                            background: '#ffffff',
                            outline: hasSelectedChunk && index === selectedChunkIndex ? '3px solid #facc15' : undefined,
                            outlineOffset: -3,
                            fontSize: 11,
                            textAlign: 'center',
                            boxSizing: 'border-box',
                            cursor: 'pointer',
                            userSelect: 'none',
                          }}
                        >
                          {displayValue}
                        </button>
                      );
                    })}
                  </div>
                  <p style={{ textAlign: 'center', fontSize: 11, color: '#374151', margin: '4px 0 0' }}>
                    Relabel stays in place until Submit · click chunk or use ←/→ · type digits · ↑/↓ adjust by 1
                  </p>
                  {saveError && (
                    <p role="alert" style={{ color: '#b91c1c', fontSize: 11, margin: '2px 0' }}>
                      {saveError}
                    </p>
                  )}
                </>
              )}
            </div>
            <ColorScaleBar height={SPEC_TARGET_H} />
          </div>
        </div>

        {/* Average + CDF: titles centered over their visual plots (not the y-axis) */}
        {template.average && averageMagnitude ? (
          <>
            <div style={{ flexShrink: 0, display: 'flex', flexDirection: 'column' }}>
              <div
                style={{
                  display: 'flex',
                  height: SIDE_TITLE_H,
                  marginBottom: 6,
                  alignItems: 'flex-end',
                }}
              >
                <div style={{ width: YAXIS_W, flexShrink: 0 }} aria-hidden />
                <div
                  style={{
                    width: AVG_CANVAS_W,
                    display: 'flex',
                    justifyContent: 'center',
                    alignItems: 'flex-end',
                  }}
                >
                  <p style={sideTitleStyle}>Average</p>
                </div>
              </div>
              <div style={{ display: 'flex', alignItems: 'flex-start' }}>
                <YAxis
                  freqBins={template.average.rows}
                  height={SPEC_TARGET_H}
                  freqLo={source.freqLo}
                  freqHi={hasFreqRange ? maxFreqHz : source.freqHi}
                />
                <SpectrogramCanvas
                  magnitude={averageMagnitude}
                  magnitudeIsUint8
                  freqBins={template.average.rows}
                  timeCols={template.average.cols}
                  cutoffs={[]}
                  brightness={source.brightness ?? 1}
                  normMax={template.average.normMax}
                  displayWidth={AVG_CANVAS_W}
                  containerHeight={SPEC_TARGET_H}
                  displayHeight={canvasDisplayHeight}
                />
              </div>
              <div style={{ display: 'flex' }}>
                <div style={{ width: YAXIS_W, flexShrink: 0 }} aria-hidden />
                <p style={{
                  width: AVG_CANVAS_W,
                  textAlign: 'center',
                  fontSize: 10,
                  color: '#374151',
                  margin: '4px 0 0',
                }}
                >
                  {template.average.rows}×{template.average.cols}
                </p>
              </div>
            </div>

            <div style={{ flexShrink: 0, width: DISTANCE_CDF_PLOT_W, display: 'flex', flexDirection: 'column' }}>
              <div
                style={{
                  display: 'flex',
                  height: SIDE_TITLE_H,
                  marginBottom: 6,
                  alignItems: 'flex-end',
                }}
              >
                <div style={{ width: DISTANCE_CDF_PAD.left, flexShrink: 0 }} aria-hidden />
                <div
                  style={{
                    width: CDF_INNER_W,
                    display: 'flex',
                    justifyContent: 'center',
                    alignItems: 'flex-end',
                  }}
                >
                  <p style={sideTitleStyle}>CDF</p>
                </div>
              </div>
              <DistanceCdfPlot distances={distanceNorms} height={SPEC_TARGET_H} />
              <div style={{ display: 'flex' }}>
                <div style={{ width: DISTANCE_CDF_PAD.left, flexShrink: 0 }} aria-hidden />
                <p style={{
                  width: CDF_INNER_W,
                  textAlign: 'center',
                  fontSize: 10,
                  color: '#374151',
                  margin: '4px 0 0',
                }}
                >
                  {distanceNorms.length > 0
                    ? `n=${distanceNorms.length}`
                    : 'n/a'}
                </p>
              </div>
            </div>
          </>
        ) : null}
      </div>
    </div>
    {actionForm}
    </>
  );
}
