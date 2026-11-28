import { memo, useEffect, useRef, useState, type RefObject } from 'react';
import { API_BASE } from '../../api/base';
import { JET, JET_RGBA, magnitudeToIndex } from '../../utils/colormap';
import { repeatOverlayColor } from './repeatOverlayColors';
import type { RepeatOverlay } from './types';

export const SPEC_TARGET_H = 400;

/** Browsers reject canvases near ~16k–32k px; stay well under that (matches LaneCanvas). */
const MAX_CANVAS_CSS_W = 8192;

/** High-res tile length: 610 frames of 0.12288 s, matching backend TILE_FRAMES (frame-aligned, seamless). */
const TILE_SEC = (610 * 1536) / 12500;
const MAX_CACHED_TILES = 48;
const MAX_CONCURRENT_TILE_FETCHES = 6;
const MAX_HIRES_SPAN_SEC = 600;
/** Prefetch covers this many seconds of audio at the current scroll/playback speed. */
const LOOKAHEAD_SEC = 4;
const MIN_AHEAD_TILES = 2;
const MAX_AHEAD_TILES = 12;
/** Treat scroll as stopped after this many ms without movement (drops lookahead to minimum). */
const IDLE_AFTER_MS = 400;

interface HiResTile {
  startSec: number;
  endSec: number;
  freqBins: number;
  canvas: HTMLCanvasElement;
}

function buildTileCanvas(mag: Uint8Array, freqBins: number, timeCols: number): HTMLCanvasElement {
  const off = document.createElement('canvas');
  off.width = timeCols;
  off.height = freqBins;
  const ctx = off.getContext('2d');
  if (!ctx) return off;
  const imageData = ctx.createImageData(timeCols, freqBins);
  const out = imageData.data;
  for (let row = 0; row < freqBins; row++) {
    const canvasRow = freqBins - 1 - row;
    const rowBase = canvasRow * timeCols;
    const magBase = row * timeCols;
    for (let col = 0; col < timeCols; col++) {
      const jetOff = mag[magBase + col] * 4;
      const px = (rowBase + col) * 4;
      out[px] = JET_RGBA[jetOff];
      out[px + 1] = JET_RGBA[jetOff + 1];
      out[px + 2] = JET_RGBA[jetOff + 2];
      out[px + 3] = 255;
    }
  }
  ctx.putImageData(imageData, 0, 0);
  return off;
}

const NO_LINES: { col: number; color: string }[] = [];
const NO_CUTOFFS: number[] = [];

/** White at distance 0 → red at distance 1. */
function cutoffColor(distance: number | null | undefined): string {
  if (distance == null || !Number.isFinite(distance)) return 'rgba(255,255,255,0.9)';
  const d = Math.max(0, Math.min(1, distance));
  const gb = Math.round(255 * (1 - d));
  return `rgb(255,${gb},${gb})`;
}

interface Props {
  magnitude: Float32Array | Uint8Array;
  magnitudeIsUint8: boolean;
  freqBins: number;
  timeCols: number;
  /** Song/theme/phrase/motif boundary markers, colored to match their lane row. */
  structureLines?: { col: number; color: string }[];
  /** Column boundaries (right edge of each chunk) for stitched repeat strips. */
  cutoffs?: number[];
  /** Per-cutoff distance-to-template in [0, 1] (null = undefined); tints the cutoff line. */
  cutoffDistances?: Array<number | null>;
  /** Column range outlined as the current selection. */
  highlightedRange?: { startCol: number; endCol: number } | null;
  /** Colored boxes (one per matched repeat chunk) drawn as DOM overlays. */
  repeatOverlays?: RepeatOverlay[];
  /** Scroll container the lines overlay syncs to, so it only draws the visible slice. */
  scrollRef?: RefObject<HTMLDivElement | null>;
  brightness: number;
  normMax?: number;
  onColClick?: (col: number) => void;
  /** CSS width in px (horizontal zoom); defaults to native timeCols (1px/column) */
  displayWidth?: number;
  /** Visible height in px (the clipped viewport); defaults to SPEC_TARGET_H */
  containerHeight?: number;
  /** CSS height in px the canvas is stretched to before being clipped by containerHeight (frequency zoom); defaults to containerHeight */
  displayHeight?: number;
  /** Job id + audio duration: when panel.bin is decimated (>0.25 s/col), fetch full-res slices for the visible range. */
  jobId?: string;
  audioDurationSec?: number;
}

export const SpectrogramCanvas = memo(function SpectrogramCanvas({
  magnitude,
  magnitudeIsUint8,
  freqBins,
  timeCols,
  structureLines = NO_LINES,
  cutoffs = NO_CUTOFFS,
  cutoffDistances,
  highlightedRange = null,
  repeatOverlays,
  scrollRef,
  brightness,
  normMax,
  onColClick,
  displayWidth,
  containerHeight,
  displayHeight,
  jobId,
  audioDurationSec,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const hiResCanvasRef = useRef<HTMLCanvasElement>(null);
  const linesCanvasRef = useRef<HTMLCanvasElement>(null);
  const tileCacheRef = useRef<Map<number, HiResTile>>(new Map());
  const inFlightRef = useRef<Map<number, AbortController>>(new Map());
  const failedTilesRef = useRef<Set<number>>(new Set());
  const cssWidth = displayWidth ?? timeCols;
  const cssContainerHeight = containerHeight ?? SPEC_TARGET_H;
  const cssHeight = displayHeight ?? cssContainerHeight;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const w = timeCols;
    canvas.width = w;
    canvas.height = freqBins;

    const imageData = ctx.createImageData(w, freqBins);
    const data = imageData.data;

    if (magnitudeIsUint8) {
      const mag = magnitude as Uint8Array;
      for (let row = 0; row < freqBins; row++) {
        const canvasRow = freqBins - 1 - row;
        const rowBase = canvasRow * w;
        const magBase = row * timeCols;
        for (let col = 0; col < timeCols; col++) {
          const jetOff = mag[magBase + col] * 4;
          const px = (rowBase + col) * 4;
          data[px] = JET_RGBA[jetOff];
          data[px + 1] = JET_RGBA[jetOff + 1];
          data[px + 2] = JET_RGBA[jetOff + 2];
          data[px + 3] = 255;
        }
      }
    } else {
      let maxVal = normMax ?? 0;
      if (!maxVal) {
        const mag = magnitude as Float32Array;
        for (let i = 0; i < mag.length; i++) {
          if (mag[i] > maxVal) maxVal = mag[i];
        }
      }
      if (maxVal === 0) maxVal = 1;
      const mag = magnitude as Float32Array;
      for (let row = 0; row < freqBins; row++) {
        const canvasRow = freqBins - 1 - row;
        const rowOffset = canvasRow * w;
        for (let col = 0; col < timeCols; col++) {
          const idx = magnitudeToIndex(mag[row * timeCols + col], maxVal, brightness);
          const [r, g, b] = JET[idx];
          const px = (rowOffset + col) * 4;
          data[px] = r;
          data[px + 1] = g;
          data[px + 2] = b;
          data[px + 3] = 255;
        }
      }
    }

    ctx.putImageData(imageData, 0, 0);
  }, [magnitude, magnitudeIsUint8, freqBins, timeCols, brightness, normMax]);

  // Structure boundary lines on a separate overlay canvas, sized in CSS pixels
  // (not native timeCols) and windowed to the visible scroll range — same
  // technique as LaneCanvas below. A native-resolution canvas can't produce a
  // crisp zoom-independent line (one data column alone can span 15+ CSS px
  // once stretched), and drawing every line across the full (possibly
  // million-pixel-wide) displayWidth is both slow and exceeds canvas limits.
  const [scrollLeft, setScrollLeft] = useState(0);
  const [viewWidth, setViewWidth] = useState(800);
  useEffect(() => {
    const el = scrollRef?.current;
    if (!el) return;
    const sync = () => {
      setScrollLeft(el.scrollLeft);
      setViewWidth(Math.max(1, el.clientWidth));
    };
    sync();
    el.addEventListener('scroll', sync, { passive: true });
    const ro = new ResizeObserver(sync);
    ro.observe(el);
    return () => {
      el.removeEventListener('scroll', sync);
      ro.disconnect();
    };
  }, [scrollRef, cssWidth]);

  const overlayCssW = Math.max(1, Math.min(scrollRef ? viewWidth : cssWidth, MAX_CANVAS_CSS_W));

  // When panel.bin is heavily decimated (e.g. 22h hybrid file at 1.35 s/col),
  // fetch a full-resolution (0.123 s/col) slice covering the visible viewport.
  const isDecimated = Boolean(
    jobId && audioDurationSec && timeCols > 0 && audioDurationSec / timeCols > 0.25,
  );

  // Reset tile cache and abort in-flight fetches when switching recordings.
  useEffect(() => {
    for (const ac of inFlightRef.current.values()) ac.abort();
    inFlightRef.current.clear();
    tileCacheRef.current.clear();
    failedTilesRef.current.clear();
  }, [jobId]);

  // Stream & prefetch 75-second full-res tiles as the user scrolls or listens,
  // painting cached tiles directly onto a viewport-sized overlay canvas with
  // zero React-state lag and without aborting visible/upcoming fetches mid-scroll.
  useEffect(() => {
    const canvas = hiResCanvasRef.current;
    if (!isDecimated || !jobId || !audioDurationSec || cssWidth <= 0) {
      for (const ac of inFlightRef.current.values()) ac.abort();
      inFlightRef.current.clear();
      if (canvas) {
        const ctx = canvas.getContext('2d');
        ctx?.clearRect(0, 0, canvas.width, canvas.height);
      }
      return;
    }

    let disposed = false;
    // Scroll-motion state (per effect run): speed/direction drive the lookahead.
    let velSecPerSec = 0;
    let scrollDir: 1 | -1 = 1;
    let lastLeft: number | null = null;
    let lastT = 0;
    let lastMoveT = 0;
    const currentSpeed = () =>
      performance.now() - lastMoveT < IDLE_AFTER_MS ? Math.abs(velSecPerSec) : 0;
    const maxTile = Math.max(0, Math.floor(Math.max(0, audioDurationSec - 1e-3) / TILE_SEC));

    const getViewport = () => {
      const el = scrollRef?.current;
      const left = el ? el.scrollLeft : 0;
      const width = el ? Math.max(1, el.clientWidth) : Math.max(1, cssWidth);
      return { left, width };
    };

    const paint = (curLeft: number, curWidth: number) => {
      const c = hiResCanvasRef.current;
      if (!c) return;
      const w = Math.max(1, Math.min(Math.ceil(curWidth), MAX_CANVAS_CSS_W));
      if (c.width !== w) c.width = w;
      if (c.height !== freqBins) c.height = freqBins;
      c.style.left = `${curLeft}px`;
      c.style.width = `${w}px`;

      const ctx = c.getContext('2d');
      if (!ctx) return;
      ctx.clearRect(0, 0, w, freqBins);

      const visStart = Math.max(0, (curLeft / cssWidth) * audioDurationSec);
      const visEnd = Math.min(audioDurationSec, ((curLeft + w) / cssWidth) * audioDurationSec);
      const span = visEnd - visStart;
      if (span <= 0 || span > MAX_HIRES_SPAN_SEC) return;

      const firstTile = Math.max(0, Math.floor(visStart / TILE_SEC));
      const lastTile = Math.min(maxTile, Math.floor(Math.max(visStart, visEnd - 1e-4) / TILE_SEC));
      const cache = tileCacheRef.current;

      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = 'high';
      for (let k = firstTile; k <= lastTile; k++) {
        const tile = cache.get(k);
        if (!tile) continue;
        // Touch LRU
        cache.delete(k);
        cache.set(k, tile);
        const x0 = (tile.startSec / audioDurationSec) * cssWidth - curLeft;
        const x1 = (tile.endSec / audioDurationSec) * cssWidth - curLeft;
        ctx.drawImage(tile.canvas, x0, 0, x1 - x0, freqBins);
      }
    };

    const schedule = (curLeft: number, curWidth: number) => {
      if (disposed) return;
      const w = Math.max(1, Math.min(Math.ceil(curWidth), MAX_CANVAS_CSS_W));
      const visStart = Math.max(0, (curLeft / cssWidth) * audioDurationSec);
      const visEnd = Math.min(audioDurationSec, ((curLeft + w) / cssWidth) * audioDurationSec);
      const span = visEnd - visStart;
      const inFlight = inFlightRef.current;

      if (span <= 0 || span > MAX_HIRES_SPAN_SEC) {
        for (const ac of inFlight.values()) ac.abort();
        inFlight.clear();
        return;
      }

      const firstVis = Math.max(0, Math.floor(visStart / TILE_SEC));
      const lastVis = Math.min(maxTile, Math.floor(Math.max(visStart, visEnd - 1e-4) / TILE_SEC));
      // Lookahead scales with measured scroll/playback speed: enough tiles to
      // cover LOOKAHEAD_SEC of audio at the current speed (clamped to a min/max).
      const speed = currentSpeed();
      const aheadTiles = Math.min(
        MAX_AHEAD_TILES,
        Math.max(MIN_AHEAD_TILES, Math.ceil((speed * LOOKAHEAD_SEC) / TILE_SEC)),
      );
      const center = Math.floor(((visStart + visEnd) / 2) / TILE_SEC);

      // Priority order (earliest entries are fetched first):
      //  1. tiles on screen, nearest to the viewport center first
      //  2. tiles ahead in the scroll/playback direction, nearest first
      //  3. one tile behind, in case the direction reverses
      const visTiles: number[] = [];
      for (let k = firstVis; k <= lastVis; k++) visTiles.push(k);
      visTiles.sort((a, b) => Math.abs(a - center) - Math.abs(b - center));

      const ahead: number[] = [];
      for (let i = 1; i <= aheadTiles; i++) {
        const k = scrollDir > 0 ? lastVis + i : firstVis - i;
        if (k >= 0 && k <= maxTile) ahead.push(k);
      }
      const behind = scrollDir > 0 ? firstVis - 1 : lastVis + 1;
      const behindList = behind >= 0 && behind <= maxTile ? [behind] : [];

      const prioritized = [...visTiles, ...ahead, ...behindList];
      const desired = new Set(prioritized);
      for (const [k, ac] of inFlight.entries()) {
        if (!desired.has(k)) {
          ac.abort();
          inFlight.delete(k);
        }
      }

      const cache = tileCacheRef.current;
      const failed = failedTilesRef.current;
      for (const tileIdx of prioritized) {
        if (inFlight.size >= MAX_CONCURRENT_TILE_FETCHES) break;
        if (cache.has(tileIdx) || inFlight.has(tileIdx) || failed.has(tileIdx)) continue;

        const reqStart = tileIdx * TILE_SEC;
        const reqEnd = (tileIdx + 1) * TILE_SEC;
        const ac = new AbortController();
        inFlight.set(tileIdx, ac);
        const url = `${API_BASE}/api/jobs/${encodeURIComponent(jobId)}/spectrogram-slice?startSec=${reqStart}&endSec=${reqEnd}`;

        fetch(url, { signal: ac.signal })
          .then(async res => {
            inFlight.delete(tileIdx);
            if (!res.ok) {
              if (res.status === 404) failed.add(tileIdx);
              return;
            }
            const fb = Number(res.headers.get('X-Freq-Bins'));
            const tc = Number(res.headers.get('X-Time-Cols'));
            const buf = new Uint8Array(await res.arrayBuffer());
            if (!fb || !tc || buf.length !== fb * tc) return;
            const tileCanvas = buildTileCanvas(buf, fb, tc);
            cache.set(tileIdx, {
              startSec: reqStart,
              endSec: Math.min(audioDurationSec, reqEnd),
              freqBins: fb,
              canvas: tileCanvas,
            });
            while (cache.size > MAX_CACHED_TILES) {
              const oldestKey = cache.keys().next().value;
              if (oldestKey === undefined) break;
              cache.delete(oldestKey);
            }
            if (!disposed) {
              const vp = getViewport();
              paint(vp.left, vp.width);
              schedule(vp.left, vp.width);
            }
          })
          .catch(() => {
            inFlight.delete(tileIdx);
          });
      }
    };

    const sync = () => {
      const { left, width } = getViewport();
      const now = performance.now();
      if (lastLeft === null) {
        lastLeft = left;
        lastT = now;
      } else if (left !== lastLeft) {
        const dt = (now - lastT) / 1000;
        if (dt > 0 && dt < 1) {
          // Smoothed scroll speed in audio-seconds per wall-second (sign = direction).
          const dSec = ((left - lastLeft) / cssWidth) * audioDurationSec;
          velSecPerSec = 0.5 * velSecPerSec + 0.5 * (dSec / dt);
        } else {
          velSecPerSec = 0;
        }
        scrollDir = left > lastLeft ? 1 : -1;
        lastLeft = left;
        lastT = now;
        lastMoveT = now;
      }
      paint(left, width);
      schedule(left, width);
    };

    sync();
    const el = scrollRef?.current;
    if (!el) return () => { disposed = true; };
    el.addEventListener('scroll', sync, { passive: true });
    const ro = new ResizeObserver(sync);
    ro.observe(el);
    return () => {
      disposed = true;
      el.removeEventListener('scroll', sync);
      ro.disconnect();
    };
  }, [isDecimated, jobId, audioDurationSec, cssWidth, freqBins, scrollRef]);

  useEffect(() => {
    const canvas = linesCanvasRef.current;
    if (!canvas || timeCols <= 0 || cssWidth <= 0) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.max(1, Math.round(overlayCssW * dpr));
    canvas.height = Math.max(1, Math.round(cssHeight * dpr));
    canvas.style.width = `${overlayCssW}px`;
    canvas.style.height = `${cssHeight}px`;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, overlayCssW, cssHeight);

    const col0 = (scrollLeft / cssWidth) * timeCols;
    const col1 = ((scrollLeft + overlayCssW) / cssWidth) * timeCols;
    const haloWidth = 3;

    if (highlightedRange) {
      const hx0 = (highlightedRange.startCol / timeCols) * cssWidth - scrollLeft;
      const hx1 = (highlightedRange.endCol / timeCols) * cssWidth - scrollLeft;
      if (hx1 >= 0 && hx0 <= overlayCssW) {
        ctx.fillStyle = 'rgba(250, 204, 21, 0.15)';
        ctx.fillRect(hx0, 0, hx1 - hx0, cssHeight);
        ctx.strokeStyle = '#facc15';
        ctx.lineWidth = 2;
        ctx.strokeRect(hx0 + 1, 1, Math.max(0, hx1 - hx0 - 2), cssHeight - 2);
      }
    }

    const lines = cutoffs.length === 0
      ? structureLines
      : [
        ...structureLines,
        ...cutoffs.map((col, i) => ({ col, color: cutoffColor(cutoffDistances?.[i]) })),
      ];

    for (const line of lines) {
      if (line.col < col0 - 1 || line.col > col1 + 1) continue;
      const x = (line.col / timeCols) * cssWidth - scrollLeft;
      const haloLeft = Math.min(Math.max(0, x - haloWidth / 2), Math.max(0, overlayCssW - haloWidth));
      ctx.fillStyle = 'rgba(0, 0, 0, 0.45)';
      ctx.fillRect(haloLeft, 0, haloWidth, cssHeight);
      ctx.fillStyle = line.color;
      ctx.fillRect(haloLeft + (haloWidth - 1) / 2, 0, 1, cssHeight);
    }
  }, [
    structureLines, cutoffs, cutoffDistances, highlightedRange,
    timeCols, cssWidth, cssHeight, scrollLeft, overlayCssW,
  ]);

  function handleClick(e: React.MouseEvent<HTMLCanvasElement>) {
    if (!onColClick) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const col = Math.floor((x / rect.width) * timeCols);
    onColClick(Math.max(0, Math.min(timeCols - 1, col)));
  }

  return (
    <div style={{ position: 'relative', width: cssWidth, height: cssContainerHeight }}>
      {/* Clipping scoped to just the main canvas (vertical freq-zoom) — an
          overflow:hidden ancestor here would otherwise break the lines
          overlay's position:sticky below, which needs an unclipped path up
          to the real scrolling container. */}
      <div style={{ position: 'absolute', inset: 0, overflow: 'hidden' }}>
        <canvas
          ref={canvasRef}
          style={{
            display: 'block',
            position: 'absolute',
            bottom: 0,
            left: 0,
            width: cssWidth,
            height: cssHeight,
            cursor: onColClick ? 'pointer' : 'default',
          }}
          onClick={handleClick}
        />
        {isDecimated && (
          <canvas
            ref={hiResCanvasRef}
            style={{
              display: 'block',
              position: 'absolute',
              bottom: 0,
              height: cssHeight,
              pointerEvents: 'none',
            }}
          />
        )}
      </div>
      {/* Repeat chunk boxes: absolute DOM overlays above the canvas but below the
          sticky lines/highlight canvas (which follows in DOM order). */}
      {repeatOverlays?.map(overlay => {
        const left = (overlay.startCol / timeCols) * cssWidth;
        const width = Math.max(2, ((overlay.endCol - overlay.startCol) / timeCols) * cssWidth);
        const { fill, border } = repeatOverlayColor(overlay.templateIndex);
        return (
          <div
            key={overlay.chunkId}
            aria-label={`Repeat template ${overlay.templateNumber} chunk`}
            style={{
              position: 'absolute',
              top: 0,
              height: cssContainerHeight,
              left: `${left}px`,
              width: `${width}px`,
              background: fill,
              border: `1.5px solid ${border}`,
              boxSizing: 'border-box',
              overflow: 'hidden',
              pointerEvents: 'none',
            }}
          >
            <span
              style={{
                position: 'absolute',
                top: 2,
                left: 3,
                fontSize: 10,
                fontFamily: 'monospace',
                fontWeight: 700,
                lineHeight: 1,
                color: '#fff',
                textShadow: '0 0 2px #000, 0 0 3px #000',
                whiteSpace: 'nowrap',
              }}
            >
              {overlay.templateNumber}
            </span>
          </div>
        );
      })}
      <div
        style={{
          position: 'sticky',
          left: 0,
          width: overlayCssW,
          height: cssContainerHeight,
          overflow: 'hidden',
          pointerEvents: 'none',
        }}
      >
        <canvas ref={linesCanvasRef} aria-hidden style={{ display: 'block' }} />
      </div>
    </div>
  );
});
