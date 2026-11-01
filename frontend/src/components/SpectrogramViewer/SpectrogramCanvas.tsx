import { memo, useEffect, useRef, useState, type RefObject } from 'react';
import { JET, JET_RGBA, magnitudeToIndex } from '../../utils/colormap';
import { repeatOverlayColor } from './repeatOverlayColors';
import type { RepeatOverlay } from './types';

export const SPEC_TARGET_H = 400;

/** Browsers reject canvases near ~16k–32k px; stay well under that (matches LaneCanvas). */
const MAX_CANVAS_CSS_W = 8192;

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
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const linesCanvasRef = useRef<HTMLCanvasElement>(null);
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
