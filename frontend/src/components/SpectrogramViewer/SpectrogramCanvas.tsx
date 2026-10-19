import { memo, useEffect, useRef } from 'react';
import { JET, JET_RGBA, magnitudeToIndex } from '../../utils/colormap';

export const SPEC_TARGET_H = 400;

interface Props {
  magnitude: Float32Array | Uint8Array;
  magnitudeIsUint8: boolean;
  freqBins: number;
  timeCols: number;
  /** Song/theme/phrase/motif boundary markers, colored to match their lane row. */
  structureLines: { col: number; color: string }[];
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
  structureLines,
  brightness,
  normMax,
  onColClick,
  displayWidth,
  containerHeight,
  displayHeight,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
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

  function handleClick(e: React.MouseEvent<HTMLCanvasElement>) {
    if (!onColClick) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const col = Math.floor((x / rect.width) * timeCols);
    onColClick(Math.max(0, Math.min(timeCols - 1, col)));
  }

  return (
    <div style={{ position: 'relative', width: cssWidth, height: cssContainerHeight, overflow: 'hidden' }}>
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
      {structureLines.map((line, index) => {
        const center = (line.col / timeCols) * cssWidth;
        // Centering a 3px-wide line on `center` would let it straddle the
        // container edge for the last chunk's trailing line (center === cssWidth)
        // — half of it then falls outside this overflow:hidden box and gets
        // clipped to an invisible sliver. Clamp so the whole line stays inside.
        const haloWidth = 3;
        const haloLeft = Math.min(
          Math.max(0, center - haloWidth / 2),
          Math.max(0, cssWidth - haloWidth),
        );
        const coreLeft = haloLeft + (haloWidth - 1) / 2;
        return (
          <div key={`${line.col}-${line.color}-${index}`} aria-hidden style={{ position: 'absolute', top: 0, bottom: 0, left: 0, zIndex: 1 }}>
            {/* Dark halo behind the colored core so the line stays visible against every JET hue. */}
            <div style={{ position: 'absolute', top: 0, bottom: 0, left: `${haloLeft}px`, width: `${haloWidth}px`, background: 'rgba(0, 0, 0, 0.45)' }} />
            <div style={{ position: 'absolute', top: 0, bottom: 0, left: `${coreLeft}px`, width: '1px', background: line.color }} />
          </div>
        );
      })}
    </div>
  );
});
