import { useEffect, useRef, type RefObject } from 'react';

export const XAXIS_H = 24;

interface Props {
  timeCols: number;
  audioDurationSec?: number;
  /** Total scrollable content width in px (horizontal zoom); defaults to timeCols (1px/column) */
  displayWidth?: number;
  /** Horizontal scroll container the axis tracks; the canvas only draws the visible window. */
  scrollRef: RefObject<HTMLDivElement | null>;
}

const MIN_TICK_PX = 80;
const TICK_CANDIDATES = [0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800];

function niceSecInterval(totalSec: number, contentWidthPx: number): number {
  for (const s of TICK_CANDIDATES) {
    if ((s / totalSec) * contentWidthPx >= MIN_TICK_PX) return s;
  }
  return TICK_CANDIDATES[TICK_CANDIDATES.length - 1];
}

function formatSec(sec: number): string {
  if (sec < 60) return `${parseFloat(sec.toFixed(2))}s`;
  const m = Math.floor(sec / 60);
  const s = parseFloat((sec % 60).toFixed(2));
  return s === 0 ? `${m}m` : `${m}m${s}s`;
}

/**
 * Virtualized time axis: sized to the scroll container's viewport and redrawn on
 * scroll/resize. Never allocates a canvas at the full zoomed content width, which
 * would exceed the browser canvas size limit (~32k px) at high px/sec zoom.
 */
export function XAxis({ timeCols, audioDurationSec, displayWidth, scrollRef }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const scroller = scrollRef.current;
    if (!canvas || !scroller) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const contentW = Math.max(1, displayWidth ?? timeCols);

    const draw = () => {
      const viewW = Math.max(1, scroller.clientWidth);
      canvas.width = viewW;
      canvas.height = XAXIS_H;
      canvas.style.width = `${viewW}px`;
      const off = scroller.scrollLeft;

      ctx.fillStyle = '#f5f5f5';
      ctx.fillRect(0, 0, viewW, XAXIS_H);

      ctx.strokeStyle = '#374151';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, 0);
      ctx.lineTo(viewW, 0);
      ctx.stroke();

      ctx.fillStyle = '#374151';
      ctx.font = '11px system-ui';
      ctx.textAlign = 'center';

      if (audioDurationSec && audioDurationSec > 0) {
        const interval = niceSecInterval(audioDurationSec, contentW);
        const pxPerSec = contentW / audioDurationSec;
        // Index-based iteration avoids float drift with fractional intervals (0.1s etc).
        const iFirst = Math.max(1, Math.floor(off / pxPerSec / interval));
        for (let i = iFirst; i * interval <= audioDurationSec; i++) {
          const sec = i * interval;
          const x = sec * pxPerSec - off;
          if (x > viewW) break;
          if (x < 0) continue;
          ctx.beginPath();
          ctx.moveTo(x, 0);
          ctx.lineTo(x, 5);
          ctx.stroke();
          ctx.fillText(formatSec(sec), x, XAXIS_H - 3);
        }
      } else {
        const tickInterval = 100;
        const scale = contentW / timeCols;
        for (let col = 0; col <= timeCols; col += tickInterval) {
          const x = col * scale - off;
          if (x > viewW) break;
          if (x < 0) continue;
          ctx.beginPath();
          ctx.moveTo(x, 0);
          ctx.lineTo(x, 4);
          ctx.stroke();
          ctx.fillText(String(col), x, XAXIS_H - 3);
        }
      }
    };

    draw();
    scroller.addEventListener('scroll', draw, { passive: true });
    const ro = new ResizeObserver(draw);
    ro.observe(scroller);
    return () => {
      scroller.removeEventListener('scroll', draw);
      ro.disconnect();
    };
  }, [timeCols, audioDurationSec, displayWidth, scrollRef]);

  return <canvas ref={canvasRef} style={{ display: 'block' }} />;
}
