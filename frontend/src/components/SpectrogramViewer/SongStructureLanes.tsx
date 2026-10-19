import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
  type MouseEvent as ReactMouseEvent,
  type RefObject,
} from 'react';
import type { SongStructure, SongStructureUnit } from '../../api/songStructure';
import { motifColor, phraseColor, themeColor } from './songStructureColors';

export const LANE_H = 18;
export const LANE_ORDER = ['motif', 'phrase', 'theme'] as const;
export type LaneName = (typeof LANE_ORDER)[number];

const LABEL_MIN_PX = 28;
/** Browsers reject canvases near ~16k–32k px; stay well under that. */
const MAX_CANVAS_CSS_W = 8192;

interface LaneSeg {
  startCol: number;
  endCol: number;
  color: string;
  label: string;
  title: string;
}

export function secToCol(sec: number, audioDurationSec: number, timeCols: number): number {
  if (audioDurationSec <= 0 || timeCols <= 0) return 0;
  return Math.max(0, Math.min(timeCols, Math.round((sec / audioDurationSec) * timeCols)));
}

function buildLanes(
  units: SongStructureUnit[],
  audioDurationSec: number,
  timeCols: number,
): Record<LaneName, LaneSeg[]> {
  const motif: LaneSeg[] = [];
  const theme: LaneSeg[] = [];
  const phrase: LaneSeg[] = [];

  for (const u of units) {
    const startCol = secToCol(u.startSec, audioDurationSec, timeCols);
    const endCol = secToCol(u.endSec, audioDurationSec, timeCols);
    if (endCol <= startCol) continue;
    const title = `motif ${u.motif} · song ${u.song} · theme ${u.theme} · phrase ${u.phraseId}×${u.phraseCount}`;
    motif.push({
      startCol,
      endCol,
      color: motifColor(u.motif),
      label: u.motif,
      title,
    });
    theme.push({
      startCol,
      endCol,
      color: themeColor(u.theme),
      label: String(u.theme),
      title,
    });
    phrase.push({
      startCol,
      endCol,
      color: phraseColor(u.phraseId),
      label: `${u.phraseId}·${u.phraseCount}`,
      title,
    });
  }
  return { motif, theme, phrase };
}

/** Merge adjacent same-label segments so we draw fewer rects at full zoom. */
function mergeAdjacent(segs: LaneSeg[]): LaneSeg[] {
  if (segs.length === 0) return segs;
  const out: LaneSeg[] = [{ ...segs[0] }];
  for (let i = 1; i < segs.length; i++) {
    const prev = out[out.length - 1];
    const cur = segs[i];
    if (cur.label === prev.label && cur.startCol <= prev.endCol + 1) {
      prev.endCol = Math.max(prev.endCol, cur.endCol);
    } else {
      out.push({ ...cur });
    }
  }
  return out;
}

function LaneCanvas({
  segs,
  timeCols,
  displayWidth,
  songBoundaries,
  scrollLeft,
  viewWidth,
  activeRange,
}: {
  segs: LaneSeg[];
  timeCols: number;
  displayWidth: number;
  songBoundaries: number[];
  scrollLeft: number;
  viewWidth: number;
  activeRange?: { startCol: number; endCol: number } | null;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const cssW = Math.max(1, Math.min(viewWidth, MAX_CANVAS_CSS_W));

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || timeCols <= 0 || displayWidth <= 0 || cssW <= 0) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.max(1, Math.round(cssW * dpr));
    canvas.height = Math.max(1, Math.round(LANE_H * dpr));
    canvas.style.width = `${cssW}px`;
    canvas.style.height = `${LANE_H}px`;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssW, LANE_H);
    ctx.fillStyle = '#f1f5f9';
    ctx.fillRect(0, 0, cssW, LANE_H);

    const col0 = (scrollLeft / displayWidth) * timeCols;
    const col1 = ((scrollLeft + cssW) / displayWidth) * timeCols;

    for (const seg of segs) {
      if (seg.endCol < col0 || seg.startCol > col1) continue;
      const left = (seg.startCol / timeCols) * displayWidth - scrollLeft;
      const right = (seg.endCol / timeCols) * displayWidth - scrollLeft;
      const width = Math.max(1, right - left);
      ctx.fillStyle = seg.color;
      ctx.fillRect(left, 1, width, LANE_H - 2);
      if (width >= LABEL_MIN_PX) {
        ctx.fillStyle = '#0f172a';
        ctx.font = '10px ui-sans-serif, system-ui, sans-serif';
        ctx.textBaseline = 'middle';
        const text = seg.label;
        const tw = ctx.measureText(text).width;
        if (tw + 4 < width) {
          const textX = Math.max(left + 3, 3);
          ctx.fillText(text, textX, LANE_H / 2);
        }
      }
    }

    if (activeRange) {
      const left = (activeRange.startCol / timeCols) * displayWidth - scrollLeft;
      const right = (activeRange.endCol / timeCols) * displayWidth - scrollLeft;
      ctx.strokeStyle = '#2563eb';
      ctx.lineWidth = 2;
      ctx.strokeRect(left, 0.5, Math.max(2, right - left), LANE_H - 1);
    }

    ctx.strokeStyle = 'rgba(15, 23, 42, 0.35)';
    ctx.lineWidth = 1;
    for (const col of songBoundaries) {
      const x = (col / timeCols) * displayWidth - scrollLeft;
      if (x < -1 || x > cssW + 1) continue;
      ctx.beginPath();
      ctx.moveTo(x + 0.5, 0);
      ctx.lineTo(x + 0.5, LANE_H);
      ctx.stroke();
    }
  }, [segs, timeCols, displayWidth, songBoundaries, scrollLeft, cssW, activeRange]);

  return (
    <canvas
      ref={canvasRef}
      aria-hidden
      style={{ display: 'block', width: cssW, height: LANE_H }}
    />
  );
}

/** Labels for the Y-axis column — outside the scroll, left of the colored bars. */
export function SongStructureLaneLabels({ yAxisWidth }: { yAxisWidth?: number }) {
  const style: CSSProperties = {
    fontSize: 10,
    color: '#475569',
    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Consolas, monospace',
    fontWeight: 600,
    height: LANE_H,
    lineHeight: `${LANE_H}px`,
    textAlign: 'right',
    paddingRight: 6,
    width: yAxisWidth,
    boxSizing: 'border-box',
  };
  return (
    <div style={{ marginTop: 4 }}>
      <div style={{ height: 14 }} />
      {LANE_ORDER.map(name => (
        <div key={name} style={style}>
          {name}
        </div>
      ))}
    </div>
  );
}

interface Props {
  structure: SongStructure;
  timeCols: number;
  audioDurationSec: number;
  displayWidth: number;
  scrollRef: RefObject<HTMLDivElement | null>;
  /** Column span currently highlighted (song jump or lane click). */
  activeRange?: { startCol: number; endCol: number } | null;
  /** Fired when the user clicks a colored segment (time-aligned with the spectrogram). */
  onSelectRange?: (range: { startCol: number; endCol: number } | null) => void;
}

export function SongStructureLanes({
  structure,
  timeCols,
  audioDurationSec,
  displayWidth,
  scrollRef,
  activeRange,
  onSelectRange,
}: Props) {
  const lanes = useMemo(() => {
    const raw = buildLanes(structure.units, audioDurationSec, timeCols);
    return {
      motif: mergeAdjacent(raw.motif),
      phrase: mergeAdjacent(raw.phrase),
      theme: mergeAdjacent(raw.theme),
    };
  }, [structure.units, audioDurationSec, timeCols]);

  const songBoundaries = useMemo(
    () =>
      structure.songs
        .slice(1)
        .map(s => secToCol(s.startSec, audioDurationSec, timeCols)),
    [structure.songs, audioDurationSec, timeCols],
  );

  const [scrollLeft, setScrollLeft] = useState(0);
  const [viewWidth, setViewWidth] = useState(800);
  const [hover, setHover] = useState<string | null>(null);

  useEffect(() => {
    const el = scrollRef.current;
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
  }, [scrollRef, displayWidth]);

  const colAtClientX = (clientX: number, target: HTMLElement) => {
    const el = scrollRef.current;
    if (!el) return 0;
    const rect = target.getBoundingClientRect();
    // Sticky overlay is flush with the scroll viewport — same time axis as the spectrogram.
    const x = clientX - rect.left + el.scrollLeft;
    return (x / displayWidth) * timeCols;
  };

  const segAtCol = (name: LaneName, col: number): LaneSeg | undefined =>
    lanes[name].find(s => col >= s.startCol && col < s.endCol);

  const onMove = (event: ReactMouseEvent<HTMLDivElement>) => {
    const col = colAtClientX(event.clientX, event.currentTarget);
    const unit = structure.units.find(u => {
      const a = secToCol(u.startSec, audioDurationSec, timeCols);
      const b = secToCol(u.endSec, audioDurationSec, timeCols);
      return col >= a && col < b;
    });
    setHover(
      unit
        ? `motif ${unit.motif} · song ${unit.song} · theme ${unit.theme} · phrase ${unit.phraseId}×${unit.phraseCount}`
        : null,
    );
  };

  const onClick = (event: ReactMouseEvent<HTMLDivElement>) => {
    if (!onSelectRange) return;
    const col = colAtClientX(event.clientX, event.currentTarget);
    // Which lane was clicked (top motif, mid phrase, bottom theme).
    const y = event.clientY - event.currentTarget.getBoundingClientRect().top;
    const laneIdx = Math.min(LANE_ORDER.length - 1, Math.max(0, Math.floor(y / LANE_H)));
    const name = LANE_ORDER[laneIdx];
    const seg = segAtCol(name, col);
    if (!seg) {
      onSelectRange(null);
      return;
    }
    onSelectRange({ startCol: seg.startCol, endCol: seg.endCol });
  };

  return (
    <div style={{ marginTop: 4, position: 'relative', width: displayWidth }}>
      <p
        style={{
          margin: '0 0 2px',
          fontSize: 10,
          color: '#94a3b8',
          fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Consolas, monospace',
          position: 'sticky',
          left: 0,
          width: viewWidth,
        }}
      >
        click a bar to highlight it on the spectrogram
      </p>
      <div style={{ position: 'relative', width: displayWidth }}>
        {LANE_ORDER.map(name => (
          <div key={name} style={{ width: displayWidth, height: LANE_H }}>
            <div
              style={{
                position: 'sticky',
                left: 0,
                width: viewWidth,
                height: LANE_H,
                overflow: 'hidden',
              }}
            >
              <LaneCanvas
                segs={lanes[name]}
                timeCols={timeCols}
                displayWidth={displayWidth}
                songBoundaries={songBoundaries}
                scrollLeft={scrollLeft}
                viewWidth={viewWidth}
                activeRange={activeRange}
              />
            </div>
          </div>
        ))}
        <div
          onMouseMove={onMove}
          onMouseLeave={() => setHover(null)}
          onClick={onClick}
          style={{
            position: 'sticky',
            left: 0,
            top: 0,
            width: viewWidth,
            height: LANE_H * LANE_ORDER.length,
            marginTop: -(LANE_H * LANE_ORDER.length),
            cursor: 'pointer',
            zIndex: 2,
          }}
          title={hover ?? undefined}
        />
      </div>
      {hover && (
        <p
          style={{
            margin: '2px 0 0',
            fontSize: 11,
            color: '#475569',
            position: 'sticky',
            left: 0,
            width: viewWidth,
          }}
        >
          {hover}
        </p>
      )}
    </div>
  );
}
