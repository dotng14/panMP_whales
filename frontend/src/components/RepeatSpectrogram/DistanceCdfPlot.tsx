import { useMemo } from 'react';
import { SPEC_TARGET_H } from '../SpectrogramViewer/SpectrogramCanvas';

export const DISTANCE_CDF_PLOT_W = 100;
export const DISTANCE_CDF_PAD = { top: 8, right: 8, bottom: 28, left: 28 };
const PLOT_W = DISTANCE_CDF_PLOT_W;
const PAD = DISTANCE_CDF_PAD;

interface Props {
  distances: number[];
  height?: number;
}

function empiricalCdf(sorted: number[]): Array<{ x: number; y: number }> {
  const n = sorted.length;
  if (n === 0) return [];
  const points: Array<{ x: number; y: number }> = [{ x: sorted[0], y: 0 }];
  for (let i = 0; i < n; i++) {
    points.push({ x: sorted[i], y: (i + 1) / n });
  }
  return points;
}

/** Compact empirical CDF of per-chunk distanceNorm values (0 = perfect fit). */
export function DistanceCdfPlot({ distances, height = SPEC_TARGET_H }: Props) {
  const finite = useMemo(
    () => distances.filter(d => Number.isFinite(d)).sort((a, b) => a - b),
    [distances],
  );

  const { path, medianX, innerW, innerH } = useMemo(() => {
    const innerW = PLOT_W - PAD.left - PAD.right;
    const innerH = height - PAD.top - PAD.bottom;
    if (finite.length < 2) {
      return { path: '', medianX: null as number | null, innerW, innerH };
    }
    const pts = empiricalCdf(finite);
    const xMax = Math.max(finite[finite.length - 1], 1e-6);
    const scaleX = (x: number) => PAD.left + (x / xMax) * innerW;
    const scaleY = (y: number) => PAD.top + (1 - y) * innerH;
    const d: string[] = [];
    for (let i = 0; i < pts.length; i++) {
      const px = scaleX(pts[i].x);
      const py = scaleY(pts[i].y);
      if (i === 0) {
        d.push(`M ${px} ${py}`);
      } else {
        // Step CDF: horizontal then vertical.
        d.push(`H ${px}`);
        d.push(`V ${py}`);
      }
    }
    const mid = finite[Math.floor((finite.length - 1) / 2)];
    return {
      path: d.join(' '),
      medianX: scaleX(mid),
      innerW,
      innerH,
    };
  }, [finite, height]);

  if (finite.length < 2) {
    return (
      <div
        style={{
          width: PLOT_W,
          height,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: '#9ca3af',
          fontSize: 11,
          border: '1px dashed #e5e7eb',
          boxSizing: 'border-box',
        }}
      >
        n/a
      </div>
    );
  }

  const xMax = Math.max(finite[finite.length - 1], 1e-6);

  return (
    <svg
      width={PLOT_W}
      height={height}
      viewBox={`0 0 ${PLOT_W} ${height}`}
      role="img"
      aria-label="Empirical CDF of chunk distance to template average"
      style={{ display: 'block', flexShrink: 0 }}
    >
      <rect
        x={PAD.left}
        y={PAD.top}
        width={innerW}
        height={innerH}
        fill="#fafafa"
        stroke="#e5e7eb"
      />
      {/* Grid */}
      {[0.25, 0.5, 0.75].map(g => (
        <line
          key={`h-${g}`}
          x1={PAD.left}
          x2={PAD.left + innerW}
          y1={PAD.top + (1 - g) * innerH}
          y2={PAD.top + (1 - g) * innerH}
          stroke="#f3f4f6"
        />
      ))}
      {medianX != null && (
        <line
          x1={medianX}
          x2={medianX}
          y1={PAD.top}
          y2={PAD.top + innerH}
          stroke="#c4b5fd"
          strokeDasharray="3 3"
        />
      )}
      <path d={path} fill="none" stroke="#7c3aed" strokeWidth={1.75} />
      {/* Axes labels */}
      <text
        x={PAD.left + innerW / 2}
        y={height - 6}
        textAnchor="middle"
        fontSize={9}
        fill="#6b7280"
        fontFamily="monospace"
      >
        distance
      </text>
      <text
        x={10}
        y={PAD.top + innerH / 2}
        textAnchor="middle"
        fontSize={9}
        fill="#6b7280"
        fontFamily="monospace"
        transform={`rotate(-90 10 ${PAD.top + innerH / 2})`}
      >
        CDF
      </text>
      <text x={PAD.left} y={height - 16} fontSize={8} fill="#9ca3af" fontFamily="monospace">
        0
      </text>
      <text
        x={PAD.left + innerW}
        y={height - 16}
        textAnchor="end"
        fontSize={8}
        fill="#9ca3af"
        fontFamily="monospace"
      >
        {xMax.toFixed(2)}
      </text>
      <text x={PAD.left - 4} y={PAD.top + 4} textAnchor="end" fontSize={8} fill="#9ca3af" fontFamily="monospace">
        1
      </text>
      <text
        x={PAD.left - 4}
        y={PAD.top + innerH}
        textAnchor="end"
        fontSize={8}
        fill="#9ca3af"
        fontFamily="monospace"
      >
        0
      </text>
    </svg>
  );
}
