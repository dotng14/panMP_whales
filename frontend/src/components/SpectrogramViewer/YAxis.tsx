export const YAXIS_W = 64;

interface Props {
  freqBins: number;
  height: number;
  freqLo?: number;
  freqHi?: number;
}

function niceHzInterval(rangeHz: number): number {
  if (rangeHz <= 200) return 25;
  if (rangeHz <= 500) return 50;
  if (rangeHz <= 2000) return 200;
  if (rangeHz <= 5000) return 500;
  if (rangeHz <= 10000) return 1000;
  return 2000;
}

export function YAxis({ freqBins, height, freqLo, freqHi }: Props) {
  const labelX = 10;
  const numX = YAXIS_W - 8;
  const tickX1 = YAXIS_W - 6;
  const axisX = YAXIS_W - 1;

  const hasHz = freqLo != null && freqHi != null && freqHi > freqLo;
  const rangeHz = hasHz ? freqHi! - freqLo! : 0;
  const interval = hasHz ? niceHzInterval(rangeHz) : 25;

  // Build tick list: either Hz values or raw bin indices
  const ticks: { label: string; y: number }[] = [];
  if (hasHz) {
    const lo = freqLo!;
    const hi = freqHi!;
    const firstTick = Math.ceil(lo / interval) * interval;
    for (let hz = firstTick; hz <= hi; hz += interval) {
      const frac = (hz - lo) / (hi - lo);
      const y = height - frac * height;
      const label = hz >= 1000 ? `${hz / 1000}k` : String(hz);
      ticks.push({ label, y });
    }
  } else {
    [0, 25, 50, 75, 100, 125, 150, 175].filter(b => b < freqBins).forEach(bin => {
      ticks.push({ label: String(bin), y: height - (bin / (freqBins - 1)) * height });
    });
  }

  return (
    <svg width={YAXIS_W} height={height} overflow="visible" style={{ display: 'block', flexShrink: 0 }}>
      <text x={labelX} y={height / 2} fill="#374151" fontSize={11} textAnchor="middle"
        transform={`rotate(-90, ${labelX}, ${height / 2})`}>
        {hasHz ? 'Frequency (Hz)' : 'Frequency bin'}
      </text>
      <line x1={axisX} y1={0} x2={axisX} y2={height} stroke="#374151" strokeWidth={1} />
      {ticks.map(({ label, y }) => (
        <g key={label}>
          <line x1={tickX1} y1={y} x2={axisX} y2={y} stroke="#374151" strokeWidth={1} />
          <text x={numX - 2} y={y + 4} fill="#374151" fontSize={11} textAnchor="end">{label}</text>
        </g>
      ))}
    </svg>
  );
}
