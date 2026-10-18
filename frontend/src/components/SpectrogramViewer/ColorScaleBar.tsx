const TICKS = [2.5, 2.0, 1.5, 1.0];

interface Props {
  height?: number;
}

export function ColorScaleBar({ height = 180 }: Props) {
  const gradId = 'jet-grad';
  // Layout (left→right): [gradient bar][ticks][numbers][rotated label]
  const barX = 4;
  const barW = 14;
  const tickX1 = barX + barW;
  const tickX2 = tickX1 + 5;
  const numX = tickX2 + 3;
  const labelX = numX + 28;
  const svgW = labelX + 12;

  return (
    <div style={{ display: 'flex', flexShrink: 0, marginLeft: 6 }}>
      <svg width={svgW} height={height} overflow="visible" style={{ display: 'block' }}>
        <defs>
          <linearGradient id={gradId} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%"   stopColor="#ff0000" />
            <stop offset="25%"  stopColor="#ffff00" />
            <stop offset="50%"  stopColor="#00ff00" />
            <stop offset="75%"  stopColor="#00ffff" />
            <stop offset="100%" stopColor="#0000ff" />
          </linearGradient>
        </defs>

        <rect x={barX} y={0} width={barW} height={height} fill={`url(#${gradId})`} />

        {TICKS.map((tick, i) => {
          const y = (i / (TICKS.length - 1)) * height;
          return (
            <g key={tick}>
              <line x1={tickX1} y1={y} x2={tickX2} y2={y} stroke="#374151" strokeWidth={1} />
              <text x={numX} y={y + 4} fill="#374151" fontSize={11} textAnchor="start">
                {tick.toFixed(1)}
              </text>
            </g>
          );
        })}

        {/* label on the far right, rotated */}
        <text x={labelX} y={height / 2} fill="#374151" fontSize={11} textAnchor="middle"
          transform={`rotate(90, ${labelX}, ${height / 2})`}>
          scaled magnitude
        </text>
      </svg>
    </div>
  );
}
