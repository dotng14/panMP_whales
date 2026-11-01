// Stable per-template hue cycle: distinct enough to tell templates apart, spaced
// around the wheel rather than adjacent so neighboring indices don't blend.
const HUES = [210, 30, 150, 280, 50, 190, 330, 100, 260, 10, 160, 300];

export function repeatOverlayColor(templateIndex: number): { fill: string; border: string } {
  const hue = HUES[((templateIndex % HUES.length) + HUES.length) % HUES.length];
  return {
    fill: `hsla(${hue}, 70%, 50%, 0.22)`,
    border: `hsla(${hue}, 70%, 50%, 0.85)`,
  };
}
