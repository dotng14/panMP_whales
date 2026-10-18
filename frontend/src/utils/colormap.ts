type RGB = [number, number, number];

function jetRgb(t: number): RGB {
  const clamp = (v: number) => Math.max(0, Math.min(1, v));
  const r = clamp(1.5 - Math.abs(4 * t - 3));
  const g = clamp(1.5 - Math.abs(4 * t - 2));
  const b = clamp(1.5 - Math.abs(4 * t - 1));
  return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
}

export const JET: RGB[] = Array.from({ length: 256 }, (_, i) => jetRgb(i / 255));

/** Flat RGBA lookup table for uint8 magnitude → ImageData (index = jet * 4). */
export const JET_RGBA = new Uint8ClampedArray(256 * 4);
for (let i = 0; i < 256; i++) {
  const [r, g, b] = JET[i];
  JET_RGBA[i * 4] = r;
  JET_RGBA[i * 4 + 1] = g;
  JET_RGBA[i * 4 + 2] = b;
  JET_RGBA[i * 4 + 3] = 255;
}

export function magnitudeToIndex(value: number, maxVal: number, brightness: number): number {
  const scaled = (value * brightness) / maxVal;
  return Math.max(0, Math.min(255, Math.round(scaled * 255)));
}
