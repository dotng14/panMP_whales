import type { SpectrogramViewerProps } from '../components/SpectrogramViewer/types';

const FREQ_BINS = 180;
const TIME_COLS = 1500;
const N_CUTOFFS = 22;

function gaussianNoise(mean: number, std: number): number {
  // Box-Muller
  const u = 1 - Math.random();
  const v = Math.random();
  return mean + std * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function gaussianKernel(sigma: number): number[] {
  const radius = Math.ceil(3 * sigma);
  const kernel: number[] = [];
  let sum = 0;
  for (let i = -radius; i <= radius; i++) {
    const v = Math.exp(-(i * i) / (2 * sigma * sigma));
    kernel.push(v);
    sum += v;
  }
  return kernel.map(v => v / sum);
}

function convolve(arr: Float32Array, kernel: number[]): Float32Array {
  const out = new Float32Array(arr.length);
  const half = Math.floor(kernel.length / 2);
  for (let i = 0; i < arr.length; i++) {
    let acc = 0;
    for (let k = 0; k < kernel.length; k++) {
      const j = i + k - half;
      acc += (j >= 0 && j < arr.length ? arr[j] : 0) * kernel[k];
    }
    out[i] = acc;
  }
  return out;
}

export function generateMockData(seed = 0): SpectrogramViewerProps {
  // deterministic-ish seed offset
  const off = seed * 137;

  const magnitude = new Float32Array(FREQ_BINS * TIME_COLS);

  // noise floor
  for (let i = 0; i < magnitude.length; i++) {
    magnitude[i] = Math.max(0.5, gaussianNoise(1.2, 0.25));
  }

  // three bright horizontal bands at low frequencies (rows 3–8, 12–18, 22–30)
  const bands = [[3, 8], [12, 18], [22, 30]];
  for (const [lo, hi] of bands) {
    for (let r = lo; r <= hi; r++) {
      for (let c = 0; c < TIME_COLS; c++) {
        magnitude[r * TIME_COLS + c] += gaussianNoise(1.8, 0.4);
      }
    }
  }

  const cutoffs = Array.from({ length: N_CUTOFFS }, (_, i) =>
    Math.round(((i + 1) * TIME_COLS) / (N_CUTOFFS + 1))
  );

  const energyRaw = new Float32Array(TIME_COLS);
  for (let c = 0; c < TIME_COLS; c++) {
    let e = Math.max(0, gaussianNoise(5000, 3000) + off);
    for (const cutoff of cutoffs) {
      const dist = Math.abs(c - cutoff);
      e += 80000 * Math.exp(-(dist * dist) / 8);
    }
    energyRaw[c] = e;
  }

  const energySmoothed = convolve(energyRaw, gaussianKernel(3));

  return {
    magnitude,
    freqBins: FREQ_BINS,
    timeCols: TIME_COLS,
    cutoffs,
    energyRaw,
    energySmoothed,
    title: seed === 0
      ? '[5/7] spwhiten_000004_j000180_000224.bin'
      : '[5/7] spwhiten_000001_j000180_000224.bin',
    brightness: seed === 0 ? 1.0 : 1.5,
  };
}

