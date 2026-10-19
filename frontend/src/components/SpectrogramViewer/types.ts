export interface SpectrogramViewerProps {
  /** Row-major magnitude; uint8 (v2 panel) or float32 (legacy). */
  magnitude: Float32Array | Uint8Array;
  /** True when magnitude is pre-quantized uint8 from panel v2. */
  magnitudeIsUint8?: boolean;
  freqBins: number;
  timeCols: number;
  /** Column-sum energy; omit to hide the energy plot */
  energyRaw?: Float32Array;
  /** Gaussian-smoothed (σ=3) energy; omit to hide smoothed line */
  energySmoothed?: Float32Array;
  title: string;
  colormap?: 'jet' | 'viridis';
  /** Client-side brightness multiplier (legacy float32 panels), default 1.0 */
  brightness?: number;
  /** Panel-level normalization max (legacy float32 or v2 reference) */
  normMax?: number;
  /** Total audio duration in seconds (from manifest); used to sync playback scroll */
  audioDurationSec?: number;
  /** Job ID used to construct the audio URL */
  jobId?: string;
  /** Frequency axis bounds in Hz (from panel header) */
  freqLo?: number;
  freqHi?: number;
}
