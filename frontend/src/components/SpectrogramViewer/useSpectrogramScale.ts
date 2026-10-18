import { useMemo, useState } from 'react';
import type { SpectrogramViewerProps } from './types';

export const MIN_PX_PER_SEC = 5;
export const MAX_PX_PER_SEC = 500;
export const MIN_FREQ_ZOOM_HZ = 100;

export interface SpectrogramScale {
  pxPerSec: number;
  setPxPerSec: (value: number) => void;
  maxFreqHz: number;
  setMaxFreqHz: (value: number) => void;
  hasFreqRange: boolean;
  freqLo?: number;
  freqHi?: number;
  audioDurationSec?: number;
  /** Fraction of the full frequency range currently shown, stretched to fill the canvas. */
  freqCropFraction: number;
}

type ScaleSource = Pick<
  SpectrogramViewerProps,
  'audioDurationSec' | 'freqLo' | 'freqHi'
>;

/** Default horizontal zoom (pixels per second of audio). */
export const DEFAULT_PX_PER_SEC = 32;
/** Default vertical zoom cap (Hz). */
export const DEFAULT_MAX_FREQ_HZ = 5000;

/**
 * Horizontal (px/sec) and vertical (max Hz) zoom state. Hoist this into a parent
 * and hand the result to several viewers so one set of controls drives them all.
 */
export function useSpectrogramScale({
  audioDurationSec,
  freqLo,
  freqHi,
}: ScaleSource): SpectrogramScale {
  const hasFreqRange = freqLo != null && freqHi != null && freqHi > freqLo;
  const [pxPerSec, setPxPerSec] = useState(DEFAULT_PX_PER_SEC);
  const [maxFreqHz, setMaxFreqHz] = useState(() => (hasFreqRange ? Math.min(freqHi!, DEFAULT_MAX_FREQ_HZ) : 0));

  const freqCropFraction = useMemo(() => {
    if (!hasFreqRange) return 1;
    return Math.min(1, Math.max(0.02, (maxFreqHz - freqLo!) / (freqHi! - freqLo!)));
  }, [hasFreqRange, maxFreqHz, freqLo, freqHi]);

  return {
    pxPerSec,
    setPxPerSec,
    maxFreqHz,
    setMaxFreqHz,
    hasFreqRange,
    freqLo,
    freqHi,
    audioDurationSec,
    freqCropFraction,
  };
}
