import {
  MAX_PX_PER_SEC,
  MIN_FREQ_ZOOM_HZ,
  MIN_PX_PER_SEC,
  type SpectrogramScale,
} from './useSpectrogramScale';
import { ZoomControl } from './ZoomControl';

/** The single px/sec + max-Hz menu; drives every viewer sharing this scale. */
export function ScaleControls({ scale }: { scale: SpectrogramScale }) {
  return (
    <>
      {scale.audioDurationSec != null && scale.audioDurationSec > 0 && (
        <ZoomControl
          label="px/sec"
          value={scale.pxPerSec}
          min={MIN_PX_PER_SEC}
          max={MAX_PX_PER_SEC}
          step={1}
          onChange={scale.setPxPerSec}
        />
      )}
      {scale.hasFreqRange && (
        <ZoomControl
          label="Max Hz"
          value={scale.maxFreqHz}
          min={Math.min(scale.freqHi!, scale.freqLo! + MIN_FREQ_ZOOM_HZ)}
          max={scale.freqHi!}
          step={50}
          onChange={scale.setMaxFreqHz}
        />
      )}
    </>
  );
}
