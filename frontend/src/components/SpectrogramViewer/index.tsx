import { useEffect, useRef, useState, useCallback, useId, useMemo } from 'react';
import { API_BASE } from '../../api/base';
import type { SpectrogramViewerProps } from './types';
import { announcePlayback, subscribeToPlayback } from '../../utils/audioCoordinator';
import { fetchSongStructure, type SongStructure } from '../../api/songStructure';
import { SPEC_TARGET_H, SpectrogramCanvas } from './SpectrogramCanvas';
import { ColorScaleBar } from './ColorScaleBar';
import { PlayButton } from './PlayButton';
import { YAxis } from './YAxis';
import { XAxis, XAXIS_H } from './XAxis';
import { ScaleControls } from './ScaleControls';
import { HorizontalScrollbar } from './HorizontalScrollbar';
import { useSpectrogramScale, type SpectrogramScale } from './useSpectrogramScale';
import { SongStructureLaneLabels, SongStructureLanes, secToCol } from './SongStructureLanes';

export function SpectrogramViewer({
  magnitude,
  magnitudeIsUint8 = false,
  freqBins,
  timeCols,
  cutoffs,
  title,
  brightness = 1.0,
  normMax,
  audioDurationSec,
  jobId,
  freqLo,
  freqHi,
  scale: sharedScale,
}: SpectrogramViewerProps & { scale?: SpectrogramScale }) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const lineRef = useRef<HTMLDivElement>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const gainRef = useRef<GainNode | null>(null);
  const rafRef = useRef<number | null>(null);
  const playerId = `original-${useId()}`;
  const [playing, setPlaying] = useState(false);
  const [volume, setVolume] = useState(8); // default 8x gain
  const [followPlayhead, setFollowPlayhead] = useState(true);
  const followPlayheadRef = useRef(true);
  const [songStructure, setSongStructure] = useState<SongStructure | null>(null);
  const [songIndex, setSongIndex] = useState(0);
  const [songNavActive, setSongNavActive] = useState(false);
  /** Clicked lane segment (or cleared); drawn on the lanes themselves. */
  const [laneHighlight, setLaneHighlight] = useState<{ startCol: number; endCol: number } | null>(null);

  // Fall back to private scale state when no shared one is supplied (e.g. mock mode).
  const ownScale = useSpectrogramScale({ audioDurationSec, freqLo, freqHi });
  const scale = sharedScale ?? ownScale;
  const { pxPerSec, maxFreqHz, hasFreqRange, freqCropFraction } = scale;

  // Fetch motif/song/theme/phrase boundaries for this job — independent of the
  // GPL pipeline, so this can resolve even if no CSV exists for the file (404).
  useEffect(() => {
    if (!jobId) {
      setSongStructure(null);
      return;
    }
    let cancelled = false;
    fetchSongStructure(jobId)
      .then(data => {
        if (!cancelled) {
          setSongStructure(data);
          setSongIndex(0);
          setSongNavActive(false);
          setLaneHighlight(null);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setSongStructure(null);
          setLaneHighlight(null);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [jobId]);

  const spectrogramH = SPEC_TARGET_H;

  // Horizontal zoom: canvas/x-axis CSS width derived from the pixels-per-second setting.
  const displayWidth = useMemo(
    () => (audioDurationSec ? Math.max(1, Math.round(audioDurationSec * pxPerSec)) : timeCols),
    [audioDurationSec, pxPerSec, timeCols],
  );

  // Vertical zoom: freqCropFraction of the full range is shown, stretched to fill spectrogramH.
  const canvasDisplayHeight = spectrogramH / freqCropFraction;

  // When the horizontal zoom changes, keep the time at the viewport center anchored
  // and re-position the playhead so the view doesn't jump to an unrelated spot.
  const prevDisplayWidthRef = useRef(displayWidth);
  useEffect(() => {
    const prev = prevDisplayWidthRef.current;
    prevDisplayWidthRef.current = displayWidth;
    const el = scrollRef.current;
    if (!el || prev === displayWidth || prev <= 0) return;
    const scale = displayWidth / prev;
    el.scrollLeft = (el.scrollLeft + el.clientWidth / 2) * scale - el.clientWidth / 2;
    const audio = audioRef.current;
    const line = lineRef.current;
    if (audio && line && audioDurationSec) {
      line.style.transform = `translateX(${(audio.currentTime / audioDurationSec) * displayWidth}px)`;
    }
  }, [displayWidth, audioDurationSec]);

  // Horizontal scroll wheel handler
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const handler = (e: WheelEvent) => {
      e.preventDefault();
      el.scrollLeft += e.deltaY + e.deltaX;
    };
    el.addEventListener('wheel', handler, { passive: false });
    return () => el.removeEventListener('wheel', handler);
  }, []);

  // Create audio element + Web Audio gain node when jobId is available
  useEffect(() => {
    if (!jobId) return;
    const audio = new Audio(`${API_BASE}/api/jobs/${jobId}/audio`);
    audio.preload = 'metadata';
    audio.crossOrigin = 'anonymous';
    audioRef.current = audio;

    const ctx = new AudioContext();
    const source = ctx.createMediaElementSource(audio);
    const gain = ctx.createGain();
    gain.gain.value = volume;
    source.connect(gain);
    gain.connect(ctx.destination);
    gainRef.current = gain;

    const onEnded = () => {
      setPlaying(false);
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    };
    audio.addEventListener('ended', onEnded);
    return () => {
      audio.removeEventListener('ended', onEnded);
      audio.pause();
      ctx.close();
      audioRef.current = null;
      gainRef.current = null;
    };
  }, [jobId]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => subscribeToPlayback(activePlayerId => {
    if (activePlayerId === playerId) return;
    audioRef.current?.pause();
    setPlaying(false);
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
  }), [playerId]);

  // Update gain when volume slider changes
  useEffect(() => {
    if (gainRef.current) gainRef.current.gain.value = volume;
  }, [volume]);

  const syncLine = useCallback(() => {
    const audio = audioRef.current;
    const el = scrollRef.current;
    const line = lineRef.current;
    if (!audio || !el || !line || !audioDurationSec) return;

    const progress = audio.currentTime / audioDurationSec;
    const canvasX = progress * displayWidth;

    // GPU-composited transform — no layout reflow on every frame.
    line.style.transform = `translateX(${canvasX}px)`;
    if (followPlayheadRef.current) {
      // Ease toward center rather than snapping — smooth camera follow.
      el.scrollLeft += (canvasX - el.clientWidth / 2 - el.scrollLeft) * 0.08;
    }

    if (!audio.paused) {
      rafRef.current = requestAnimationFrame(syncLine);
    }
  }, [audioDurationSec, displayWidth]);

  const handlePlayPause = useCallback(() => {
    const audio = audioRef.current;
    if (!audio) return;
    if (audio.paused) {
      announcePlayback(playerId);
      audio.play().then(() => {
        setPlaying(true);
        rafRef.current = requestAnimationFrame(syncLine);
      }).catch(err => console.error('Audio playback failed:', err));
    } else {
      audio.pause();
      setPlaying(false);
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    }
  }, [syncLine, playerId]);

  const handleColClick = useCallback((col: number) => {
    const audio = audioRef.current;
    if (!audio || !audioDurationSec) return;

    const seekSec = (col / timeCols) * audioDurationSec;
    audio.currentTime = seekSec;

    const el = scrollRef.current;
    const line = lineRef.current;
    if (el && line) {
      const canvasX = (seekSec / audioDurationSec) * displayWidth;
      line.style.transform = `translateX(${canvasX}px)`;
      if (followPlayheadRef.current) el.scrollLeft = canvasX - el.clientWidth / 2; // snap on seek
    }

    // Seeking only moves the playhead; playback state is left untouched so a
    // stopped viewer stays stopped when the user clicks a new section.
    if (!audio.paused && rafRef.current === null) {
      rafRef.current = requestAnimationFrame(syncLine);
    }
  }, [audioDurationSec, timeCols, displayWidth, syncLine]);

  const activeSong = songStructure?.songs[songIndex] ?? null;
  const activeSongRange = useMemo(() => {
    if (!songNavActive || !activeSong || !audioDurationSec) return null;
    return {
      startCol: secToCol(activeSong.startSec, audioDurationSec, timeCols),
      endCol: secToCol(activeSong.endSec, audioDurationSec, timeCols),
    };
  }, [songNavActive, activeSong, audioDurationSec, timeCols]);

  const structureHighlight = laneHighlight ?? activeSongRange;

  const jumpToColRange = useCallback((range: { startCol: number; endCol: number }) => {
    const el = scrollRef.current;
    if (!el || !audioDurationSec) return;
    const centerCol = (range.startCol + range.endCol) / 2;
    el.scrollTo({
      left: (centerCol / timeCols) * displayWidth - el.clientWidth / 2,
      behavior: 'smooth',
    });
    const startSec = (range.startCol / timeCols) * audioDurationSec;
    const audio = audioRef.current;
    if (audio) {
      audio.currentTime = startSec;
      const line = lineRef.current;
      if (line) {
        line.style.transform = `translateX(${(startSec / audioDurationSec) * displayWidth}px)`;
      }
    }
  }, [audioDurationSec, timeCols, displayWidth]);

  const jumpToSong = useCallback((index: number) => {
    if (!songStructure || !audioDurationSec) return;
    const song = songStructure.songs[index];
    if (!song) return;
    setSongIndex(index);
    setSongNavActive(true);
    setLaneHighlight(null);
    const range = {
      startCol: secToCol(song.startSec, audioDurationSec, timeCols),
      endCol: secToCol(song.endSec, audioDurationSec, timeCols),
    };
    jumpToColRange(range);
  }, [songStructure, audioDurationSec, timeCols, jumpToColRange]);

  const onLaneSelect = useCallback((range: { startCol: number; endCol: number } | null) => {
    setLaneHighlight(range);
    setSongNavActive(false);
    if (range) jumpToColRange(range);
  }, [jumpToColRange]);

  // Green line covers the spectrogram only (not the x-axis below)
  const totalH = spectrogramH;

  // Measure controls width so the right spacer mirrors it, truly centering the spectrogram
  const controlsRef = useRef<HTMLDivElement>(null);
  const [spacerW, setSpacerW] = useState(128);
  useEffect(() => {
    if (controlsRef.current) setSpacerW(controlsRef.current.offsetWidth);
  }, [jobId]);

  return (
    <div className="spectrogram-outer">
      <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>

        {/* Left: play controls — vertically centered with the spectrogram block */}
        <div ref={controlsRef} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 6, flexShrink: 0 }}>
          <PlayButton playing={playing} onClick={jobId ? handlePlayPause : undefined} />

          {jobId && (
            <>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 11, color: '#6b7280' }}>
                <span>🔈</span>
                <input
                  type="range"
                  min={1}
                  max={20}
                  step={1}
                  value={volume}
                  onChange={e => setVolume(Number(e.target.value))}
                  style={{ width: 72, accentColor: '#f59e0b' }}
                />
                <span>🔊</span>
              </div>

              <button
                onClick={() => {
                  const next = !followPlayhead;
                  followPlayheadRef.current = next;
                  setFollowPlayhead(next);
                }}
                style={{
                  fontSize: 11,
                  padding: '2px 8px',
                  borderRadius: 4,
                  border: '1px solid #d1d5db',
                  background: followPlayhead ? '#d1fae5' : '#f9fafb',
                  color: followPlayhead ? '#065f46' : '#6b7280',
                  cursor: 'pointer',
                  whiteSpace: 'nowrap',
                }}
              >
                {followPlayhead ? '🔒 Follow' : '🔓 Free scroll'}
              </button>
            </>
          )}

          <ScaleControls scale={scale} />

          {songStructure && songStructure.songs.length > 0 && (
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 4, marginTop: 8 }}>
              <div style={{ display: 'flex', gap: 4 }}>
                <button
                  type="button"
                  disabled={songIndex <= 0}
                  onClick={() => jumpToSong(songIndex - 1)}
                  style={{
                    fontSize: 11,
                    padding: '3px 8px',
                    borderRadius: 4,
                    border: '1px solid #d1d5db',
                    background: songIndex <= 0 ? '#f3f4f6' : '#fff',
                    color: '#374151',
                    cursor: songIndex <= 0 ? 'default' : 'pointer',
                  }}
                >
                  ← Song
                </button>
                <button
                  type="button"
                  disabled={songIndex >= songStructure.songs.length - 1}
                  onClick={() => jumpToSong(songIndex + 1)}
                  style={{
                    fontSize: 11,
                    padding: '3px 8px',
                    borderRadius: 4,
                    border: '1px solid #d1d5db',
                    background: songIndex >= songStructure.songs.length - 1 ? '#f3f4f6' : '#fff',
                    color: '#374151',
                    cursor: songIndex >= songStructure.songs.length - 1 ? 'default' : 'pointer',
                  }}
                >
                  Song →
                </button>
              </div>
              <span style={{ fontSize: 11, color: '#4b5563', whiteSpace: 'nowrap' }}>
                Song {songStructure.songs[songIndex]?.song ?? songIndex + 1}
                {' / '}
                {songStructure.songs.length}
              </span>
            </div>
          )}
        </div>

        {/* Center: title + spectrogram — takes all remaining space */}
        <div style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column' }}>
          <p style={{ textAlign: 'center', fontSize: 13, color: '#d97706', marginBottom: 6, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.08em', fontFamily: 'monospace', wordBreak: 'break-all' }}>
            {title}
          </p>

          <div style={{ display: 'flex', alignItems: 'flex-start' }}>
            {/* Y-axis column */}
            <div style={{ display: 'flex', flexDirection: 'column', flexShrink: 0 }}>
              <YAxis freqBins={freqBins} height={spectrogramH} freqLo={freqLo} freqHi={hasFreqRange ? maxFreqHz : freqHi} />
              {songStructure && audioDurationSec != null && audioDurationSec > 0 && (
                <SongStructureLaneLabels />
              )}
              {/* Spacer matching XAxis + label height */}
              <div style={{ height: XAXIS_H + 20 }} />
            </div>

            {/* Scrollable canvas area + non-scrolling axis label */}
            <div style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column' }}>
              <div ref={scrollRef} className="spectrogram-scroll" style={{ minWidth: 0, overflowY: 'hidden', position: 'relative' }}>
                <SpectrogramCanvas
                  magnitude={magnitude}
                  magnitudeIsUint8={magnitudeIsUint8}
                  freqBins={freqBins}
                  timeCols={timeCols}
                  cutoffs={cutoffs}
                  brightness={brightness}
                  normMax={normMax}
                  onColClick={jobId && audioDurationSec ? handleColClick : undefined}
                  displayWidth={displayWidth}
                  containerHeight={spectrogramH}
                  displayHeight={canvasDisplayHeight}
                />
                <div
                  ref={lineRef}
                  aria-hidden
                  style={{
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    width: 3,
                    height: totalH,
                    borderLeft: '3px dashed #86efac',
                    pointerEvents: 'none',
                    zIndex: 10,
                    willChange: 'transform', // hint browser to promote to compositor layer
                  }}
                />
                {songStructure && audioDurationSec != null && audioDurationSec > 0 && (
                  <SongStructureLanes
                    structure={songStructure}
                    timeCols={timeCols}
                    audioDurationSec={audioDurationSec}
                    displayWidth={displayWidth}
                    scrollRef={scrollRef}
                    activeRange={structureHighlight}
                    onSelectRange={onLaneSelect}
                  />
                )}
              </div>
              <HorizontalScrollbar scrollRef={scrollRef} contentWidth={displayWidth} />
              <XAxis timeCols={timeCols} audioDurationSec={audioDurationSec} displayWidth={displayWidth} scrollRef={scrollRef} />
              <p style={{ textAlign: 'center', fontSize: 11, color: '#374151', margin: '4px 0 0', fontFamily: 'system-ui' }}>
                Time (s)
              </p>
            </div>

            <ColorScaleBar height={spectrogramH} />
          </div>
        </div>

        {/* Right spacer — mirrors controls width to keep spectrogram truly centered */}
        <div style={{ width: spacerW, flexShrink: 0 }} aria-hidden />

      </div>
    </div>
  );
}
