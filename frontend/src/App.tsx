import { useEffect, useState } from 'react';
import { SpectrogramViewer } from './components/SpectrogramViewer';
import { RepeatSpectrogram } from './components/RepeatSpectrogram';
import { RepeatActionForm } from './components/RepeatSpectrogram/RepeatActionForm';
import { SpectrogramButton } from './components/SpectrogramViewer/PlayButton';
import { useSpectrogramScale } from './components/SpectrogramViewer/useSpectrogramScale';
import type { SpectrogramViewerProps } from './components/SpectrogramViewer/types';
import { fetchRepeats, type RepeatMatches } from './api/repeats';
import { WavFileUpload } from './components/WavFileUpload';
import { RecordingPicker } from './components/RecordingPicker';
import { ScanViewer } from './components/ScanViewer';
import { AnalysisPicker } from './components/AnalysisPicker';
import { describeChoice, type AnalysisChoice } from './api/catalog';
import { JobProgress } from './components/JobProgress';
import { useGplJob } from './hooks/useGplJob';
import { generateMockData } from './utils/mockData';

const USE_MOCK = import.meta.env.VITE_USE_MOCK === 'true';
const INPUT_MODE = import.meta.env.VITE_INPUT_MODE ?? 'library';
const UI_SESSION_KEY = 'spectrogram-viewer.ui-session';
const UI_SESSION_TTL_MS = 6 * 60 * 60 * 1000;

const mockData = USE_MOCK ? generateMockData(0) : null;

function isOverlayRef(value: unknown): boolean {
  if (value === null) return true;
  if (typeof value !== 'object' || value === undefined) return false;
  const v = value as Record<string, unknown>;
  return typeof v.id === 'number' && typeof v.label === 'string';
}

// A session saved before the picker took a dictionary and a CSV together
// fails this check and is dropped (the picker then starts from its defaults).
function isAnalysisChoice(value: unknown): value is AnalysisChoice {
  if (typeof value !== 'object' || value === null) return false;
  const v = value as Record<string, unknown>;
  return 'dictionary' in v && 'csv' in v && isOverlayRef(v.dictionary) && isOverlayRef(v.csv);
}

/**
 * One spectrogram panel. With a job and a chosen dictionary, the repeat-review
 * strip is shown under it; both share one zoom scale so a single set of controls drives them.
 */
function PanelSection({
  panel,
  analysis,
  jobId,
  showRepeats,
  onDictionaryChanged,
  onCatalogRefresh,
}: {
  panel: SpectrogramViewerProps;
  analysis: AnalysisChoice | undefined;
  jobId: string | null;
  showRepeats: boolean;
  /** An Actions run produced a dictionary: switch the analysis onto it. */
  onDictionaryChanged?: (newDictionaryId: number) => void;
  /** The catalog gained a dictionary: refresh pickers that list it. */
  onCatalogRefresh?: () => void;
}) {
  const [actionNotice, setActionNotice] = useState<string | null>(null);
  const [actionsOpen, setActionsOpen] = useState(false);
  const scale = useSpectrogramScale({
    audioDurationSec: panel.audioDurationSec,
    freqLo: panel.freqLo,
    freqHi: panel.freqHi,
  });
  const [matches, setMatches] = useState<RepeatMatches | null>(null);
  const [repeatsError, setRepeatsError] = useState<string | null>(null);
  const [highlightedRange, setHighlightedRange] = useState<{ startCol: number; endCol: number } | null>(null);
  const dictionaryId = analysis?.dictionary?.id;
  const effectiveJobId = panel.jobId ?? jobId;
  const repeatsWanted = showRepeats && effectiveJobId != null && dictionaryId !== undefined;

  useEffect(() => {
    setMatches(null);
    setRepeatsError(null);
    setHighlightedRange(null);
    if (!repeatsWanted || effectiveJobId == null || dictionaryId === undefined) return;
    let cancelled = false;
    fetchRepeats(effectiveJobId, dictionaryId)
      .then(data => { if (!cancelled) setMatches(data); })
      .catch(err => { if (!cancelled) setRepeatsError(err instanceof Error ? err.message : String(err)); });
    return () => {
      cancelled = true;
    };
  }, [repeatsWanted, effectiveJobId, dictionaryId]);

  // Switching dictionaries refetches the repeats (the effect above), which
  // remounts the review strip and its Actions form; keep a note of what happened.
  const handleDictionaryChanged = (newDictionaryId: number) => {
    setActionNotice(`Actions built dictionary ${newDictionaryId}; now showing its repeats.`);
    onDictionaryChanged?.(newDictionaryId);
  };

  return (
    <>
      <SpectrogramViewer
        {...panel}
        analysis={analysis}
        scale={scale}
        highlightedRange={highlightedRange}
        repeats={repeatsWanted ? matches : null}
      />
      {showRepeats && effectiveJobId != null && !repeatsWanted && (
        <div
          style={{
            marginTop: 12,
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            gap: 6,
          }}
        >
          <SpectrogramButton
            onClick={() => setActionsOpen(true)}
            title="Dictionary actions: randomized builder and action history"
            style={{ padding: '6px 12px' }}
          >
            Actions
          </SpectrogramButton>
          <span style={{ fontSize: 12, color: '#6b7280' }}>
            Build a dictionary (randomized_builder) or view history for this recording.
          </span>
          <RepeatActionForm
            open={actionsOpen}
            jobId={effectiveJobId}
            dictionaryId={null}
            onDictionaryChanged={newId => {
              setActionsOpen(false);
              onCatalogRefresh?.();
              handleDictionaryChanged(newId);
            }}
            onClose={() => setActionsOpen(false)}
          />
        </div>
      )}
      {actionNotice && (
        <p role="status" style={{ color: '#166534', fontSize: 12, marginTop: 8 }}>{actionNotice}</p>
      )}
      {repeatsWanted && repeatsError && (
        <p role="alert" style={{ color: '#b91c1c', fontSize: 12, marginTop: 8 }}>
          Repeated matches unavailable: {repeatsError}
        </p>
      )}
      {repeatsWanted && matches && effectiveJobId != null && dictionaryId !== undefined && (
        <div style={{ marginTop: 16 }}>
          <RepeatSpectrogram
            jobId={effectiveJobId}
            dictionaryId={dictionaryId}
            source={panel}
            matches={matches}
            scale={scale}
            onShowOriginal={setHighlightedRange}
            onMatchesChange={setMatches}
            onDictionaryChanged={handleDictionaryChanged}
            onCatalogRefresh={onCatalogRefresh}
          />
        </div>
      )}
    </>
  );
}

export default function App() {
  const [restoredUi] = useState(() => {
    try {
      const raw = window.localStorage.getItem(UI_SESSION_KEY);
      if (!raw) return null;
      const parsed = JSON.parse(raw) as {
        savedAt?: number;
        pickerCollapsed?: boolean;
        selectedFile?: { fileId: string; name: string } | null;
        analysis?: AnalysisChoice;
        running?: { fileName: string; overlay: string } | null;
      };
      if (typeof parsed.savedAt !== 'number' || Date.now() - parsed.savedAt > UI_SESSION_TTL_MS) {
        window.localStorage.removeItem(UI_SESSION_KEY);
        return null;
      }
      return {
        pickerCollapsed: Boolean(parsed.pickerCollapsed),
        selectedFile:
          parsed.selectedFile
          && typeof parsed.selectedFile.fileId === 'string'
          && typeof parsed.selectedFile.name === 'string'
            ? parsed.selectedFile
            : null,
        analysis: isAnalysisChoice(parsed.analysis) ? parsed.analysis : undefined,
        running:
          parsed.running
          && typeof parsed.running.fileName === 'string'
          && typeof parsed.running.overlay === 'string'
            ? parsed.running
            : null,
      };
    } catch {
      window.localStorage.removeItem(UI_SESSION_KEY);
      return null;
    }
  });
  const [libraryCacheRefresh, setLibraryCacheRefresh] = useState<{
    fileId: string;
    nonce: number;
  } | null>(null);
  const [pickerCollapsed, setPickerCollapsed] = useState(restoredUi?.pickerCollapsed ?? false);
  // Step 1 of the library flow: which WAV the overlay picker is showing for.
  const [selectedFile, setSelectedFile] = useState<{ fileId: string; name: string } | null>(
    restoredUi?.selectedFile ?? null,
  );
  // Step 2: the overlay (dictionary, CSV, or null for none); undefined until chosen.
  const [analysis, setAnalysis] = useState<AnalysisChoice | undefined>(restoredUi?.analysis);
  // What is running, for the collapsed summary bar.
  const [running, setRunning] = useState<{ fileName: string; overlay: string } | null>(
    restoredUi?.running ?? null,
  );
  const [activeTab, setActiveTab] = useState<'viewer' | 'scan'>('viewer');
  const { status, jobId, progress, panels, error, upload, processLibraryFile, reset } = useGplJob({
    onLibraryJobComplete: fileId =>
      setLibraryCacheRefresh({ fileId, nonce: Date.now() }),
    onSessionExpired: () => {
      // Dead job after refresh/backend restart — show the picker again.
      setPickerCollapsed(false);
      setRunning(null);
    },
  });

  const isProcessing = status === 'uploading' || status === 'processing';
  const isPickerHidden = status !== 'idle' && pickerCollapsed;

  useEffect(() => {
    try {
      if (!isPickerHidden && !selectedFile && !running) {
        window.localStorage.removeItem(UI_SESSION_KEY);
        return;
      }
      window.localStorage.setItem(UI_SESSION_KEY, JSON.stringify({
        savedAt: Date.now(),
        pickerCollapsed,
        selectedFile,
        analysis,
        running,
      }));
    } catch {
      // Best-effort persistence only.
    }
  }, [isPickerHidden, pickerCollapsed, selectedFile, analysis, running]);

  // Auto-minimize the file picker once processing starts, freeing up screen space.
  const handleUpload = (file: File) => {
    setPickerCollapsed(true);
    upload(file);
  };
  // Both menus collapse here: the user has chosen a file and what to overlay on it.
  const handleChooseAnalysis = (choice: AnalysisChoice) => {
    if (!selectedFile) return;
    setAnalysis(choice);
    setRunning({ fileName: selectedFile.name, overlay: describeChoice(choice) });
    setPickerCollapsed(true);
    processLibraryFile(selectedFile.fileId);
  };
  // An Actions run built a new dictionary: make it the chosen overlay (keeping the
  // CSV), which refetches the repeats, and refresh the picker's catalog list.
  const handleDictionaryChanged = (newDictionaryId: number) => {
    const next: AnalysisChoice = {
      csv: analysis?.csv ?? null,
      dictionary: { id: newDictionaryId, label: `dictionary ${newDictionaryId}` },
    };
    setAnalysis(next);
    setRunning(prev => (prev ? { ...prev, overlay: describeChoice(next) } : prev));
  };
  const handleCatalogRefresh = () => {
    if (selectedFile) setLibraryCacheRefresh({ fileId: selectedFile.fileId, nonce: Date.now() });
  };
  // Re-expand once the user resets, so they can pick another file.
  const handleReset = () => {
    setPickerCollapsed(false);
    setSelectedFile(null);
    setAnalysis(undefined);
    setRunning(null);
    reset();
  };

  return (
    <div className="p-2 md:p-4">
      <div style={{ display: 'flex', alignItems: 'center', gap: 16, marginBottom: 16 }}>
        <h1 className="text-gray-900 text-lg font-semibold font-mono" style={{ margin: 0 }}>
          Spectrogram Viewer
        </h1>
        {!USE_MOCK && (
          <div
            role="tablist"
            style={{
              display: 'inline-flex',
              borderRadius: 6,
              overflow: 'hidden',
              border: '1px solid #d1d5db',
            }}
          >
            {(
              [
                ['viewer', 'Recordings'],
                ['scan', '20-Year Scan'],
              ] as const
            ).map(([tab, label], idx) => {
              const active = activeTab === tab;
              return (
                <button
                  key={tab}
                  type="button"
                  role="tab"
                  aria-selected={active}
                  onClick={() => setActiveTab(tab)}
                  style={{
                    padding: '4px 12px',
                    border: 'none',
                    borderLeft: idx > 0 ? '1px solid #d1d5db' : 'none',
                    background: active ? '#1d4ed8' : '#fff',
                    color: active ? '#fff' : '#374151',
                    fontSize: 13,
                    fontWeight: active ? 600 : 400,
                    cursor: 'pointer',
                  }}
                >
                  {label}
                </button>
              );
            })}
          </div>
        )}
      </div>

      {activeTab === 'scan' && !USE_MOCK ? (
        <ScanViewer />
      ) : (
        <>
      {!USE_MOCK && (
        <div style={{ marginBottom: 20 }}>
          {status !== 'idle' && (
            <button
              onClick={() => setPickerCollapsed(c => !c)}
              style={{
                marginBottom: isPickerHidden ? 0 : 8,
                padding: '3px 10px',
                borderRadius: 4,
                border: '1px solid #d1d5db',
                background: '#f9fafb',
                color: '#374151',
                cursor: 'pointer',
                fontSize: 12,
              }}
            >
              {isPickerHidden
                ? running
                  ? `🎵 ${running.fileName} · ${running.overlay} — change`
                  : '📂 Show file picker'
                : '📂 Hide file picker'}
            </button>
          )}
          {/* Hidden rather than unmounted: collapsing must not lose the browsed
              directory, the selected file, or the overlay picker under it. */}
          <div style={{ display: isPickerHidden ? 'none' : 'block' }}>
            {INPUT_MODE === 'upload'
              ? <WavFileUpload onFile={handleUpload} disabled={isProcessing} />
              : (
                <>
                  <RecordingPicker
                    onSelectFile={setSelectedFile}
                    selectedFileId={selectedFile?.fileId ?? null}
                    disabled={isProcessing}
                    libraryCacheRefresh={libraryCacheRefresh}
                  />
                  {selectedFile && (
                    <AnalysisPicker
                      key={`${selectedFile.fileId}:${libraryCacheRefresh?.nonce ?? 0}`}
                      fileId={selectedFile.fileId}
                      fileName={selectedFile.name}
                      initialChoice={analysis}
                      onChoose={handleChooseAnalysis}
                      onBack={() => setSelectedFile(null)}
                      disabled={isProcessing}
                    />
                  )}
                </>
              )}
          </div>
          <div style={{ marginTop: 10 }}>
            <JobProgress status={status} progress={progress} error={error} onReset={handleReset} />
          </div>
        </div>
      )}

      {USE_MOCK && mockData ? (
        <section className="mb-10">
          <SpectrogramViewer {...mockData} />
        </section>
      ) : (
        panels.map((panel, i) => (
          <section key={panel.title ?? i} className="mb-10">
            <PanelSection
              panel={panel}
              analysis={INPUT_MODE === 'upload' ? undefined : analysis}
              jobId={jobId}
              showRepeats={i === 0}
              onDictionaryChanged={handleDictionaryChanged}
              onCatalogRefresh={handleCatalogRefresh}
            />
          </section>
        ))
      )}

      {!USE_MOCK && panels.length === 0 && status === 'idle' && INPUT_MODE === 'upload' && (
        <p style={{ color: '#888', fontSize: 13 }}>Upload a WAV file to begin processing.</p>
      )}
        </>
      )}
    </div>
  );
}
