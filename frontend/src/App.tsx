import { useEffect, useState } from 'react';
import { SpectrogramViewer } from './components/SpectrogramViewer';
import { WavFileUpload } from './components/WavFileUpload';
import { WavLibraryPicker } from './components/WavLibraryPicker';
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
  const { status, progress, panels, error, upload, processLibraryFile, reset } = useGplJob({
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
      <h1 className="text-gray-900 text-lg font-semibold mb-4 font-mono">Spectrogram Viewer</h1>

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
                  <WavLibraryPicker
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
            <SpectrogramViewer {...panel} analysis={INPUT_MODE === 'upload' ? undefined : analysis} />
          </section>
        ))
      )}

      {!USE_MOCK && panels.length === 0 && status === 'idle' && INPUT_MODE === 'upload' && (
        <p style={{ color: '#888', fontSize: 13 }}>Upload a WAV file to begin processing.</p>
      )}
    </div>
  );
}
