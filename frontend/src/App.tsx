import { useEffect, useState } from 'react';
import { SpectrogramViewer } from './components/SpectrogramViewer';
import { WavFileUpload } from './components/WavFileUpload';
import { WavLibraryPicker } from './components/WavLibraryPicker';
import { JobProgress } from './components/JobProgress';
import { useGplJob } from './hooks/useGplJob';
import { generateMockData } from './utils/mockData';

const USE_MOCK = import.meta.env.VITE_USE_MOCK === 'true';
const INPUT_MODE = import.meta.env.VITE_INPUT_MODE ?? 'library';
const UI_SESSION_KEY = 'spectrogram-viewer.ui-session';
const UI_SESSION_TTL_MS = 6 * 60 * 60 * 1000;

const mockData = USE_MOCK ? generateMockData(0) : null;

export default function App() {
  const [restoredUi] = useState(() => {
    try {
      const raw = window.localStorage.getItem(UI_SESSION_KEY);
      if (!raw) return null;
      const parsed = JSON.parse(raw) as {
        savedAt?: number;
        pickerCollapsed?: boolean;
        selectedFile?: { fileId: string; name: string } | null;
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
  const [selectedFile, setSelectedFile] = useState<{ fileId: string; name: string } | null>(
    restoredUi?.selectedFile ?? null,
  );
  const { status, progress, panels, error, upload, processLibraryFile, reset } = useGplJob({
    onLibraryJobComplete: fileId =>
      setLibraryCacheRefresh({ fileId, nonce: Date.now() }),
    onSessionExpired: () => {
      // Dead job after refresh/backend restart — show the picker again.
      setPickerCollapsed(false);
    },
  });

  const isProcessing = status === 'uploading' || status === 'processing';
  const isPickerHidden = status !== 'idle' && pickerCollapsed;

  useEffect(() => {
    try {
      if (!isPickerHidden && !selectedFile) {
        window.localStorage.removeItem(UI_SESSION_KEY);
        return;
      }
      window.localStorage.setItem(UI_SESSION_KEY, JSON.stringify({
        savedAt: Date.now(),
        pickerCollapsed,
        selectedFile,
      }));
    } catch {
      // Best-effort persistence only.
    }
  }, [isPickerHidden, pickerCollapsed, selectedFile]);

  // Auto-minimize the file picker once processing starts, freeing up screen space.
  const handleUpload = (file: File) => {
    setPickerCollapsed(true);
    upload(file);
  };
  // Picking a file immediately kicks off processing — no intermediate step.
  const handleSelectFile = (file: { fileId: string; name: string } | null) => {
    setSelectedFile(file);
    if (file) {
      setPickerCollapsed(true);
      processLibraryFile(file.fileId);
    }
  };
  // Re-expand once the user resets, so they can pick another file.
  const handleReset = () => {
    setPickerCollapsed(false);
    setSelectedFile(null);
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
                ? selectedFile
                  ? `🎵 ${selectedFile.name} — change`
                  : '📂 Show file picker'
                : '📂 Hide file picker'}
            </button>
          )}
          {/* Hidden rather than unmounted: collapsing must not lose the browsed
              directory or the selected file. */}
          <div style={{ display: isPickerHidden ? 'none' : 'block' }}>
            {INPUT_MODE === 'upload'
              ? <WavFileUpload onFile={handleUpload} disabled={isProcessing} />
              : (
                <WavLibraryPicker
                  onSelectFile={handleSelectFile}
                  selectedFileId={selectedFile?.fileId ?? null}
                  disabled={isProcessing}
                  libraryCacheRefresh={libraryCacheRefresh}
                />
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
            <SpectrogramViewer {...panel} />
          </section>
        ))
      )}

      {!USE_MOCK && panels.length === 0 && status === 'idle' && INPUT_MODE === 'upload' && (
        <p style={{ color: '#888', fontSize: 13 }}>Upload a WAV file to begin processing.</p>
      )}
    </div>
  );
}
