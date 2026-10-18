import { useCallback, useEffect, useRef, useState } from 'react';
import { createJob } from '../api/jobs';
import { fetchJob } from '../api/jobState';
import { createJobFromLibrary } from '../api/library';
import { connectJobEvents } from '../api/jobEvents';
import { fetchPanel } from '../api/panel';
import type { SpectrogramViewerProps } from '../components/SpectrogramViewer/types';

type Status = 'idle' | 'uploading' | 'processing' | 'complete' | 'error';

export interface GplJobState {
  status: Status;
  jobId: string | null;
  progress: { done: number; total: number } | null;
  panels: SpectrogramViewerProps[];
  error: string | null;
}

export interface UseGplJobOptions {
  /** Called when a library job finishes and a .gpl cache may now exist on disk. */
  onLibraryJobComplete?: (fileId: string) => void;
  /** Called when a persisted jobId is gone (404 / backend restart). */
  onSessionExpired?: () => void;
}

const STORAGE_KEY = 'spectrogram-viewer.job-session';
const STORAGE_TTL_MS = 6 * 60 * 60 * 1000;

interface StoredJobSession {
  savedAt: number;
  jobId: string;
  pendingLibraryFileId: string | null;
}

function readStoredSession(): StoredJobSession | null {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as Partial<StoredJobSession>;
    if (
      typeof parsed.savedAt !== 'number'
      || typeof parsed.jobId !== 'string'
      || !('pendingLibraryFileId' in parsed)
    ) {
      window.localStorage.removeItem(STORAGE_KEY);
      return null;
    }
    if (Date.now() - parsed.savedAt > STORAGE_TTL_MS) {
      window.localStorage.removeItem(STORAGE_KEY);
      return null;
    }
    return {
      savedAt: parsed.savedAt,
      jobId: parsed.jobId,
      pendingLibraryFileId:
        typeof parsed.pendingLibraryFileId === 'string' ? parsed.pendingLibraryFileId : null,
    };
  } catch {
    window.localStorage.removeItem(STORAGE_KEY);
    return null;
  }
}

function writeStoredSession(jobId: string | null, pendingLibraryFileId: string | null) {
  try {
    if (!jobId) {
      window.localStorage.removeItem(STORAGE_KEY);
      return;
    }
    const payload: StoredJobSession = {
      savedAt: Date.now(),
      jobId,
      pendingLibraryFileId,
    };
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
  } catch {
    // Best-effort persistence only.
  }
}

export function useGplJob(options: UseGplJobOptions = {}) {
  const [state, setState] = useState<GplJobState>({
    status: 'idle',
    jobId: null,
    progress: null,
    panels: [],
    error: null,
  });

  const disconnectRef = useRef<(() => void) | null>(null);
  const pendingLibraryFileIdRef = useRef<string | null>(null);
  const hasRestoredSessionRef = useRef(false);
  const sessionPersistReadyRef = useRef(false);
  const onLibraryJobCompleteRef = useRef(options.onLibraryJobComplete);
  onLibraryJobCompleteRef.current = options.onLibraryJobComplete;
  const onSessionExpiredRef = useRef(options.onSessionExpired);
  onSessionExpiredRef.current = options.onSessionExpired;

  useEffect(() => {
    // Avoid wiping a just-read session with jobId=null before restore finishes.
    if (!hasRestoredSessionRef.current || !sessionPersistReadyRef.current) return;
    writeStoredSession(state.jobId, pendingLibraryFileIdRef.current);
  }, [state.jobId]);

  const _loadExistingResults = useCallback(async (jobId: string) => {
    try {
      const panel = await fetchPanel(jobId);
      setState(s => {
        if (s.jobId !== jobId || s.panels.length > 0) return s;
        return { ...s, panels: [{ ...panel, jobId }] };
      });
    } catch {
      // Panel may not be ready yet.
    }
  }, []);

  const _startJobEvents = useCallback((jobId: string) => {
    const disconnect = connectJobEvents(jobId, {
      onStatus(status) {
        if (status === 'queued' || status === 'processing') {
          setState(s => ({ ...s, status: 'processing' }));
        } else if (status === 'complete') {
          setState(s => ({ ...s, status: 'complete' }));
        } else if (status === 'error') {
          setState(s => ({ ...s, status: 'error' }));
        }
      },
      onProgress(done, total) {
        setState(s => ({ ...s, progress: { done, total } }));
      },
      onPanelReady() {
        fetchPanel(jobId)
          .then(props => {
            setState(s => {
              if (s.panels.some(panel => panel.jobId === jobId)) return s;
              return { ...s, panels: [{ ...props, jobId: s.jobId ?? jobId }] };
            });
          })
          .catch(err => console.error('fetchPanel error:', err));
      },
      onComplete() {
        void _loadExistingResults(jobId);
        const fileId = pendingLibraryFileIdRef.current;
        if (fileId) {
          onLibraryJobCompleteRef.current?.(fileId);
          pendingLibraryFileIdRef.current = null;
        }
        setState(s => ({ ...s, status: 'complete' }));
      },
      onError(message: string) {
        pendingLibraryFileIdRef.current = null;
        setState(s => ({ ...s, status: 'error', error: message }));
      },
    });
    disconnectRef.current = disconnect;
  }, [_loadExistingResults]);

  const _reset = useCallback(() => {
    disconnectRef.current?.();
    disconnectRef.current = null;
    pendingLibraryFileIdRef.current = null;
    writeStoredSession(null, null);
  }, []);

  const upload = useCallback(async (file: File) => {
    _reset();
    setState({ status: 'uploading', jobId: null, progress: null, panels: [], error: null });
    try {
      const { jobId } = await createJob(file);
      setState(s => ({ ...s, status: 'processing', jobId }));
      _startJobEvents(jobId);
    } catch (err) {
      setState(s => ({ ...s, status: 'error', error: String(err) }));
    }
  }, [_reset, _startJobEvents]);

  const processLibraryFile = useCallback(async (fileId: string) => {
    _reset();
    pendingLibraryFileIdRef.current = fileId;
    writeStoredSession(null, fileId);
    setState({ status: 'processing', jobId: null, progress: null, panels: [], error: null });
    try {
      const { jobId } = await createJobFromLibrary(fileId);
      setState(s => ({ ...s, jobId }));
      _startJobEvents(jobId);
    } catch (err) {
      pendingLibraryFileIdRef.current = null;
      setState(s => ({ ...s, status: 'error', error: String(err) }));
    }
  }, [_reset, _startJobEvents]);

  const reset = useCallback(() => {
    _reset();
    setState({ status: 'idle', jobId: null, progress: null, panels: [], error: null });
  }, [_reset]);

  useEffect(() => {
    const saved = readStoredSession();
    hasRestoredSessionRef.current = true;
    if (!saved) {
      sessionPersistReadyRef.current = true;
      return;
    }

    // Keep the dead jobId out of React state until the server confirms it.
    // Setting status=processing first made refresh look stuck when the backend
    // had restarted and returned 404 for the persisted id.
    pendingLibraryFileIdRef.current = saved.pendingLibraryFileId;
    let cancelled = false;
    void fetchJob(saved.jobId)
      .then(job => {
        if (cancelled) return;
        if (job.status === 'error') {
          pendingLibraryFileIdRef.current = null;
          writeStoredSession(null, null);
          setState(s => ({ ...s, status: 'error', error: job.error ?? 'unknown error' }));
          return;
        }
        setState(s => ({
          ...s,
          status: job.status === 'complete' ? 'complete' : 'processing',
          jobId: job.jobId,
          error: job.error,
        }));
        void _loadExistingResults(job.jobId);
        if (job.status !== 'complete') {
          _startJobEvents(job.jobId);
        }
      })
      .catch(() => {
        // Always drop the stale session — even if this request was superseded —
        // so a 404 cannot leave spectrogram-viewer.job-session pointing at a dead job.
        writeStoredSession(null, null);
        pendingLibraryFileIdRef.current = null;
        onSessionExpiredRef.current?.();
        if (cancelled) return;
        setState(s => ({
          ...s,
          status: 'idle',
          jobId: null,
          progress: null,
          panels: [],
          error: null,
        }));
      })
      .finally(() => {
        if (!cancelled) sessionPersistReadyRef.current = true;
      });
    return () => {
      cancelled = true;
    };
  }, [_loadExistingResults, _startJobEvents]);

  return { ...state, upload, processLibraryFile, reset };
}
