export interface JobCreated {
  jobId: string;
  eventsUrl: string;
}

export interface JobInfo {
  jobId: string;
  status: 'queued' | 'processing' | 'complete' | 'error';
  filename: string;
  manifest: Record<string, unknown> | null;
  error: string | null;
}

export interface ArrayMeta {
  name: string;
  dtype: 'float32' | 'uint8';
  length: number;
}

export interface PanelProps {
  panelVersion?: number;
  freqBins: number;
  timeCols: number;
  cutoffs: number[];
  title: string;
  colormap: 'jet' | 'viridis';
  brightness: number;
  normMax?: number;
  audioDurationSec?: number;
  freqLo?: number;
  freqHi?: number;
}

export interface WsMessage {
  type: 'status' | 'manifest' | 'panel_ready' | 'progress' | 'complete' | 'error' | 'warning';
}

export interface WsStatus extends WsMessage { type: 'status'; status: string; }
export interface WsManifest extends WsMessage { type: 'manifest'; data: Record<string, unknown>; }
export interface WsPanelReady extends WsMessage { type: 'panel_ready'; }
export interface WsProgress extends WsMessage { type: 'progress'; done: number; total: number; }
export interface WsComplete extends WsMessage { type: 'complete'; jobId: string; }
export interface WsError extends WsMessage { type: 'error'; message: string; }
