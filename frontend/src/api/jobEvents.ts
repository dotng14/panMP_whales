import { API_BASE } from './base';
import type { WsMessage, WsManifest, WsComplete, WsError } from './types';

export interface JobEventHandlers {
  onStatus?: (status: string) => void;
  onManifest?: (data: WsManifest['data']) => void;
  onPanelReady?: () => void;
  onProgress?: (done: number, total: number) => void;
  onComplete?: (jobId: string) => void;
  onError?: (message: string) => void;
  onClose?: () => void;
}

export function connectJobEvents(jobId: string, handlers: JobEventHandlers): () => void {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const ws = new WebSocket(`${protocol}//${window.location.host}${API_BASE}/api/jobs/${jobId}/events`);

  ws.addEventListener('message', (ev) => {
    if (typeof ev.data !== 'string') return; // panel data now arrives via HTTP, not WS
    const msg = JSON.parse(ev.data) as WsMessage;
    switch (msg.type) {
      case 'status':      handlers.onStatus?.((msg as { type: 'status'; status: string }).status); break;
      case 'manifest':    handlers.onManifest?.((msg as WsManifest).data); break;
      case 'panel_ready': handlers.onPanelReady?.(); break;
      case 'complete':    handlers.onComplete?.((msg as WsComplete).jobId); break;
      case 'error':       handlers.onError?.((msg as WsError).message); break;
    }
  });

  ws.addEventListener('close', () => handlers.onClose?.());
  ws.addEventListener('error', () => handlers.onError?.('WebSocket connection error'));

  return () => ws.close();
}
