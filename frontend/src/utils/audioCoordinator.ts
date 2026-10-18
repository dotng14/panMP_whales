type PlaybackListener = (activePlayerId: string) => void;

const listeners = new Set<PlaybackListener>();

export function announcePlayback(activePlayerId: string): void {
  for (const listener of listeners) listener(activePlayerId);
}

export function subscribeToPlayback(listener: PlaybackListener): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}
