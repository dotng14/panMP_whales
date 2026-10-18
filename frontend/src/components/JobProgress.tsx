import type { GplJobState } from '../hooks/useGplJob';

type Props = Pick<GplJobState, 'status' | 'progress' | 'error'> & { onReset: () => void };

const STATUS_LABEL: Record<string, string> = {
  idle: 'Ready',
  uploading: 'Uploading…',
  processing: 'Processing…',
  complete: 'Complete',
  error: 'Error',
};

export function JobProgress({ status, progress, error, onReset }: Props) {
  if (status === 'idle') return null;

  const pct =
    progress && progress.total > 0
      ? Math.round((progress.done / progress.total) * 100)
      : null;

  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 12,
        fontSize: 13,
        color: status === 'error' ? '#dc2626' : '#444',
      }}
    >
      <span style={{ fontWeight: 600 }}>{STATUS_LABEL[status] ?? status}</span>

      {pct !== null && status === 'processing' && (
        <div
          style={{
            width: 160,
            height: 8,
            background: '#e5e7eb',
            borderRadius: 4,
            overflow: 'hidden',
          }}
        >
          <div
            style={{
              width: `${pct}%`,
              height: '100%',
              background: '#2563eb',
              transition: 'width 0.2s',
            }}
          />
        </div>
      )}

      {pct !== null && (
        <span style={{ minWidth: 36 }}>{pct}%</span>
      )}

      {error && <span>{error}</span>}

      {(status === 'complete' || status === 'error') && (
        <button
          onClick={onReset}
          style={{
            padding: '3px 10px',
            border: '1px solid #aaa',
            borderRadius: 4,
            background: '#fff',
            cursor: 'pointer',
            fontSize: 12,
          }}
        >
          Reset
        </button>
      )}
    </div>
  );
}
