import { useEffect, useRef, useState } from 'react';
import { formatHistoryTimeSec, type ViewHistoryEntry } from '../utils/viewHistory';

interface Props {
  entries: ViewHistoryEntry[];
  onSelect: (entry: ViewHistoryEntry) => void;
  onRemove: (id: string) => void;
  onClear: () => void;
  disabled?: boolean;
}

function formatAgo(ts: number): string {
  const diffSec = Math.max(0, Math.floor((Date.now() - ts) / 1000));
  if (diffSec < 60) return 'just now';
  const mins = Math.floor(diffSec / 60);
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  return `${Math.floor(hrs / 24)}d ago`;
}

export function HistoryButton({ entries, onSelect, onRemove, onClear, disabled }: Props) {
  const [open, setOpen] = useState(false);
  const [filter, setFilter] = useState('');
  const rootRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (rootRef.current && !rootRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(false);
    };
    window.addEventListener('mousedown', onDown);
    window.addEventListener('keydown', onKey);
    return () => {
      window.removeEventListener('mousedown', onDown);
      window.removeEventListener('keydown', onKey);
    };
  }, [open]);

  const q = filter.trim().toLowerCase();
  const visible = q
    ? entries.filter(
      e =>
        e.fileName.toLowerCase().includes(q)
        || formatHistoryTimeSec(e.timeSec).includes(q)
        || (e.context ?? '').toLowerCase().includes(q),
    )
    : entries;

  return (
    <div ref={rootRef} style={{ position: 'relative', display: 'inline-block' }}>
      <button
        type="button"
        disabled={disabled}
        onClick={() => setOpen(v => !v)}
        aria-expanded={open}
        aria-haspopup="listbox"
        title="Jump back to a previously visited recording and time"
        style={{
          padding: '3px 10px',
          borderRadius: 4,
          border: '1px solid #d1d5db',
          background: open ? '#eff6ff' : '#f9fafb',
          color: open ? '#1d4ed8' : '#374151',
          cursor: disabled ? 'not-allowed' : 'pointer',
          fontSize: 12,
          fontFamily: 'inherit',
        }}
      >
        🕒 History{entries.length > 0 ? ` (${entries.length})` : ''}
      </button>

      {open && (
        <div
          role="dialog"
          aria-label="Viewing history"
          style={{
            position: 'absolute',
            top: 'calc(100% + 6px)',
            left: 0,
            zIndex: 50,
            width: 420,
            maxWidth: 'calc(100vw - 24px)',
            background: '#fff',
            border: '1px solid #d1d5db',
            borderRadius: 8,
            boxShadow: '0 10px 25px rgba(0,0,0,0.12)',
            overflow: 'hidden',
          }}
        >
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              padding: '8px 10px',
              borderBottom: '1px solid #e5e7eb',
              background: '#f9fafb',
              gap: 8,
            }}
          >
            <span style={{ fontSize: 12, fontWeight: 600, color: '#111827' }}>
              Observed Times History
            </span>
            {entries.length > 0 && (
              <button
                type="button"
                onClick={() => { onClear(); setFilter(''); }}
                style={{
                  border: 'none',
                  background: 'transparent',
                  color: '#6b7280',
                  fontSize: 11,
                  cursor: 'pointer',
                  padding: '2px 4px',
                }}
              >
                Clear all
              </button>
            )}
          </div>

          {entries.length > 5 && (
            <div style={{ padding: '6px 10px', borderBottom: '1px solid #f3f4f6' }}>
              <input
                type="text"
                value={filter}
                onChange={e => setFilter(e.target.value)}
                placeholder="Filter by file or time…"
                style={{
                  width: '100%',
                  boxSizing: 'border-box',
                  fontSize: 12,
                  padding: '4px 8px',
                  border: '1px solid #d1d5db',
                  borderRadius: 4,
                }}
              />
            </div>
          )}

          <div style={{ maxHeight: 320, overflowY: 'auto' }}>
            {visible.length === 0 ? (
              <p style={{ margin: 0, padding: '16px 12px', fontSize: 12, color: '#6b7280', textAlign: 'center' }}>
                {entries.length === 0
                  ? 'No history yet. Click the spectrogram, jump to a song, or pause playback to save a time.'
                  : 'No matching entries.'}
              </p>
            ) : (
              visible.map(entry => (
                <div
                  key={entry.id}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    borderBottom: '1px solid #f3f4f6',
                  }}
                >
                  <button
                    type="button"
                    onClick={() => {
                      onSelect(entry);
                      setOpen(false);
                    }}
                    style={{
                      flex: 1,
                      minWidth: 0,
                      textAlign: 'left',
                      border: 'none',
                      background: 'transparent',
                      padding: '8px 10px',
                      cursor: 'pointer',
                      display: 'flex',
                      flexDirection: 'column',
                      gap: 2,
                    }}
                    onMouseEnter={e => { e.currentTarget.style.background = '#f3f4f6'; }}
                    onMouseLeave={e => { e.currentTarget.style.background = 'transparent'; }}
                  >
                    <span
                      style={{
                        fontSize: 12,
                        fontFamily: 'monospace',
                        color: '#111827',
                        whiteSpace: 'nowrap',
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                      }}
                    >
                      {entry.fileName}
                    </span>
                    <span style={{ fontSize: 11, color: '#4b5563', display: 'flex', gap: 8, alignItems: 'center' }}>
                      <strong style={{ color: '#1d4ed8', fontFamily: 'monospace' }}>
                        @ {formatHistoryTimeSec(entry.timeSec)}
                      </strong>
                      {entry.context && (
                        <span
                          style={{
                            background: '#eef2ff',
                            color: '#4338ca',
                            padding: '0 5px',
                            borderRadius: 3,
                            fontSize: 10,
                          }}
                        >
                          {entry.context}
                        </span>
                      )}
                      <span style={{ marginLeft: 'auto', color: '#9ca3af', fontSize: 10 }}>
                        {formatAgo(entry.visitedAt)}
                      </span>
                    </span>
                  </button>
                  <button
                    type="button"
                    aria-label="Remove history entry"
                    onClick={() => onRemove(entry.id)}
                    style={{
                      border: 'none',
                      background: 'transparent',
                      color: '#9ca3af',
                      cursor: 'pointer',
                      padding: '6px 10px',
                      fontSize: 14,
                    }}
                  >
                    ×
                  </button>
                </div>
              ))
            )}
          </div>
        </div>
      )}
    </div>
  );
}
