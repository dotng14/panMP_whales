import { useEffect, useState } from 'react';
import { getHybridDictionary, scanImageUrl, type HybridDictionary } from '../api/scan';

export function HybridDictionaryStrip() {
  const [dict, setDict] = useState<HybridDictionary | null>(null);
  const [zoomedIdx, setZoomedIdx] = useState<number | null>(null);

  useEffect(() => {
    let cancelled = false;
    getHybridDictionary()
      .then(data => {
        if (!cancelled && data.available) setDict(data);
      })
      .catch(() => {
        // Best-effort: hide strip if model isn't available.
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (zoomedIdx === null || !dict) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setZoomedIdx(null);
      else if (e.key === 'ArrowLeft') setZoomedIdx(i => (i !== null && i > 0 ? i - 1 : i));
      else if (e.key === 'ArrowRight')
        setZoomedIdx(i => (i !== null && i < dict.motifs.length - 1 ? i + 1 : i));
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [zoomedIdx, dict]);

  if (!dict || dict.motifs.length === 0) return null;

  const zoomed = zoomedIdx !== null ? dict.motifs[zoomedIdx] ?? null : null;

  return (
    <section
      style={{
        marginTop: 16,
        border: '1px solid #d1d5db',
        borderRadius: 8,
        background: '#fff',
        overflow: 'hidden',
      }}
    >
      <div
        style={{
          padding: '8px 12px',
          background: '#f9fafb',
          borderBottom: '1px solid #d1d5db',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          fontSize: 12,
          color: '#374151',
        }}
      >
        <strong>
          {dict.model ?? '2023'} Motif Dictionary ({dict.motifs.length} motifs)
        </strong>
        {dict.freqLo != null && dict.freqHi != null && (
          <span style={{ color: '#6b7280' }}>
            {dict.freqLo}–{dict.freqHi} Hz · click a motif to zoom
          </span>
        )}
      </div>
      <div
        style={{
          padding: 12,
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fill, minmax(84px, 1fr))',
          gap: 8,
        }}
      >
        {dict.motifs.map((m, idx) => (
          <button
            key={m.id}
            type="button"
            onClick={() => setZoomedIdx(idx)}
            title={`Click to zoom Motif ${m.id} (${m.durationSec}s)`}
            style={{
              padding: 0,
              border: '1px solid #e5e7eb',
              borderRadius: 6,
              overflow: 'hidden',
              background: '#111827',
              cursor: 'pointer',
              textAlign: 'left',
            }}
          >
            <img
              src={scanImageUrl(m.imageUrl)}
              alt={`Motif ${m.id}`}
              loading="lazy"
              style={{
                width: '100%',
                height: 76,
                objectFit: 'fill',
                display: 'block',
              }}
            />
            <div
              style={{
                padding: '2px 5px',
                background: '#f9fafb',
                borderTop: '1px solid #e5e7eb',
                display: 'flex',
                justifyContent: 'space-between',
                fontSize: 11,
                fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Consolas, monospace',
                color: '#111827',
              }}
            >
              <span>M{m.id}</span>
              <span style={{ color: '#6b7280' }}>{m.durationSec}s</span>
            </div>
          </button>
        ))}
      </div>

      {zoomed && zoomedIdx !== null && (
        <div
          role="dialog"
          aria-modal="true"
          aria-label={`Motif ${zoomed.id} zoomed view`}
          onClick={() => setZoomedIdx(null)}
          style={{
            position: 'fixed',
            inset: 0,
            zIndex: 1000,
            background: 'rgba(17, 24, 39, 0.72)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            padding: 20,
          }}
        >
          <div
            onClick={e => e.stopPropagation()}
            style={{
              background: '#fff',
              borderRadius: 10,
              boxShadow: '0 20px 50px rgba(0, 0, 0, 0.35)',
              maxWidth: 1000,
              width: '100%',
              overflow: 'hidden',
            }}
          >
            <div
              style={{
                padding: '10px 14px',
                background: '#f9fafb',
                borderBottom: '1px solid #e5e7eb',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                gap: 10,
              }}
            >
              <div style={{ fontSize: 14, fontWeight: 600, color: '#111827' }}>
                Motif M{zoomed.id}{' '}
                <span style={{ fontWeight: 400, fontSize: 12, color: '#6b7280' }}>
                  ({zoomed.durationSec}s · {zoomed.widthCols} frames
                  {dict.freqLo != null && dict.freqHi != null
                    ? ` · ${dict.freqLo}–${dict.freqHi} Hz`
                    : ''}
                  )
                </span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <button
                  type="button"
                  disabled={zoomedIdx <= 0}
                  onClick={() => setZoomedIdx(zoomedIdx - 1)}
                  style={{
                    padding: '3px 8px',
                    borderRadius: 4,
                    border: '1px solid #d1d5db',
                    background: zoomedIdx <= 0 ? '#f3f4f6' : '#fff',
                    cursor: zoomedIdx <= 0 ? 'default' : 'pointer',
                    fontSize: 12,
                  }}
                >
                  ◀ Prev
                </button>
                <button
                  type="button"
                  disabled={zoomedIdx >= dict.motifs.length - 1}
                  onClick={() => setZoomedIdx(zoomedIdx + 1)}
                  style={{
                    padding: '3px 8px',
                    borderRadius: 4,
                    border: '1px solid #d1d5db',
                    background: zoomedIdx >= dict.motifs.length - 1 ? '#f3f4f6' : '#fff',
                    cursor: zoomedIdx >= dict.motifs.length - 1 ? 'default' : 'pointer',
                    fontSize: 12,
                  }}
                >
                  Next ▶
                </button>
                <button
                  type="button"
                  aria-label="Close"
                  onClick={() => setZoomedIdx(null)}
                  style={{
                    padding: '3px 9px',
                    borderRadius: 4,
                    border: '1px solid #d1d5db',
                    background: '#fff',
                    cursor: 'pointer',
                    fontSize: 13,
                    fontWeight: 600,
                  }}
                >
                  ×
                </button>
              </div>
            </div>
            <div style={{ padding: 12, background: '#fff', display: 'flex', justifyContent: 'center' }}>
              <img
                src={scanImageUrl(zoomed.zoomImageUrl)}
                alt={`Motif ${zoomed.id} zoomed`}
                style={{
                  width: '100%',
                  maxHeight: '72vh',
                  objectFit: 'contain',
                  display: 'block',
                }}
              />
            </div>
          </div>
        </div>
      )}
    </section>
  );
}
