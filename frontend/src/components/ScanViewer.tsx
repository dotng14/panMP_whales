import { useEffect, useMemo, useState } from 'react';
import { getScanSummary, scanImageUrl, type ScanSummary } from '../api/scan';

export function ScanViewer() {
  const [summary, setSummary] = useState<ScanSummary | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activeOverview, setActiveOverview] = useState<string | null>(null);
  const [selectedMonth, setSelectedMonth] = useState<string>('');

  useEffect(() => {
    let cancelled = false;
    getScanSummary()
      .then(data => {
        if (cancelled) return;
        setSummary(data);
        if (data.overviews.length > 0) setActiveOverview(data.overviews[0].id);
        if (data.months.length > 0) setSelectedMonth(data.months[0].month);
      })
      .catch(e => {
        if (!cancelled) setError(String(e));
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const years = useMemo(() => {
    if (!summary) return [];
    return Array.from(new Set(summary.months.map(m => m.month.slice(0, 4))));
  }, [summary]);

  const selectedYear = selectedMonth.slice(0, 4);
  const monthsInYear = useMemo(() => {
    if (!summary || !selectedYear) return [];
    return summary.months.filter(m => m.month.startsWith(`${selectedYear}-`));
  }, [summary, selectedYear]);

  const currentIdx = useMemo(() => {
    if (!summary) return -1;
    return summary.months.findIndex(m => m.month === selectedMonth);
  }, [summary, selectedMonth]);

  const currentMonth = currentIdx >= 0 && summary ? summary.months[currentIdx] : null;
  const currentOverview = summary?.overviews.find(o => o.id === activeOverview) ?? null;

  if (error) {
    return (
      <div style={{ padding: 16, color: '#b91c1c', fontSize: 13 }}>
        Failed to load 20-year scan results: {error}
      </div>
    );
  }
  if (!summary) {
    return <div style={{ padding: 16, color: '#6b7280', fontSize: 13 }}>Loading 20-year scan…</div>;
  }
  if (!summary.available) {
    return (
      <div style={{ padding: 16, color: '#6b7280', fontSize: 13 }}>
        No 20-year scan results found on the server.
      </div>
    );
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20, maxWidth: 1100 }}>
      {summary.months.length > 0 && (
        <section
          style={{
            border: '1px solid #d1d5db',
            borderRadius: 8,
            background: '#fff',
            overflow: 'hidden',
          }}
        >
          <div
            style={{
              display: 'flex',
              flexWrap: 'wrap',
              alignItems: 'center',
              gap: 10,
              padding: '10px 14px',
              background: '#f9fafb',
              borderBottom: '1px solid #d1d5db',
              fontSize: 13,
            }}
          >
            <strong style={{ color: '#111827' }}>Monthly hourly summaries</strong>
            <span style={{ color: '#6b7280', fontSize: 12 }}>
              ({summary.months.length} months, {summary.months[summary.months.length - 1]?.month} →{' '}
              {summary.months[0]?.month})
            </span>
            <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 6 }}>
              <button
                type="button"
                disabled={currentIdx < 0 || currentIdx >= summary.months.length - 1}
                onClick={() => setSelectedMonth(summary.months[currentIdx + 1].month)}
                style={{
                  padding: '3px 8px',
                  borderRadius: 4,
                  border: '1px solid #d1d5db',
                  background: '#fff',
                  cursor: 'pointer',
                }}
                title="Older month"
              >
                ◀ Older
              </button>
              <select
                aria-label="Year"
                value={selectedYear}
                onChange={e => {
                  const yr = e.target.value;
                  const first = summary.months.find(m => m.month.startsWith(`${yr}-`));
                  if (first) setSelectedMonth(first.month);
                }}
                style={{
                  padding: '4px 8px',
                  borderRadius: 4,
                  border: '1px solid #d1d5db',
                  background: '#fff',
                }}
              >
                {years.map(y => (
                  <option key={y} value={y}>
                    {y}
                  </option>
                ))}
              </select>
              <select
                aria-label="Month"
                value={selectedMonth}
                onChange={e => setSelectedMonth(e.target.value)}
                style={{
                  padding: '4px 8px',
                  borderRadius: 4,
                  border: '1px solid #d1d5db',
                  background: '#fff',
                }}
              >
                {monthsInYear.map(m => (
                  <option key={m.month} value={m.month}>
                    {m.month}
                  </option>
                ))}
              </select>
              <button
                type="button"
                disabled={currentIdx <= 0}
                onClick={() => setSelectedMonth(summary.months[currentIdx - 1].month)}
                style={{
                  padding: '3px 8px',
                  borderRadius: 4,
                  border: '1px solid #d1d5db',
                  background: '#fff',
                  cursor: 'pointer',
                }}
                title="Newer month"
              >
                Newer ▶
              </button>
            </div>
          </div>

          {currentMonth && (
            <div style={{ padding: 14 }}>
              {currentMonth.recordings != null && (
                <div style={{ fontSize: 12, color: '#4b5563', marginBottom: 10 }}>
                  <strong>{currentMonth.month}</strong>: {currentMonth.recordings} recordings ·{' '}
                  {currentMonth.hours?.toLocaleString()} hours ({currentMonth.minutes?.toLocaleString()} min) ·
                  mean energy ratio <code>{currentMonth.meanEnergyRatio?.toFixed(4)}</code> ·
                  mean 0.1–1 Hz FFT fraction <code>{currentMonth.meanFftFraction?.toFixed(4)}</code>
                </div>
              )}
              <img
                src={scanImageUrl(currentMonth.imageUrl)}
                alt={`${currentMonth.month} hourly means`}
                style={{ width: '100%', height: 'auto', display: 'block', borderRadius: 4 }}
              />
            </div>
          )}
        </section>
      )}

      {summary.overviews.length > 0 && (
        <section
          style={{
            border: '1px solid #d1d5db',
            borderRadius: 8,
            background: '#fff',
            overflow: 'hidden',
          }}
        >
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 8,
              padding: '10px 14px',
              background: '#f9fafb',
              borderBottom: '1px solid #d1d5db',
              fontSize: 13,
            }}
          >
            <strong style={{ color: '#111827', marginRight: 8 }}>20-year overview</strong>
            {summary.overviews.map(ov => {
              const active = ov.id === activeOverview;
              return (
                <button
                  key={ov.id}
                  type="button"
                  onClick={() => setActiveOverview(ov.id)}
                  style={{
                    padding: '4px 10px',
                    borderRadius: 6,
                    border: active ? '1px solid #1d4ed8' : '1px solid #d1d5db',
                    background: active ? '#1d4ed8' : '#fff',
                    color: active ? '#fff' : '#374151',
                    fontSize: 12,
                    fontWeight: active ? 600 : 400,
                    cursor: 'pointer',
                  }}
                >
                  {ov.label}
                </button>
              );
            })}
          </div>
          {currentOverview && (
            <div style={{ padding: 14 }}>
              <img
                src={scanImageUrl(currentOverview.imageUrl)}
                alt={currentOverview.label}
                style={{ width: '100%', height: 'auto', display: 'block', borderRadius: 4 }}
              />
            </div>
          )}
        </section>
      )}
    </div>
  );
}
