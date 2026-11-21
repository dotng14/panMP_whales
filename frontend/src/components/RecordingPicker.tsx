import { useCallback, useEffect, useRef, useState, type CSSProperties } from 'react';
import {
  getFacets,
  matchRecordings,
  type FacetValue,
  type MatchEntry,
  type MatchResponse,
  type RecordingFilters,
} from '../api/library';
import {
  SOCAL_CHIP_KEYS,
  SOCAL_CHIP_LABELS,
  filledComponents,
  parseWavName,
  type ComponentKey,
} from '../lib/socalWavName';

const PAGE_SIZE = 50;

function fmtSize(bytes: number): string {
  if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(1)} GB`;
  if (bytes >= 1e6) return `${(bytes / 1e6).toFixed(0)} MB`;
  return `${(bytes / 1e3).toFixed(0)} KB`;
}

function fmtDate(iso: string): string {
  return new Date(iso).toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' });
}

type ChipState = Partial<Record<ComponentKey, string>>;

interface Props {
  onSelectFile: (file: { fileId: string; name: string } | null) => void;
  selectedFileId: string | null;
  disabled?: boolean;
  libraryCacheRefresh?: { fileId: string; nonce: number } | null;
}

function chipsToFilters(chips: ChipState, pasteFallback: string): RecordingFilters {
  const filters: RecordingFilters = { ...chips };
  if (chips.family) filters.family = chips.family;
  else if (Object.keys(chips).length > 0) filters.family = 'SOCAL';
  // When nothing parsed, use the paste box as substring search.
  if (Object.keys(chips).length === 0 && pasteFallback.trim()) {
    filters.q = pasteFallback.trim();
  }
  return filters;
}

export function RecordingPicker({ onSelectFile, selectedFileId, disabled, libraryCacheRefresh }: Props) {
  const [paste, setPaste] = useState('');
  const [chips, setChips] = useState<ChipState>({});
  const [openKey, setOpenKey] = useState<ComponentKey | null>(null);
  const [facets, setFacets] = useState<Record<string, FacetValue[]>>({});
  const [match, setMatch] = useState<MatchResponse | null>(null);
  const [offset, setOffset] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const lastCacheRefreshNonceRef = useRef<number | null>(null);
  const listScrollRef = useRef<HTMLDivElement>(null);
  const onSelectFileRef = useRef(onSelectFile);
  useEffect(() => { onSelectFileRef.current = onSelectFile; });

  const load = useCallback(
    async (
      nextChips: ChipState,
      pasteText: string,
      off: number,
      opts?: { preserveSelection?: boolean; silent?: boolean },
    ) => {
      const scrollTop =
        opts?.silent && listScrollRef.current ? listScrollRef.current.scrollTop : null;
      if (!opts?.silent) setLoading(true);
      setError(null);
      const filters = chipsToFilters(nextChips, pasteText);
      try {
        const [facetRes, matchRes] = await Promise.all([
          getFacets(filters),
          matchRecordings({ ...filters, offset: off, limit: PAGE_SIZE }),
        ]);
        setFacets(facetRes.facets);
        setMatch(matchRes);
        if (!opts?.preserveSelection) {
          onSelectFileRef.current(null);
        }
        if (scrollTop !== null) {
          requestAnimationFrame(() => {
            if (listScrollRef.current) listScrollRef.current.scrollTop = scrollTop;
          });
        }
      } catch (e) {
        setError(String(e));
      } finally {
        if (!opts?.silent) setLoading(false);
      }
    },
    [],
  );

  // Initial load (all recordings under library roots).
  useEffect(() => {
    load({}, '', 0);
  }, [load]);

  useEffect(() => {
    if (!libraryCacheRefresh) return;
    if (libraryCacheRefresh.nonce === lastCacheRefreshNonceRef.current) return;
    lastCacheRefreshNonceRef.current = libraryCacheRefresh.nonce;
    setMatch(prev =>
      prev
        ? {
            ...prev,
            entries: prev.entries.map(entry =>
              entry.fileId === libraryCacheRefresh.fileId ? { ...entry, cached: true } : entry,
            ),
          }
        : prev,
    );
    load(chips, paste, offset, { preserveSelection: true, silent: true });
  }, [libraryCacheRefresh]); // eslint-disable-line react-hooks/exhaustive-deps

  function applyPaste(raw: string) {
    setPaste(raw);
    setOffset(0);
    setOpenKey(null);
    const parsed = parseWavName(raw);
    const next = parsed ? (filledComponents(parsed) as ChipState) : {};
    // Keep family for SOCAL when parsed.
    if (parsed?.family) next.family = parsed.family;
    setChips(next);
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => load(next, raw, 0), 250);
  }

  function setChip(key: ComponentKey, value: string | null) {
    setOpenKey(null);
    setOffset(0);
    const next: ChipState = { ...chips };
    if (value == null || value === '') delete next[key];
    else next[key] = value;
    if (Object.keys(next).some(k => k !== 'family') && !next.family) {
      next.family = 'SOCAL';
    }
    if (Object.keys(next).every(k => k === 'family')) {
      delete next.family;
    }
    setChips(next);
    load(next, paste, 0);
  }

  function clearAll() {
    setPaste('');
    setChips({});
    setOffset(0);
    setOpenKey(null);
    load({}, '', 0);
  }

  const totalPages = match ? Math.ceil(match.total / PAGE_SIZE) : 0;
  const currentPage = Math.floor(offset / PAGE_SIZE) + 1;

  return (
    <div
      style={{
        border: '1px solid #d1d5db',
        borderRadius: 8,
        overflow: 'hidden',
        fontSize: 13,
        maxWidth: 720,
        background: '#fff',
      }}
    >
      <div style={{ padding: '10px 12px', background: '#f9fafb', borderBottom: '1px solid #d1d5db' }}>
        <label style={{ display: 'block', fontSize: 12, color: '#6b7280', marginBottom: 4 }}>
          Paste recording name
        </label>
        <div style={{ display: 'flex', gap: 8 }}>
          <input
            type="text"
            value={paste}
            disabled={disabled}
            placeholder="SOCAL_H_76_230426_193500_df100.x.x.wav"
            onChange={e => applyPaste(e.target.value)}
            style={{
              flex: 1,
              padding: '6px 10px',
              borderRadius: 6,
              border: '1px solid #d1d5db',
              fontSize: 13,
              fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Consolas, monospace',
            }}
          />
          {(paste || Object.keys(chips).length > 0) && (
            <button
              type="button"
              onClick={clearAll}
              disabled={disabled}
              style={{
                padding: '6px 10px',
                borderRadius: 6,
                border: '1px solid #d1d5db',
                background: '#fff',
                cursor: 'pointer',
                fontSize: 13,
              }}
            >
              Clear
            </button>
          )}
        </div>
        <div style={{ marginTop: 6, fontSize: 12, color: '#6b7280' }}>
          Parsed fields fill the chips below; open empty chips to narrow the list.
        </div>
      </div>

      <div style={{ padding: '10px 12px', borderBottom: '1px solid #e5e7eb' }}>
        <div style={{ fontSize: 12, color: '#6b7280', marginBottom: 8 }}>Components</div>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
          {SOCAL_CHIP_KEYS.map(key => {
            const value = chips[key];
            const empty = value == null || value === '';
            const options = facets[key] ?? [];
            const open = openKey === key;
            return (
              <div key={key} style={{ position: 'relative' }}>
                <button
                  type="button"
                  disabled={disabled}
                  aria-expanded={open}
                  onClick={() => setOpenKey(open ? null : key)}
                  style={{
                    minWidth: 72,
                    minHeight: 44,
                    textAlign: 'left',
                    padding: '6px 10px',
                    borderRadius: 8,
                    border: open ? '1px solid #2563eb' : '1px solid #d1d5db',
                    borderStyle: empty ? 'dashed' : 'solid',
                    background: empty ? '#f3f4f6' : '#fff',
                    cursor: 'pointer',
                    color: 'inherit',
                    font: 'inherit',
                  }}
                >
                  <span style={{ display: 'block', fontSize: 11, color: '#6b7280' }}>
                    {SOCAL_CHIP_LABELS[key]}
                  </span>
                  <span
                    style={{
                      display: 'block',
                      marginTop: 2,
                      fontWeight: empty ? 400 : 500,
                      color: empty ? '#9ca3af' : '#111827',
                      fontVariantNumeric: 'tabular-nums',
                    }}
                  >
                    {empty ? 'Any' : value}
                  </span>
                </button>
                {open && (
                  <div
                    role="listbox"
                    aria-label={`${SOCAL_CHIP_LABELS[key]} options`}
                    style={{
                      position: 'absolute',
                      left: 0,
                      top: 'calc(100% + 4px)',
                      zIndex: 20,
                      minWidth: 140,
                      maxHeight: 200,
                      overflowY: 'auto',
                      background: '#fff',
                      border: '1px solid #d1d5db',
                      borderRadius: 8,
                      padding: 4,
                      boxShadow: '0 8px 24px rgba(0,0,0,0.12)',
                    }}
                  >
                    {!empty && (
                      <button
                        type="button"
                        role="option"
                        onClick={() => setChip(key, null)}
                        style={optStyle}
                      >
                        Clear
                      </button>
                    )}
                    {options.length === 0 && (
                      <div style={{ padding: '8px 10px', color: '#9ca3af', fontSize: 12 }}>
                        No options
                      </div>
                    )}
                    {options.map(opt => (
                      <button
                        key={opt.value}
                        type="button"
                        role="option"
                        aria-selected={opt.value === value}
                        onClick={() => setChip(key, opt.value)}
                        style={{
                          ...optStyle,
                          background: opt.value === value ? '#dbeafe' : 'transparent',
                        }}
                      >
                        <span>{opt.value}</span>
                        <span style={{ color: '#9ca3af' }}>{opt.count}</span>
                      </button>
                    ))}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>

      <div style={{ padding: '8px 12px', fontSize: 12, color: '#6b7280', borderBottom: '1px solid #f3f4f6' }}>
        {match ? `${match.total.toLocaleString()} matching recording${match.total === 1 ? '' : 's'}` : '…'}
      </div>

      <div ref={listScrollRef} style={{ maxHeight: 320, overflowY: 'auto' }}>
        {loading && (
          <div style={{ padding: '20px 12px', color: '#9ca3af', textAlign: 'center' }}>Loading…</div>
        )}
        {error && <div style={{ padding: 12, color: '#dc2626' }}>{error}</div>}
        {!loading && match && match.entries.length === 0 && (
          <div style={{ padding: '20px 12px', color: '#9ca3af', textAlign: 'center' }}>
            No recordings match these components.
          </div>
        )}
        {!loading &&
          match?.entries.map((entry: MatchEntry) => {
            const isSelected = entry.fileId === selectedFileId;
            return (
              <div
                key={entry.fileId}
                onClick={() => {
                  if (disabled) return;
                  onSelectFile({ fileId: entry.fileId, name: entry.name });
                  setOpenKey(null);
                }}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  padding: '6px 12px',
                  gap: 8,
                  cursor: 'pointer',
                  background: isSelected ? '#dbeafe' : 'transparent',
                  borderBottom: '1px solid #f3f4f6',
                }}
                onMouseEnter={e => {
                  if (!isSelected) (e.currentTarget as HTMLDivElement).style.background = '#f9fafb';
                }}
                onMouseLeave={e => {
                  if (!isSelected) (e.currentTarget as HTMLDivElement).style.background = 'transparent';
                }}
              >
                <span
                  style={{
                    flex: 1,
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    whiteSpace: 'nowrap',
                    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Consolas, monospace',
                    fontSize: 12,
                  }}
                >
                  {entry.name}
                </span>
                {entry.cached && (
                  <span
                    title="Pre-processed — loads instantly"
                    style={{
                      fontSize: 11,
                      fontWeight: 600,
                      color: '#15803d',
                      background: '#dcfce7',
                      border: '1px solid #86efac',
                      borderRadius: 4,
                      padding: '1px 6px',
                    }}
                  >
                    ready
                  </span>
                )}
                {entry.sizeBytes != null && (
                  <span style={{ color: '#9ca3af', minWidth: 60, textAlign: 'right' }}>
                    {fmtSize(entry.sizeBytes)}
                  </span>
                )}
                {entry.mtime && (
                  <span style={{ color: '#9ca3af', minWidth: 70, textAlign: 'right' }}>
                    {fmtDate(entry.mtime)}
                  </span>
                )}
              </div>
            );
          })}
      </div>

      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          padding: '8px 12px',
          borderTop: '1px solid #d1d5db',
          background: '#f9fafb',
        }}
      >
        <button
          type="button"
          onClick={() => {
            const next = Math.max(0, offset - PAGE_SIZE);
            setOffset(next);
            load(chips, paste, next, { preserveSelection: true });
          }}
          disabled={offset === 0 || loading}
          style={{
            padding: '3px 10px',
            borderRadius: 4,
            border: '1px solid #d1d5db',
            background: '#fff',
            cursor: offset === 0 ? 'default' : 'pointer',
            fontSize: 13,
          }}
        >
          ◀
        </button>
        <span style={{ flex: 1, textAlign: 'center', color: '#6b7280' }}>
          {match
            ? `Page ${currentPage} of ${Math.max(1, totalPages)} (${match.total.toLocaleString()} items)`
            : ''}
        </span>
        <button
          type="button"
          onClick={() => {
            const next = offset + PAGE_SIZE;
            setOffset(next);
            load(chips, paste, next, { preserveSelection: true });
          }}
          disabled={!match || offset + PAGE_SIZE >= match.total || loading}
          style={{
            padding: '3px 10px',
            borderRadius: 4,
            border: '1px solid #d1d5db',
            background: '#fff',
            cursor: 'pointer',
            fontSize: 13,
          }}
        >
          ▶
        </button>
      </div>
    </div>
  );
}

const optStyle: CSSProperties = {
  display: 'flex',
  justifyContent: 'space-between',
  gap: 12,
  width: '100%',
  border: 0,
  background: 'transparent',
  color: 'inherit',
  font: 'inherit',
  fontSize: 13,
  padding: '8px 10px',
  borderRadius: 6,
  cursor: 'pointer',
  textAlign: 'left',
};
