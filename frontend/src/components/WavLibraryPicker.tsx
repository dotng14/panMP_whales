import { useCallback, useEffect, useRef, useState } from 'react';
import {
  browseLibrary,
  getLibraryRoots,
  type BrowseResponse,
  type LibraryRoot,
} from '../api/library';

const PAGE_SIZE = 50;

function fmtSize(bytes: number): string {
  if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(1)} GB`;
  if (bytes >= 1e6) return `${(bytes / 1e6).toFixed(0)} MB`;
  return `${(bytes / 1e3).toFixed(0)} KB`;
}

function fmtDate(iso: string): string {
  return new Date(iso).toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' });
}

interface Props {
  /** Fired when the highlighted WAV changes; `null` when the selection is cleared. */
  onSelectFile: (file: { fileId: string; name: string } | null) => void;
  /** The selection lives with the owner (it drives the dictionary menu below). */
  selectedFileId: string | null;
  disabled?: boolean;
  /** Bumped when a library job completes — updates the ready badge without changing directory. */
  libraryCacheRefresh?: { fileId: string; nonce: number } | null;
}

export function WavLibraryPicker({ onSelectFile, selectedFileId, disabled, libraryCacheRefresh }: Props) {
  const [roots, setRoots] = useState<LibraryRoot[]>([]);
  const [rootIdx, setRootIdx] = useState(0);
  const [path, setPath] = useState('');
  const [q, setQ] = useState('');
  const [offset, setOffset] = useState(0);
  const [browse, setBrowse] = useState<BrowseResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const lastCacheRefreshNonceRef = useRef<number | null>(null);
  const listScrollRef = useRef<HTMLDivElement>(null);
  // Lets `load` clear the owner's selection without taking it as a dependency.
  const onSelectFileRef = useRef(onSelectFile);
  useEffect(() => { onSelectFileRef.current = onSelectFile; });

  useEffect(() => {
    getLibraryRoots()
      .then(r => { setRoots(r); })
      .catch(e => setError(String(e)));
  }, []);

  const load = useCallback(
    async (
      root: number,
      p: string,
      search: string,
      off: number,
      opts?: { preserveSelection?: boolean; silent?: boolean },
    ) => {
      const scrollTop =
        opts?.silent && listScrollRef.current ? listScrollRef.current.scrollTop : null;

      if (!opts?.silent) {
        setLoading(true);
      }
      setError(null);
      try {
        const res = await browseLibrary({
          root,
          path: p,
          q: search,
          offset: off,
          limit: PAGE_SIZE,
          recursive: !!search,
        });
        setBrowse(res);
        if (!opts?.preserveSelection) {
          onSelectFileRef.current(null);
        }
        if (scrollTop !== null) {
          requestAnimationFrame(() => {
            if (listScrollRef.current) {
              listScrollRef.current.scrollTop = scrollTop;
            }
          });
        }
      } catch (e) {
        setError(String(e));
      } finally {
        if (!opts?.silent) {
          setLoading(false);
        }
      }
    },
    []
  );

  const markFileCached = useCallback((fileId: string) => {
    const scrollTop = listScrollRef.current?.scrollTop ?? null;
    setBrowse(prev =>
      prev
        ? {
            ...prev,
            entries: prev.entries.map(entry =>
              entry.type === 'file' && entry.fileId === fileId
                ? { ...entry, cached: true }
                : entry,
            ),
          }
        : prev,
    );
    if (scrollTop !== null) {
      requestAnimationFrame(() => {
        if (listScrollRef.current) {
          listScrollRef.current.scrollTop = scrollTop;
        }
      });
    }
  }, []);

  // Initial load and when root/path/offset change
  useEffect(() => {
    if (roots.length > 0) load(rootIdx, path, q, offset);
  }, [rootIdx, path, offset, roots.length]); // q excluded — handled by debounce

  // After a library job completes, show the ready badge immediately and confirm via browse.
  useEffect(() => {
    if (!libraryCacheRefresh) return;
    if (libraryCacheRefresh.nonce === lastCacheRefreshNonceRef.current) return;
    lastCacheRefreshNonceRef.current = libraryCacheRefresh.nonce;
    markFileCached(libraryCacheRefresh.fileId);
    if (roots.length > 0) {
      load(rootIdx, path, q, offset, { preserveSelection: true, silent: true });
    }
  }, [libraryCacheRefresh, roots.length, rootIdx, path, q, offset, load, markFileCached]);

  // Debounced search
  function handleSearchChange(val: string) {
    setQ(val);
    setOffset(0);
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => load(rootIdx, path, val, 0), 300);
  }

  function navigateTo(newPath: string) {
    setPath(newPath);
    setOffset(0);
    setQ('');
  }

  const totalPages = browse ? Math.ceil(browse.total / PAGE_SIZE) : 0;
  const currentPage = Math.floor(offset / PAGE_SIZE) + 1;

  return (
    <div style={{ border: '1px solid #d1d5db', borderRadius: 8, overflow: 'hidden', fontSize: 13, maxWidth: 680 }}>
      {/* Header: root switcher + search */}
      <div style={{ display: 'flex', gap: 8, padding: '8px 12px', background: '#f9fafb', borderBottom: '1px solid #d1d5db' }}>
        {roots.length > 1 && (
          <select
            value={rootIdx}
            onChange={e => { setRootIdx(Number(e.target.value)); setPath(''); setOffset(0); setQ(''); }}
            style={{ padding: '3px 6px', borderRadius: 4, border: '1px solid #d1d5db', fontSize: 13 }}
          >
            {roots.map(r => <option key={r.id} value={r.id}>{r.label}</option>)}
          </select>
        )}
        <input
          type="text"
          placeholder="Search filenames…"
          value={q}
          onChange={e => handleSearchChange(e.target.value)}
          style={{ flex: 1, padding: '3px 8px', borderRadius: 4, border: '1px solid #d1d5db', fontSize: 13 }}
        />
        {q && (
          <button
            onClick={() => { setQ(''); setOffset(0); load(rootIdx, path, '', 0); }}
            style={{ padding: '3px 8px', borderRadius: 4, border: '1px solid #d1d5db', background: '#fff', cursor: 'pointer', fontSize: 13 }}
          >
            Clear
          </button>
        )}
      </div>

      {/* Breadcrumb / parent nav */}
      {!q && path && (
        <div
          style={{ padding: '4px 12px', background: '#f3f4f6', borderBottom: '1px solid #e5e7eb', color: '#6b7280', cursor: 'pointer' }}
          onClick={() => navigateTo(browse?.parentPath ?? '')}
        >
          ↑ {path}
        </div>
      )}

      {/* Entry list */}
      <div ref={listScrollRef} style={{ maxHeight: 320, overflowY: 'auto' }}>
        {loading && (
          <div style={{ padding: '20px 12px', color: '#9ca3af', textAlign: 'center' }}>Loading…</div>
        )}
        {error && (
          <div style={{ padding: '12px', color: '#dc2626' }}>{error}</div>
        )}
        {!loading && browse && browse.entries.length === 0 && (
          <div style={{ padding: '20px 12px', color: '#9ca3af', textAlign: 'center' }}>No files found.</div>
        )}
        {!loading && browse?.entries.map((entry, i) => {
          const isSelected = entry.type === 'file' && entry.fileId === selectedFileId;
          return (
            <div
              key={i}
              onClick={() => {
                if (disabled) return;
                if (entry.type === 'dir') navigateTo(entry.path!);
                else if (entry.fileId) onSelectFile({ fileId: entry.fileId, name: entry.name });
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
              onMouseEnter={e => { if (!isSelected) (e.currentTarget as HTMLDivElement).style.background = '#f9fafb'; }}
              onMouseLeave={e => { if (!isSelected) (e.currentTarget as HTMLDivElement).style.background = 'transparent'; }}
            >
              <span style={{ fontSize: 15 }}>{entry.type === 'dir' ? '📁' : '🎵'}</span>
              <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                {entry.name}
              </span>
              {entry.type === 'file' && entry.cached && (
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
              {entry.type === 'file' && entry.sizeBytes && (
                <span style={{ color: '#9ca3af', minWidth: 60, textAlign: 'right' }}>{fmtSize(entry.sizeBytes)}</span>
              )}
              {entry.type === 'file' && entry.mtime && (
                <span style={{ color: '#9ca3af', minWidth: 70, textAlign: 'right' }}>{fmtDate(entry.mtime)}</span>
              )}
            </div>
          );
        })}
      </div>

      {/* Footer: pagination + process button */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '8px 12px', borderTop: '1px solid #d1d5db', background: '#f9fafb' }}>
        <button
          onClick={() => setOffset(o => Math.max(0, o - PAGE_SIZE))}
          disabled={offset === 0 || loading}
          style={{ padding: '3px 10px', borderRadius: 4, border: '1px solid #d1d5db', background: '#fff', cursor: offset === 0 ? 'default' : 'pointer', fontSize: 13 }}
        >
          ◀
        </button>
        <span style={{ flex: 1, textAlign: 'center', color: '#6b7280' }}>
          {browse ? `Page ${currentPage} of ${Math.max(1, totalPages)} (${browse.total.toLocaleString()} items)` : ''}
        </span>
        <button
          onClick={() => setOffset(o => o + PAGE_SIZE)}
          disabled={!browse || offset + PAGE_SIZE >= browse.total || loading}
          style={{ padding: '3px 10px', borderRadius: 4, border: '1px solid #d1d5db', background: '#fff', cursor: 'pointer', fontSize: 13 }}
        >
          ▶
        </button>
      </div>
    </div>
  );
}
