import { useEffect, useMemo, useState, type CSSProperties } from 'react';
import {
  getFileAnalyses,
  patchCatalogLabel,
  type AnalysisChoice,
  type CatalogItem,
} from '../api/catalog';
import { InlineRename } from './InlineRename';

const SPECTROGRAM_ONLY_LABEL = 'Spectrogram only';
const KIND_FILTER_KEY = 'spectrogram-viewer.picker-kinds';

type SortField = 'name' | 'time' | 'index';
type Kind = CatalogItem['kind'];
/** `undefined` = nothing picked yet; `'none'` = the spectrogram-only row. */
type Selection = string | undefined;

interface Props {
  fileId: string;
  fileName: string;
  onChoose: (choice: AnalysisChoice) => void;
  onBack: () => void;
  initialChoice?: AnalysisChoice;
  disabled?: boolean;
}

const keyOf = (item: { kind: Kind; id: number }) => `${item.kind}:${item.id}`;

function readKindFilter(): Record<Kind, boolean> {
  try {
    const parsed = JSON.parse(window.localStorage.getItem(KIND_FILTER_KEY) ?? '{}') as Partial<Record<Kind, unknown>>;
    return { dictionary: parsed.dictionary !== false, csv: parsed.csv !== false };
  } catch {
    return { dictionary: true, csv: true };
  }
}

function fmtTimestamp(iso: string | null): string {
  if (!iso) return '';
  return new Date(iso).toLocaleString(undefined, {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  });
}

const rowStyle = (selected: boolean): CSSProperties => ({
  display: 'flex',
  alignItems: 'center',
  gap: 8,
  padding: '6px 12px',
  cursor: 'pointer',
  background: selected ? '#dbeafe' : 'transparent',
  borderBottom: '1px solid #f3f4f6',
});

const BADGE_COLORS: Record<string, { fg: string; bg: string; border: string }> = {
  seed: { fg: '#1d4ed8', bg: '#dbeafe', border: '#93c5fd' },
  refinement: { fg: '#7c3aed', bg: '#ede9fe', border: '#c4b5fd' },
  matching: { fg: '#7c3aed', bg: '#ede9fe', border: '#c4b5fd' },
  edit: { fg: '#7c3aed', bg: '#ede9fe', border: '#c4b5fd' },
  csv: { fg: '#047857', bg: '#d1fae5', border: '#6ee7b7' },
};

const BADGE_TITLES: Record<string, string> = {
  seed: 'Randomized build',
  refinement: 'Built from a parent dictionary',
  matching: 'Matching pursuit with a parent dictionary',
  edit: 'Edited from a parent dictionary',
  csv: 'Song-structure CSV',
};

function sortButtonStyle(active: boolean): CSSProperties {
  return {
    padding: '2px 8px',
    fontSize: 11,
    borderRadius: 4,
    border: active ? '1px solid #93c5fd' : '1px solid #d1d5db',
    background: active ? '#dbeafe' : '#fff',
    color: active ? '#1d4ed8' : '#374151',
    fontWeight: active ? 600 : 400,
    cursor: 'pointer',
  };
}

/** Step 2 of the library flow: pick a dictionary or song-structure CSV to overlay. */
export function AnalysisPicker({ fileId, fileName, onChoose, onBack, initialChoice, disabled }: Props) {
  const [items, setItems] = useState<CatalogItem[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [sortField, setSortField] = useState<SortField>('time');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');
  const [showKind, setShowKind] = useState<Record<Kind, boolean>>(readKindFilter);
  const [selected, setSelected] = useState<Selection>(undefined);

  // The owner remounts this per file (keyed on fileId), so one fetch per mount.
  useEffect(() => {
    let cancelled = false;
    getFileAnalyses(fileId)
      .then(res => {
        if (cancelled) return;
        setItems(res.items);
        const restored = initialChoice === null
          ? 'none'
          : initialChoice && res.items.some(i => keyOf(i) === keyOf(initialChoice))
            ? keyOf(initialChoice)
            : undefined;
        setSelected(restored ?? (res.items.length > 0 ? keyOf(res.items[0]) : 'none'));
      })
      .catch(e => { if (!cancelled) setError(String(e)); });
    return () => { cancelled = true; };
  }, [fileId, initialChoice]);

  useEffect(() => {
    try {
      window.localStorage.setItem(KIND_FILTER_KEY, JSON.stringify(showKind));
    } catch {
      // Best-effort persistence only.
    }
  }, [showKind]);

  const handleSortToggle = (field: SortField) => {
    if (sortField === field) {
      setSortOrder(current => (current === 'asc' ? 'desc' : 'asc'));
    } else {
      setSortField(field);
      setSortOrder(field === 'time' ? 'desc' : 'asc');
    }
  };

  const counts = useMemo(() => ({
    dictionary: items?.filter(i => i.kind === 'dictionary').length ?? 0,
    csv: items?.filter(i => i.kind === 'csv').length ?? 0,
  }), [items]);

  const visibleItems = useMemo(() => {
    if (!items) return null;
    const copy = items.filter(i => showKind[i.kind]);
    const sign = sortOrder === 'asc' ? 1 : -1;
    copy.sort((a, b) => {
      if (sortField === 'name') {
        const aNamed = a.named ? 0 : 1;
        const bNamed = b.named ? 0 : 1;
        if (aNamed !== bNamed) return aNamed - bNamed;
        return sign * a.displayName.toLocaleLowerCase().localeCompare(b.displayName.toLocaleLowerCase());
      }
      if (sortField === 'time') {
        const aTime = a.createdAt ? new Date(a.createdAt).getTime() : 0;
        const bTime = b.createdAt ? new Date(b.createdAt).getTime() : 0;
        return sign * (aTime - bTime);
      }
      // Ids are per kind, so index order groups dictionaries before CSVs.
      if (a.kind !== b.kind) return a.kind === 'dictionary' ? -1 : 1;
      return sign * (a.id - b.id);
    });
    return copy;
  }, [items, showKind, sortField, sortOrder]);

  const chosen = items?.find(i => keyOf(i) === selected) ?? null;
  const canConfirm = selected !== undefined && !disabled;

  const confirm = () => {
    if (!canConfirm) return;
    onChoose(chosen ? { kind: chosen.kind, id: chosen.id, label: chosen.displayName } : null);
  };

  const saveLabel = async (item: CatalogItem, next: string) => {
    const updated = await patchCatalogLabel(item.kind, item.id, next === '' ? null : next);
    setItems(current =>
      current ? current.map(i => (keyOf(i) === keyOf(item) ? { ...i, ...updated } as CatalogItem : i)) : current,
    );
  };

  const kindToggle = (kind: Kind, text: string) => (
    <label
      title={`Show or hide ${text.toLowerCase()} in this list`}
      style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 11, color: '#374151', cursor: 'pointer' }}
    >
      <input
        type="checkbox"
        checked={showKind[kind]}
        onChange={() => setShowKind(v => ({ ...v, [kind]: !v[kind] }))}
        style={{ margin: 0, accentColor: '#2563eb', cursor: 'pointer' }}
      />
      {text} ({counts[kind]})
    </label>
  );

  return (
    <div style={{ border: '1px solid #d1d5db', borderRadius: 8, overflow: 'hidden', fontSize: 13, maxWidth: 760, marginTop: 10 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '8px 12px', background: '#f9fafb', borderBottom: '1px solid #d1d5db', flexWrap: 'wrap' }}>
        <span style={{ fontWeight: 600, color: '#374151' }}>Overlay</span>
        <span style={{ color: '#6b7280', maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}> — 🎵 {fileName}</span>
        <span style={{ flex: 1 }} />
        {kindToggle('dictionary', 'Dictionaries')}
        {kindToggle('csv', 'CSVs')}
        <span style={{ fontSize: 11, color: '#6b7280', marginLeft: 6 }}>Sort:</span>
        <button type="button" title="Sort alphabetically, named first (click to reverse)" onClick={() => handleSortToggle('name')} style={sortButtonStyle(sortField === 'name')}>
          {sortField === 'name' ? (sortOrder === 'asc' ? 'A–Z ↓' : 'Z–A ↑') : 'A–Z'}
        </button>
        <button type="button" title="Sort by catalog index (click to reverse)" onClick={() => handleSortToggle('index')} style={sortButtonStyle(sortField === 'index')}>
          {sortField === 'index' ? (sortOrder === 'asc' ? 'Index #↑' : 'Index #↓') : 'Index'}
        </button>
        <button type="button" title="Sort by timestamp (click to reverse)" onClick={() => handleSortToggle('time')} style={sortButtonStyle(sortField === 'time')}>
          {sortField === 'time' ? (sortOrder === 'desc' ? 'Time (Newest) ↓' : 'Time (Oldest) ↑') : 'Timestamps'}
        </button>
      </div>

      <div style={{ maxHeight: 260, overflowY: 'auto' }}>
        {items === null && !error && (
          <div style={{ padding: '20px 12px', color: '#9ca3af', textAlign: 'center' }}>Loading…</div>
        )}
        {error && <div style={{ padding: '12px', color: '#dc2626' }}>{error}</div>}

        {visibleItems?.map(item => {
          const key = keyOf(item);
          const badge = item.kind === 'dictionary' ? item.source : 'csv';
          const colors = BADGE_COLORS[badge];
          const parentId = item.kind === 'dictionary' ? item.parentId : null;
          return (
            <div key={key} onClick={() => setSelected(key)} onDoubleClick={confirm} style={rowStyle(selected === key)}>
              <span style={{ fontSize: 15 }}>{item.kind === 'dictionary' ? '📚' : '📄'}</span>
              <span
                title={parentId != null ? `${item.kind} ${item.id}, parent dictionary ${parentId}` : `${item.kind} ${item.id}`}
                style={{ fontFamily: 'monospace', fontSize: 12, color: '#6b7280', minWidth: parentId != null ? 72 : 36, flexShrink: 0 }}
              >
                #{item.id}
                {parentId != null && <span style={{ color: '#9ca3af' }}> ← #{parentId}</span>}
              </span>
              <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', display: 'flex', alignItems: 'center', gap: 4, minWidth: 0 }}>
                <span title={item.name} style={{ overflow: 'hidden', textOverflow: 'ellipsis', color: item.named ? '#111827' : '#6b7280' }}>
                  {item.displayName}
                </span>
                <InlineRename
                  value={item.named ? (item.label ?? '') : ''}
                  untitled={!item.named}
                  ariaLabel={item.kind === 'dictionary' ? 'Rename dictionary' : 'Rename CSV'}
                  disabled={disabled}
                  onSave={next => saveLabel(item, next)}
                />
                {item.kind === 'dictionary' && item.description && (
                  <span style={{ color: '#9ca3af' }}> — {item.description}</span>
                )}
              </span>
              <span
                title={BADGE_TITLES[badge]}
                style={{ fontSize: 11, fontWeight: 600, color: colors.fg, background: colors.bg, border: `1px solid ${colors.border}`, borderRadius: 4, padding: '1px 6px' }}
              >
                {badge}
              </span>
              <span style={{ color: '#9ca3af', minWidth: 150, textAlign: 'right' }}>
                {item.kind === 'dictionary'
                  ? `${item.nTemplates} atoms · ${item.nMatchings.toLocaleString()} placements`
                  : `${item.nRows.toLocaleString()} units · ${item.nSongs} songs`}
              </span>
              <span style={{ color: '#9ca3af', minWidth: 180, textAlign: 'right', flexShrink: 0 }}>
                {fmtTimestamp(item.createdAt)}
              </span>
            </div>
          );
        })}

        {items !== null && (
          <div onClick={() => setSelected('none')} onDoubleClick={confirm} style={rowStyle(selected === 'none')}>
            <span style={{ fontSize: 15 }}>🖼️</span>
            <span style={{ flex: 1 }}>
              {SPECTROGRAM_ONLY_LABEL}
              <span style={{ color: '#9ca3af' }}>
                {items.length === 0
                  ? ' — nothing in the catalog for this file yet'
                  : visibleItems && visibleItems.length < items.length
                    ? ` — ${items.length - visibleItems.length} hidden by the filters`
                    : ' — no overlay'}
              </span>
            </span>
          </div>
        )}
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '8px 12px', borderTop: '1px solid #d1d5db', background: '#f9fafb' }}>
        <button
          onClick={onBack}
          style={{ padding: '3px 10px', borderRadius: 4, border: '1px solid #d1d5db', background: '#fff', cursor: 'pointer', fontSize: 13 }}
        >
          ← Back
        </button>
        <span style={{ flex: 1 }} />
        <button
          onClick={confirm}
          disabled={!canConfirm}
          style={{
            padding: '5px 14px',
            borderRadius: 4,
            border: 'none',
            background: canConfirm ? '#2563eb' : '#d1d5db',
            color: '#fff',
            cursor: canConfirm ? 'pointer' : 'default',
            fontWeight: 600,
            fontSize: 13,
          }}
        >
          Analyze
        </button>
      </div>
    </div>
  );
}
