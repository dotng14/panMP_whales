import { useEffect, useMemo, useState, type CSSProperties } from 'react';
import {
  getFileAnalyses,
  patchCatalogLabel,
  describeChoice,
  type AnalysisChoice,
  type CatalogItem,
  type OverlayRef,
} from '../api/catalog';
import { InlineRename } from './InlineRename';

const SPECTROGRAM_ONLY_LABEL = 'Spectrogram only';
const NO_SPECTROGRAM_LABEL = 'No spectrogram yet';
const KIND_FILTER_KEY = 'spectrogram-viewer.picker-kinds';

type SortField = 'name' | 'time' | 'index';
type Kind = CatalogItem['kind'];
/** At most one dictionary and one CSV by id; both null = the spectrogram-only row. */
type Selection = Record<Kind, number | null>;

interface Props {
  fileId: string;
  fileName: string;
  onChoose: (choice: AnalysisChoice) => void;
  onBack: () => void;
  initialChoice?: AnalysisChoice;
  disabled?: boolean;
}

const keyOf = (item: { kind: Kind; id: number }) => `${item.kind}:${item.id}`;

const createdMs = (item: CatalogItem) => (item.createdAt ? new Date(item.createdAt).getTime() : 0);

/** The newest item of a kind (the default pick), or null when there is none. */
function newestId(items: CatalogItem[], kind: Kind): number | null {
  let best: CatalogItem | null = null;
  for (const item of items) {
    if (item.kind === kind && (best === null || createdMs(item) > createdMs(best))) best = item;
  }
  return best?.id ?? null;
}

/** The restored id when it is still in the catalog, else the newest of that kind. */
function restoredId(items: CatalogItem[], kind: Kind, ref: OverlayRef | null | undefined): number | null {
  if (ref === null) return null;
  if (ref && items.some(i => i.kind === kind && i.id === ref.id)) return ref.id;
  return newestId(items, kind);
}

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

/** Step 2 of the library flow: pick a dictionary and/or a song-structure CSV to overlay. */
export function AnalysisPicker({ fileId, fileName, onChoose, onBack, initialChoice, disabled }: Props) {
  const [items, setItems] = useState<CatalogItem[] | null>(null);
  const [gplCached, setGplCached] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sortField, setSortField] = useState<SortField>('time');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');
  const [showKind, setShowKind] = useState<Record<Kind, boolean>>(readKindFilter);
  const [selected, setSelected] = useState<Selection | undefined>(undefined);

  // The owner remounts this per file (keyed on fileId), so one fetch per mount.
  // The choice to restore is the one at mount: confirming a pick updates
  // initialChoice, and refetching then would only repeat the same list.
  const [restoreChoice] = useState(initialChoice);
  useEffect(() => {
    let cancelled = false;
    getFileAnalyses(fileId)
      .then(res => {
        if (cancelled) return;
        setItems(res.items);
        setGplCached(res.gplCached);
        // Default: the newest dictionary and the newest CSV, drawn together.
        setSelected({
          dictionary: restoredId(res.items, 'dictionary', restoreChoice?.dictionary),
          csv: restoredId(res.items, 'csv', restoreChoice?.csv),
        });
      })
      .catch(e => { if (!cancelled) setError(String(e)); });
    return () => { cancelled = true; };
  }, [fileId, restoreChoice]);

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
      if (sortField === 'time') return sign * (createdMs(a) - createdMs(b));
      // Ids are per kind, so index order groups dictionaries before CSVs.
      if (a.kind !== b.kind) return a.kind === 'dictionary' ? -1 : 1;
      return sign * (a.id - b.id);
    });
    return copy;
  }, [items, showKind, sortField, sortOrder]);

  const isSelected = (item: CatalogItem) => selected?.[item.kind] === item.id;
  const noneSelected = selected !== undefined && selected.dictionary === null && selected.csv === null;
  // Clicking an item picks it for its kind (replacing the other one of that
  // kind) or, when already picked, drops it.
  const toggled = (sel: Selection, item: CatalogItem): Selection =>
    ({ ...sel, [item.kind]: sel[item.kind] === item.id ? null : item.id });
  const NONE: Selection = { dictionary: null, csv: null };

  // A kind hidden by its filter checkbox is not drawn either.
  const choiceFor = (sel: Selection): AnalysisChoice => {
    const ref = (kind: Kind): OverlayRef | null => {
      const item = showKind[kind] ? items?.find(i => i.kind === kind && i.id === sel[kind]) : undefined;
      return item ? { id: item.id, label: item.displayName } : null;
    };
    return { dictionary: ref('dictionary'), csv: ref('csv') };
  };
  const canConfirm = selected !== undefined && !disabled;

  const confirm = (sel: Selection | undefined = selected) => {
    if (!sel || disabled) return;
    onChoose(choiceFor(sel));
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

        {items !== null && (
          <div onClick={() => setSelected(NONE)} onDoubleClick={() => confirm(NONE)} style={rowStyle(noneSelected)}>
            <span style={{ fontSize: 15 }}>🖼️</span>
            <span style={{ flex: 1 }}>
              {gplCached ? SPECTROGRAM_ONLY_LABEL : NO_SPECTROGRAM_LABEL}
              <span style={{ color: '#9ca3af' }}>
                {items.length === 0
                  ? gplCached ? ' — no dictionaries or CSVs for this file yet' : ' — nothing in the catalog for this file yet'
                  : visibleItems && visibleItems.length < items.length
                    ? ` — ${items.length - visibleItems.length} hidden by the filters`
                    : ' — no overlay'}
              </span>
            </span>
          </div>
        )}

        {visibleItems?.map(item => {
          const key = keyOf(item);
          const badge = item.kind === 'dictionary' ? item.source : 'csv';
          const colors = BADGE_COLORS[badge];
          const parentId = item.kind === 'dictionary' ? item.parentId : null;
          return (
            <div
              key={key}
              onClick={() => setSelected(sel => toggled(sel ?? NONE, item))}
              // The two clicks before a double-click toggle the row off and on again.
              onDoubleClick={() => confirm({ ...(selected ?? NONE), [item.kind]: item.id })}
              style={rowStyle(isSelected(item))}
            >
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

      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '8px 12px', borderTop: '1px solid #d1d5db', background: '#f9fafb' }}>
        <button
          onClick={onBack}
          style={{ padding: '3px 10px', borderRadius: 4, border: '1px solid #d1d5db', background: '#fff', cursor: 'pointer', fontSize: 13 }}
        >
          ← Back
        </button>
        <span style={{ flex: 1, color: '#6b7280', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
          {selected && `Overlay: ${describeChoice(choiceFor(selected))}`}
        </span>
        <button
          onClick={() => confirm()}
          disabled={!canConfirm}
          title={gplCached ? 'Open the saved GPL spectrogram' : 'Run GPL on this file to make its spectrogram (saved for next time)'}
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
          {gplCached ? 'View' : 'Run GPL'}
        </button>
      </div>
    </div>
  );
}
