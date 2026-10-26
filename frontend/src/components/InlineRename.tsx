import { useEffect, useRef, useState, type CSSProperties, type KeyboardEvent, type MouseEvent } from 'react';

interface Props {
  value: string;
  untitled?: boolean;
  ariaLabel: string;
  disabled?: boolean;
  onSave: (next: string) => Promise<void> | void;
}

const pencilStyle: CSSProperties = {
  border: 'none',
  background: 'transparent',
  color: '#6b7280',
  cursor: 'pointer',
  padding: '0 2px',
  fontSize: 13,
  lineHeight: 1,
  flexShrink: 0,
};

const inputStyle: CSSProperties = {
  minWidth: 80,
  maxWidth: 180,
  fontSize: 12,
  padding: '1px 4px',
  border: '1px solid #93c5fd',
  borderRadius: 4,
  outline: 'none',
};

/** Pencil → inline text field. Empty save clears the user name. */
export function InlineRename({ value, untitled, ariaLabel, disabled, onSave }: Props) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(value);
  const [saving, setSaving] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (editing) {
      setDraft(untitled ? '' : value);
      inputRef.current?.focus();
      inputRef.current?.select();
    }
  }, [editing, untitled, value]);

  const stop = (event: MouseEvent) => {
    event.stopPropagation();
  };

  const commit = async () => {
    if (saving) return;
    const next = draft.trim();
    const current = untitled ? '' : value.trim();
    if (next === current) {
      setEditing(false);
      return;
    }
    setSaving(true);
    try {
      await onSave(next);
      setEditing(false);
    } finally {
      setSaving(false);
    }
  };

  const onKey = (event: KeyboardEvent<HTMLInputElement>) => {
    event.stopPropagation();
    if (event.key === 'Enter') {
      event.preventDefault();
      void commit();
    } else if (event.key === 'Escape') {
      event.preventDefault();
      setEditing(false);
    }
  };

  if (editing) {
    return (
      <input
        ref={inputRef}
        value={draft}
        disabled={saving}
        aria-label={ariaLabel}
        onClick={stop}
        onChange={event => setDraft(event.target.value)}
        onBlur={() => void commit()}
        onKeyDown={onKey}
        style={inputStyle}
      />
    );
  }

  return (
    <button
      type="button"
      aria-label={ariaLabel}
      title={ariaLabel}
      disabled={disabled}
      onClick={event => {
        stop(event);
        if (!disabled) setEditing(true);
      }}
      style={pencilStyle}
    >
      ✎
    </button>
  );
}
