import { useState, type ChangeEvent } from 'react';

interface Props {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
}

export function ZoomControl({ label, value, min, max, step, onChange }: Props) {
  const [text, setText] = useState(String(Math.round(value)));

  // Keep the text field in sync when value changes from outside (e.g. slider drag, zoom reset).
  // Compare rounded-to-rounded so this settles in one render instead of looping forever.
  if (Math.round(Number(text)) !== Math.round(value) && document.activeElement?.getAttribute('data-zoom-label') !== label) {
    setText(String(Math.round(value)));
  }

  function commitText() {
    const n = Number(text);
    if (Number.isFinite(n)) {
      const clamped = Math.min(max, Math.max(min, n));
      onChange(clamped);
      setText(String(Math.round(clamped)));
    } else {
      setText(String(Math.round(value)));
    }
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2, fontSize: 11, color: '#6b7280' }}>
      <span>{label}</span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e: ChangeEvent<HTMLInputElement>) => onChange(Number(e.target.value))}
        style={{ width: 90, accentColor: '#2563eb' }}
      />
      <input
        type="number"
        data-zoom-label={label}
        min={min}
        max={max}
        step={step}
        value={text}
        onChange={e => setText(e.target.value)}
        onBlur={commitText}
        onKeyDown={e => { if (e.key === 'Enter') e.currentTarget.blur(); }}
        style={{ width: 56, padding: '1px 4px', borderRadius: 4, border: '1px solid #d1d5db', fontSize: 11, textAlign: 'center' }}
      />
    </div>
  );
}
