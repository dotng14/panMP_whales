import { useState, type ChangeEvent } from 'react';

interface Props {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
}

function formatZoom(value: number, step: number): string {
  if (step >= 1) return String(Math.round(value));
  // Enough decimals to express the step (e.g. 0.01 → 2 places).
  const decimals = Math.max(0, Math.ceil(-Math.log10(step)));
  return value.toFixed(decimals);
}

function displayKey(value: number, step: number): string {
  return formatZoom(value, step);
}

export function ZoomControl({ label, value, min, max, step, onChange }: Props) {
  const [text, setText] = useState(formatZoom(value, step));

  // Keep the text field in sync when value changes from outside (e.g. slider drag, zoom reset).
  if (
    displayKey(Number(text), step) !== displayKey(value, step) &&
    document.activeElement?.getAttribute('data-zoom-label') !== label
  ) {
    setText(formatZoom(value, step));
  }

  function commitText() {
    const n = Number(text);
    if (Number.isFinite(n)) {
      const clamped = Math.min(max, Math.max(min, n));
      onChange(clamped);
      setText(formatZoom(clamped, step));
    } else {
      setText(formatZoom(value, step));
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
