import { useRef } from 'react';

interface Props {
  onFile: (file: File) => void;
  disabled?: boolean;
}

export function WavFileUpload({ onFile, disabled }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);

  function handleChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (file) onFile(file);
    e.target.value = '';
  }

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
      <input
        ref={inputRef}
        type="file"
        accept=".wav"
        style={{ display: 'none' }}
        onChange={handleChange}
        disabled={disabled}
      />
      <button
        onClick={() => inputRef.current?.click()}
        disabled={disabled}
        style={{
          padding: '8px 20px',
          background: disabled ? '#ccc' : '#2563eb',
          color: '#fff',
          border: 'none',
          borderRadius: 6,
          cursor: disabled ? 'default' : 'pointer',
          fontSize: 14,
          fontWeight: 600,
        }}
      >
        Upload WAV
      </button>
    </div>
  );
}
