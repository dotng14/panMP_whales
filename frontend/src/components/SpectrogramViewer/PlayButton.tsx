import type { ButtonHTMLAttributes, ReactNode } from 'react';

interface Props {
  playing: boolean;
  onClick?: () => void;
}

interface SpectrogramButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  children: ReactNode;
}

export function SpectrogramButton({
  children,
  className = '',
  style,
  ...props
}: SpectrogramButtonProps) {
  return (
    <button
      type="button"
      className={`shrink-0 px-5 py-3 rounded-lg font-semibold text-black disabled:opacity-40 disabled:cursor-not-allowed ${className}`}
      style={{ background: '#f59e0b', minWidth: 80, ...style }}
      {...props}
    >
      {children}
    </button>
  );
}

export function PlayButton({ playing, onClick }: Props) {
  return (
    <SpectrogramButton onClick={onClick}>
      {playing ? 'Pause' : 'Play'}
    </SpectrogramButton>
  );
}
