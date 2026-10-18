import { useEffect, useRef, useState, type RefObject } from 'react';

export const SCROLLBAR_H = 12;

interface Props {
  /** The native scroll container this bar mirrors/drives. */
  scrollRef: RefObject<HTMLDivElement | null>;
  /** Total scrollable content width in px; triggers a re-measure on zoom changes. */
  contentWidth: number;
}

const MIN_THUMB_PX = 24;

/**
 * Custom always-visible horizontal scrollbar drawn with plain divs. Native scrollbars
 * are unreliable here: macOS overlay scrollbars stay hidden until an active scroll/hover
 * regardless of CSS, and Safari in particular ignores ::-webkit-scrollbar styling that
 * would otherwise force a classic bar on Chrome. Drawing our own renders identically on
 * every browser/OS instead of depending on OS-level scrollbar preferences.
 */
export function HorizontalScrollbar({ scrollRef, contentWidth }: Props) {
  const trackRef = useRef<HTMLDivElement>(null);
  const [thumb, setThumb] = useState({ left: 0, width: 0 });

  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;

    const update = () => {
      const viewW = el.clientWidth;
      const scrollW = Math.max(el.scrollWidth, viewW);
      const width = Math.max(MIN_THUMB_PX, (viewW / scrollW) * viewW);
      const maxThumbLeft = viewW - width;
      const maxScroll = scrollW - viewW;
      const left = maxScroll > 0 ? (el.scrollLeft / maxScroll) * maxThumbLeft : 0;
      setThumb({ left, width });
    };

    update();
    el.addEventListener('scroll', update, { passive: true });
    const ro = new ResizeObserver(update);
    ro.observe(el);
    return () => {
      el.removeEventListener('scroll', update);
      ro.disconnect();
    };
  }, [scrollRef, contentWidth]);

  const scrollToClientX = (clientX: number) => {
    const el = scrollRef.current;
    const track = trackRef.current;
    if (!el || !track) return;
    const rect = track.getBoundingClientRect();
    const viewW = el.clientWidth;
    const scrollW = Math.max(el.scrollWidth, viewW);
    const maxScroll = scrollW - viewW;
    if (maxScroll <= 0) return;
    const width = Math.max(MIN_THUMB_PX, (viewW / scrollW) * viewW);
    const maxThumbLeft = viewW - width;
    const x = clientX - rect.left - width / 2;
    const ratio = maxThumbLeft > 0 ? Math.min(1, Math.max(0, x / maxThumbLeft)) : 0;
    el.scrollLeft = ratio * maxScroll;
  };

  const handleThumbPointerDown = (e: React.PointerEvent) => {
    e.preventDefault();
    e.stopPropagation();
    const onMove = (ev: PointerEvent) => scrollToClientX(ev.clientX);
    const onUp = () => {
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', onUp);
    };
    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', onUp);
  };

  const handleTrackClick = (e: React.MouseEvent) => {
    if (e.target !== trackRef.current) return; // let the thumb's own drag handle clicks on itself
    scrollToClientX(e.clientX);
  };

  return (
    <div
      ref={trackRef}
      onClick={handleTrackClick}
      style={{
        position: 'relative',
        height: SCROLLBAR_H,
        background: '#e5e7eb',
        borderRadius: SCROLLBAR_H / 2,
        marginTop: 4,
        cursor: 'pointer',
        flexShrink: 0,
      }}
    >
      <div
        onPointerDown={handleThumbPointerDown}
        style={{
          position: 'absolute',
          top: 0,
          left: thumb.left,
          width: thumb.width,
          height: SCROLLBAR_H,
          background: '#9ca3af',
          borderRadius: SCROLLBAR_H / 2,
          cursor: 'grab',
        }}
      />
    </div>
  );
}
