import { API_BASE } from './base';
import type { ArrayMeta, PanelProps } from './types';
import type { SpectrogramViewerProps } from '../components/SpectrogramViewer/types';

interface PanelBinHeader extends PanelProps {
  arrays: ArrayMeta[];
}

function headerPadding(headerLen: number): number {
  const end = 4 + headerLen;
  return (4 - (end % 4)) % 4;
}

function readArray(
  buf: ArrayBuffer,
  offset: number,
  meta: ArrayMeta,
): { data: Float32Array | Uint8Array; nextOffset: number } {
  if (meta.dtype === 'uint8') {
    const bytes = meta.length;
    return {
      data: new Uint8Array(buf, offset, bytes),
      nextOffset: offset + bytes,
    };
  }
  const bytes = meta.length * 4;
  if (offset % 4 !== 0) {
    return {
      data: new Float32Array(buf.slice(offset, offset + bytes)),
      nextOffset: offset + bytes,
    };
  }
  return {
    data: new Float32Array(buf, offset, meta.length),
    nextOffset: offset + bytes,
  };
}

/**
 * Fetch and decode the binary panel for a job over HTTP.
 *
 * panel.bin layout (v2):
 *   [4 bytes uint32 LE: JSON header byte length]
 *   [N bytes: JSON header, UTF-8]
 *   [0–3 bytes: zero padding — blob offset is 4-byte aligned]
 *   [uint8 magnitude][float32 energyRaw][float32 energySmoothed]
 */
export async function fetchPanel(jobId: string): Promise<SpectrogramViewerProps> {
  const res = await fetch(`${API_BASE}/api/jobs/${jobId}/panel`);
  if (!res.ok) throw new Error(`Panel fetch failed: ${res.status}`);
  const buf = await res.arrayBuffer();

  const headerLen = new DataView(buf).getUint32(0, true);
  const header = JSON.parse(
    new TextDecoder().decode(new Uint8Array(buf, 4, headerLen)),
  ) as PanelBinHeader;

  let offset = 4 + headerLen + headerPadding(headerLen);
  const arrays: Record<string, Float32Array | Uint8Array> = {};
  for (const meta of header.arrays) {
    const { data, nextOffset } = readArray(buf, offset, meta);
    if (nextOffset > buf.byteLength) {
      throw new Error(`Array "${meta.name}" exceeds panel buffer length`);
    }
    arrays[meta.name] = data;
    offset = nextOffset;
  }

  const magnitude = arrays['magnitude'];
  const magMeta = header.arrays.find(a => a.name === 'magnitude');
  const magnitudeIsUint8 = magMeta?.dtype === 'uint8';

  return {
    magnitude,
    magnitudeIsUint8,
    energyRaw: arrays['energyRaw'] as Float32Array,
    energySmoothed: arrays['energySmoothed'] as Float32Array,
    freqBins: header.freqBins,
    timeCols: header.timeCols,
    title: header.title,
    colormap: header.colormap,
    brightness: header.brightness,
    normMax: header.normMax,
    audioDurationSec: header.audioDurationSec,
    freqLo: header.freqLo,
    freqHi: header.freqHi,
  };
}
