import type { JobCreated } from './types';

export async function createJob(file: File): Promise<JobCreated> {
  const form = new FormData();
  form.append('file', file);
  const res = await fetch('/api/jobs', { method: 'POST', body: form });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Upload failed (${res.status}): ${text}`);
  }
  return res.json() as Promise<JobCreated>;
}
