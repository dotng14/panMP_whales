import { API_BASE } from './base';
import type { JobInfo } from './types';

export async function fetchJob(jobId: string): Promise<JobInfo> {
  const res = await fetch(`${API_BASE}/api/jobs/${jobId}`);
  if (!res.ok) throw new Error(`Job fetch failed: ${res.status}`);
  return res.json() as Promise<JobInfo>;
}
