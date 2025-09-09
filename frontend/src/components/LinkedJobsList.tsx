import React from "react";

export type JobNode = { id: string; label?: string; description?: string; linkedin_url?: string };

type Props = {
  jobs: JobNode[];
  onChange: (jobId: string, linkedin_url: string) => void;
  onDelete: (jobId: string) => void;
  onSave: (jobId: string) => void;
};

export default function LinkedJobsList({ jobs, onChange, onDelete, onSave }: Props) {
  return (
    <div className="mt-3">
      <div className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">Linked Jobs</div>
      <ul className="space-y-3">
        {jobs.map(j => (
          <li key={j.id}>
            <div className="rounded-xl border border-slate-200 bg-white/80 p-4 shadow-sm ring-1 ring-transparent transition hover:shadow-md hover:ring-indigo-100">
              <div className="mb-3 flex items-center justify-between">
                <div className="min-w-0">
                  <div className="truncate text-[13px] font-semibold text-slate-800 md:text-sm">{(j.label || j.description) ? (j.label || j.description) : j.id}</div>
                  <div className="mt-0.5 inline-flex items-center gap-2">
                    <code className="truncate rounded bg-slate-50 px-1.5 py-0.5 text-[10px] font-mono text-slate-500 ring-1 ring-slate-200">{j.id}</code>
                    {j.linkedin_url ? (
                      <span className="inline-flex items-center rounded-full bg-emerald-50 px-2 py-0.5 text-[10px] font-medium text-emerald-700 ring-1 ring-emerald-200">Linked ✓</span>
                    ) : (
                      <span className="inline-flex items-center rounded-full bg-amber-50 px-2 py-0.5 text-[10px] font-medium text-amber-700 ring-1 ring-amber-200">Not linked</span>
                    )}
                    {j.linkedin_url && (
                      <a href={j.linkedin_url} target="_blank" rel="noreferrer" className="text-[10px] font-medium text-indigo-600 hover:text-indigo-700">Open</a>
                    )}
                  </div>
                </div>
              </div>
              <div className="grid grid-cols-1 gap-3 md:grid-cols-5">
                <div className="md:col-span-4">
                  <label className="mb-1 block text-xs font-medium text-slate-700">LinkedIn URL</label>
                  <input
                    className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-200"
                    placeholder="https://www.linkedin.com/in/..."
                    value={j.linkedin_url || ''}
                    onChange={e => onChange(j.id, e.target.value)}
                  />
                  <p className="mt-1 text-[11px] text-slate-500">Optional. Anchors this job to a real profile for validation.</p>
                </div>
                <div className="flex items-end justify-end gap-2">
                  <button
                    type="button"
                    onClick={() => onDelete(j.id)}
                    className="inline-flex items-center gap-1.5 rounded-md border border-red-600 bg-white px-3 py-2 text-xs font-medium text-red-700 shadow-sm transition hover:bg-red-50 focus:outline-none focus:ring-2 focus:ring-red-500 focus:ring-offset-2"
                  >
                    Delete
                  </button>
                  <button
                    type="button"
                    onClick={() => onSave(j.id)}
                    className="inline-flex items-center gap-1.5 rounded-md bg-indigo-600 px-3 py-2 text-xs font-medium text-white shadow-sm transition hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2"
                  >
                    Save
                  </button>
                </div>
              </div>
            </div>
          </li>
        ))}
        {jobs.length === 0 && (
          <li className="px-2 py-2 text-xs text-slate-500">No jobs linked to this persona.</li>
        )}
      </ul>
    </div>
  );
}

