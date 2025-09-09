import React from "react";

export type JobNode = {
  id: string;
  label?: string;
  description?: string;
  linkedin_url?: string;
};

type Props = {
  jobs: JobNode[];
  onChange: (jobId: string, linkedin_url: string) => void;
  onDelete: (jobId: string) => void;
  onSave: (jobId: string) => void;
};

export default function LinkedJobsList({ jobs, onChange, onDelete, onSave }: Props) {
  if (!jobs || jobs.length === 0) {
    return <div className="px-2 py-2 text-sm text-slate-500">No jobs linked to this persona.</div>;
  }

  return (
    <ul className="space-y-3">
      {jobs.map((j) => {
        const title = j.label || j.description || j.id;
        const isLinked = Boolean(j.linkedin_url);
        return (
          <li key={j.id}>
            <div className="group rounded-xl border border-slate-200 bg-white p-4 shadow-sm ring-1 ring-transparent transition hover:shadow-md hover:ring-indigo-100">
              <div className="mb-3">
                <div className="min-w-0">
                  <div className="truncate text-sm font-semibold text-slate-800">{title}</div>
                  <div className="mt-1 inline-flex flex-wrap items-center gap-2">
                    <code className="truncate rounded bg-slate-50 px-1.5 py-0.5 text-[11px] font-mono text-slate-600 ring-1 ring-slate-200">{j.id}</code>
                    {isLinked ? (
                      <span className="inline-flex items-center rounded-full bg-emerald-50 px-2 py-0.5 text-[11px] font-medium text-emerald-700 ring-1 ring-emerald-200">Linked ✓</span>
                    ) : (
                      <span className="inline-flex items-center rounded-full bg-amber-50 px-2 py-0.5 text-[11px] font-medium text-amber-700 ring-1 ring-amber-200">Not linked</span>
                    )}
                    {isLinked && (
                      <a
                        href={j.linkedin_url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="inline-flex items-center gap-1 rounded-md bg-indigo-50 px-1.5 py-0.5 text-[11px] font-medium text-indigo-700 ring-1 ring-indigo-200 transition hover:bg-indigo-100"
                      >
                        Open <span aria-hidden>↗</span>
                      </a>
                    )}
                  </div>
                </div>
              </div>
              <div className="grid grid-cols-1 gap-3 md:grid-cols-12">
                <div className="md:col-span-9">
                  <label className="mb-1 block text-xs font-medium text-slate-700">LinkedIn URL</label>
                  <input
                    type="url"
                    autoComplete="off"
                    className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm text-slate-800 placeholder-slate-400 shadow-sm transition focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-200"
                    placeholder="https://www.linkedin.com/in/..."
                    value={j.linkedin_url || ""}
                    onChange={(e) => onChange(j.id, e.target.value)}
                  />
                </div>
                <div className="md:col-span-3 flex items-end justify-end gap-2">
                  <button
                    type="button"
                    onClick={() => onDelete(j.id)}
                    className="inline-flex items-center gap-1.5 rounded-lg border border-red-600 bg-white px-3 py-2 text-xs font-medium text-red-700 shadow-sm transition hover:bg-red-50 focus:outline-none focus-visible:ring-2 focus-visible:ring-red-500 focus-visible:ring-offset-2"
                  >
                    Delete
                  </button>
                  <button
                    type="button"
                    onClick={() => onSave(j.id)}
                    className="inline-flex items-center gap-1.5 rounded-lg bg-indigo-600 px-3 py-2 text-xs font-medium text-white shadow-sm transition hover:bg-indigo-700 focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 focus-visible:ring-offset-2"
                  >
                    Save
                  </button>
                </div>
              </div>
            </div>
          </li>
        );
      })}
    </ul>
  );
}
