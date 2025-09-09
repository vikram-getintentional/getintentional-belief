import { useEffect, useMemo, useState } from "react";

type NodeItem = {
  id: string;
  label: string;
  description?: string;
  linkedin_url?: string;
};

export default function JobAnchorEditor({ productId }: { productId: string }) {
  const token = (typeof window !== 'undefined') ? localStorage.getItem("token") : null;
  const auth = useMemo(() => ({ Authorization: `Bearer ${token}`, "Content-Type": "application/json" }), [token]);

  const [jobs, setJobs] = useState<NodeItem[]>([]);
  const [saving, setSaving] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [confirm, setConfirm] = useState<{
    open: boolean;
    targetId?: string;
    preview?: { counts: Record<string, number>; nodes: { id: string; node_type: string; label: string }[] } | null;
    loading?: boolean;
    error?: string | null;
  }>({ open: false, preview: null, loading: false, error: null });

  const fetchJobs = async () => {
    try {
      const res = await fetch(`http://localhost:8000/graph/nodes/job?product_id=${encodeURIComponent(productId)}`, { headers: auth });
      const data = await res.json();
      setJobs((data.nodes || []).map((n: any) => ({ id: n.id, label: n.label, description: n.description, linkedin_url: n.linkedin_url })));
    } catch (_e) {
      setError("Failed to load jobs");
    }
  };

  useEffect(() => { if (productId) fetchJobs(); /* eslint-disable-next-line */ }, [productId]);

  const setJobField = (id: string, value: string) => setJobs(prev => prev.map(j => j.id === id ? { ...j, linkedin_url: value } : j));

  const updateJob = async (j: NodeItem) => {
    setSaving(j.id);
    setError(null);
    try {
      setNotice(null);
      const res = await fetch(`http://localhost:8000/graph/job/${encodeURIComponent(j.id)}` ,{
        method: "PUT",
        headers: auth,
        body: JSON.stringify({ product_id: productId, linkedin_url: j.linkedin_url })
      });
      if (!res.ok) throw new Error(await res.text());
      setNotice("Job anchor saved");
    } catch (_e) {
      setError("Failed to save job anchor");
    } finally {
      setSaving(null);
    }
  };

  const previewDelete = async (id: string) => {
    setConfirm({ open: true, targetId: id, preview: null, loading: true, error: null });
    try {
      const url = new URL(`http://localhost:8000/graph/job/${encodeURIComponent(id)}/orphan-preview`);
      url.searchParams.set("product_id", productId);
      const res = await fetch(url.toString(), { headers: { Authorization: `Bearer ${token}` } });
      const data = await res.json();
      if (!res.ok) throw new Error(data?.detail || 'Preview failed');
      setConfirm({ open: true, targetId: id, preview: { counts: data.counts || {}, nodes: data.nodes || [] }, loading: false, error: null });
    } catch (_e) {
      setConfirm(c => ({ ...c, loading: false, error: 'Could not load orphan preview. You can still proceed.' }));
    }
  };

  const performDelete = async () => {
    if (!confirm.targetId) { setConfirm({ open: false, preview: null, loading: false, error: null }); return; }
    try {
      const url = new URL(`http://localhost:8000/graph/job/${encodeURIComponent(confirm.targetId)}`);
      url.searchParams.set('product_id', productId);
      if (confirm.preview && (confirm.preview.nodes || []).length > 0) url.searchParams.set('force', 'true');
      const res = await fetch(url.toString(), { method: 'DELETE', headers: { Authorization: `Bearer ${token}` } });
      if (res.status === 409) {
        const data = await res.json().catch(() => ({}));
        const detail = data?.detail || data;
        setConfirm(c => ({ ...c, preview: { counts: detail.counts || {}, nodes: detail.nodes || [] }, loading: false }));
        return;
      }
      if (!res.ok) throw new Error(await res.text());
      await fetchJobs();
      setConfirm({ open: false, preview: null, loading: false, error: null });
      setNotice('Job deleted');
    } catch (_e) {
      setError('Failed to delete job');
    }
  };

  return (
    <section className="mt-12">
      <div className="mb-3">
        <h3 className="text-lg font-semibold text-slate-800">Job Anchors</h3>
        <p className="text-sm text-slate-500">Attach LinkedIn profiles to job nodes and manage deletions with safety checks.</p>
      </div>
      {error && <div className="mb-3 rounded border border-red-200 bg-red-50 px-3 py-2 text-red-700">{error}</div>}
      {notice && <div className="mb-3 rounded border border-emerald-200 bg-emerald-50 px-3 py-2 text-emerald-700">{notice}</div>}
      <div className="space-y-3">
        {jobs.map(j => (
          <div key={j.id} className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
            <div className="mb-2 flex items-center justify-between">
              <div className="text-xs text-slate-400">{j.id}</div>
              <div className="text-sm font-medium text-slate-800">{j.label || j.description || j.id}</div>
            </div>
            <div>
              <label className="mb-1 block text-sm font-medium text-slate-700">LinkedIn URL</label>
              <input className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500" placeholder="https://www.linkedin.com/in/..." value={j.linkedin_url || ''} onChange={e => setJobField(j.id, e.target.value)} />
            </div>
            <div className="mt-3 flex items-center justify-end gap-2">
              <button type="button" onClick={() => previewDelete(j.id)} className="inline-flex items-center gap-1.5 rounded-md border border-red-600 bg-white px-4 py-2 text-sm font-medium text-red-700 shadow-sm transition hover:bg-red-50 focus:outline-none focus:ring-2 focus:ring-red-500 focus:ring-offset-2">Delete</button>
              <button type="button" onClick={() => updateJob(j)} disabled={saving === j.id} className="inline-flex items-center gap-1.5 rounded-md bg-indigo-600 px-4 py-2 text-sm font-medium text-white shadow-sm transition hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 disabled:opacity-60 disabled:cursor-not-allowed">{saving === j.id ? 'Saving…' : 'Save'}</button>
            </div>
          </div>
        ))}
        {jobs.length === 0 && (
          <div className="rounded border border-dashed border-slate-300 bg-slate-50 p-6 text-center text-slate-500">No jobs yet</div>
        )}
      </div>

      {confirm.open && (
        <div className="fixed inset-0 z-50 flex items-center justify-center">
          <div className="absolute inset-0 bg-slate-900/50 backdrop-blur-sm" onClick={() => setConfirm({ open: false, preview: null, loading: false, error: null })} />
          <div className="relative mx-4 w-full max-w-2xl overflow-hidden rounded-xl border border-slate-200 bg-white shadow-2xl">
            <div className="border-b border-slate-200 bg-slate-50 px-5 py-3">
              <h4 className="text-lg font-semibold text-slate-800">Delete Job</h4>
              <p className="text-sm text-slate-600">Review impacted nodes before confirming.</p>
            </div>
            <div className="max-h-[60vh] overflow-auto px-5 py-4">
              {confirm.loading && <div className="text-slate-500">Loading impact…</div>}
              {confirm.error && <div className="mb-3 rounded border border-amber-200 bg-amber-50 px-3 py-2 text-amber-800">{confirm.error}</div>}
              {!confirm.loading && (
                <div className="space-y-3">
                  {Object.values(confirm.preview?.counts || {}).reduce((a: any, b: any) => (a as number) + (b as number), 0) > 0 ? (
                    <div className="rounded-md border border-amber-200 bg-amber-50 p-3 text-amber-900">
                      <div className="mb-1 font-medium">This delete will orphan nodes:</div>
                      <div className="flex flex-wrap gap-2">
                        {Object.entries(confirm.preview?.counts || {}).map(([k, v]) => (
                          <span key={k} className="inline-flex items-center rounded-full bg-slate-100 px-2 py-0.5 text-xs font-medium text-slate-700">{k}: {v as number}</span>
                        ))}
                      </div>
                    </div>
                  ) : (
                    <div className="rounded-md border border-emerald-200 bg-emerald-50 p-3 text-emerald-800">No downstream nodes will be orphaned.</div>
                  )}
                  {(confirm.preview?.nodes || []).length > 0 && (
                    <div>
                      <div className="mb-2 text-sm font-medium text-slate-700">Impacted nodes</div>
                      <ul className="divide-y divide-slate-200 rounded-md border border-slate-200">
                        {confirm.preview?.nodes?.map(n => (
                          <li key={n.id} className="flex items-center justify-between gap-3 px-3 py-2">
                            <div className="min-w-0">
                              <div className="truncate text-sm font-medium text-slate-800">{n.label}</div>
                              <div className="truncate text-xs text-slate-500">{n.id}</div>
                            </div>
                            <span className="inline-flex items-center rounded-full bg-slate-100 px-2 py-0.5 text-xs font-medium text-slate-700">{n.node_type}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              )}
            </div>
            <div className="flex items-center justify-end gap-2 border-t border-slate-200 bg-slate-50 px-5 py-3">
              <button onClick={() => setConfirm({ open: false, preview: null, loading: false, error: null })} className="rounded-md border border-slate-300 bg-white px-3 py-2 text-sm font-medium text-slate-700 shadow-sm hover:bg-slate-50">Cancel</button>
              <button onClick={performDelete} className="rounded-md bg-red-700 px-3 py-2 text-sm font-medium text-white shadow-sm hover:bg-red-800">Delete anyway</button>
            </div>
          </div>
        </div>
      )}
    </section>
  );
}

