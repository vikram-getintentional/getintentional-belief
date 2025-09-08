import { useEffect, useMemo, useState } from "react";

type NodeItem = { id: string; label: string } & Record<string, any>;
type Persona = { id: string; title?: string; department?: string; seniority?: string };

type FamilyItem = {
  id: string;
  capability_id: string;
  pain?: { id: string; text: string };
  job?: { id: string; text: string };
  persona?: Persona;
  relevance?: number;
  likelihood?: number;
};

const REL_OPTS = ["Critical", "Core", "Supportive", "Ancillary"] as const;

export default function PainFamilyEditor({ productId }: { productId: string }) {
  const token = localStorage.getItem("token");
  const auth = useMemo(() => ({ Authorization: `Bearer ${token}`, "Content-Type": "application/json" }), [token]);

  const [capabilities, setCapabilities] = useState<NodeItem[]>([]);
  const [selectedCap, setSelectedCap] = useState<string>("");
  const [families, setFamilies] = useState<FamilyItem[]>([]);
  const [pains, setPains] = useState<NodeItem[]>([]);
  const [jobs, setJobs] = useState<NodeItem[]>([]);
  const [personas, setPersonas] = useState<NodeItem[]>([]);
  const [newPain, setNewPain] = useState<{ mode: "existing"|"new"; value: string; id?: string }>({ mode: "existing", value: "", id: "" });
  const [newJob, setNewJob] = useState<{ mode: "existing"|"new"; value: string; id?: string }>({ mode: "existing", value: "", id: "" });
  const [newPersona, setNewPersona] = useState<{ mode: "existing"|"new"; id?: string; title?: string; department?: string; seniority?: string; linkedin_url?: string }>({ mode: "existing", id: "" });
  const [relevance, setRelevance] = useState<string>("");
  const [likelihood, setLikelihood] = useState<number>(50);
  const [error, setError] = useState<string| null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [confirm, setConfirm] = useState<{
    open: boolean;
    nodeType?: 'job' | 'persona';
    targetId?: string;
    preview?: { counts: Record<string, number>; nodes: { id: string; node_type: string; label: string }[] } | null;
    loading?: boolean;
    error?: string | null;
  }>({ open: false, preview: null, loading: false, error: null });

  const fetchCaps = async () => {
    const res = await fetch(`http://localhost:8000/graph/nodes/capability?product_id=${encodeURIComponent(productId)}`, { headers: auth });
    const data = await res.json();
    setCapabilities(data.nodes || []);
    if ((data.nodes || []).length && !selectedCap) setSelectedCap((data.nodes || [])[0].id);
  };

  const fetchDropdowns = async () => {
    const [pRes, jRes, perRes] = await Promise.all([
      fetch(`http://localhost:8000/graph/nodes/pain?product_id=${encodeURIComponent(productId)}`, { headers: auth }),
      fetch(`http://localhost:8000/graph/nodes/job?product_id=${encodeURIComponent(productId)}`, { headers: auth }),
      fetch(`http://localhost:8000/graph/nodes/persona?product_id=${encodeURIComponent(productId)}`, { headers: auth })
    ]);
    const [pData, jData, perData] = await Promise.all([pRes.json(), jRes.json(), perRes.json()]);
    setPains(pData.nodes || []);
    setJobs(jData.nodes || []);
    setPersonas(perData.nodes || []);
  };

  const fetchFamilies = async () => {
    if (!selectedCap) return;
    const url = new URL(`http://localhost:8000/graph/pain_families`);
    url.searchParams.set("product_id", productId);
    url.searchParams.set("capability_id", selectedCap);
    const res = await fetch(url, { headers: auth });
    const data = await res.json();
    setFamilies(data.items || []);
  };

  useEffect(() => { fetchCaps(); fetchDropdowns(); /* eslint-disable-next-line */ }, [productId]);
  useEffect(() => { fetchFamilies(); /* eslint-disable-next-line */ }, [selectedCap]);

  const resetNew = () => {
    setNewPain({ mode: "existing", value: "", id: "" });
    setNewJob({ mode: "existing", value: "", id: "" });
    setNewPersona({ mode: "existing", id: "" });
    setRelevance("");
    setLikelihood(50);
  };

  const addFamily = async () => {
    setError(null);
    setNotice(null);
    if (!selectedCap) { setError("Select a capability first"); return; }
    // Guardrails
    const pain_id = newPain.mode === "existing" ? newPain.id : undefined;
    const job_id = newJob.mode === "existing" ? newJob.id : undefined;
    const persona_id = newPersona.mode === "existing" ? newPersona.id : undefined;
    const pain_ok = (newPain.mode === "existing" && !!pain_id) || (newPain.mode === "new" && !!newPain.value.trim());
    const job_ok = (newJob.mode === "existing" && !!job_id) || (newJob.mode === "new" && !!newJob.value.trim());
    // If creating a new persona, enforce job present per guardrails
    if (newPersona.mode === "new" && !job_ok) { setError("To add a new persona, select or create a job first."); return; }
    if (!pain_ok) { setError("Select or enter a pain."); return; }
    const payload: any = { product_id: productId, capability_id: selectedCap, relevance: relevance || undefined, likelihood };
    if (newPain.mode === "new") payload.pain = newPain.value;
    else payload.pain_id = pain_id;
    if (newJob.mode === "new") payload.job = newJob.value;
    else payload.job_id = job_id;
    if (newPersona.mode === "new") payload.persona = { title: newPersona.title || "", department: newPersona.department || "", seniority: newPersona.seniority || "", linkedin_url: newPersona.linkedin_url || "" };
    else if (persona_id) payload.persona_id = persona_id;
    try {
      const res = await fetch(`http://localhost:8000/graph/pain_family`, { method: "POST", headers: auth, body: JSON.stringify(payload) });
      if (!res.ok) throw new Error(await res.text());
      resetNew();
      await fetchFamilies();
      setNotice("Pain family added");
    } catch (_e) {
      setError("Failed to create pain family (check selections and guardrails)");
    }
  };

  const previewDelete = async (nodeType: 'job' | 'persona', id: string) => {
    setConfirm({ open: true, nodeType, targetId: id, preview: null, loading: true, error: null });
    try {
      const url = new URL(`http://localhost:8000/graph/${nodeType}/${encodeURIComponent(id)}/orphan-preview`);
      url.searchParams.set("product_id", productId);
      const res = await fetch(url.toString(), { headers: auth });
      const data = await res.json();
      if (!res.ok) throw new Error(data?.detail || 'Preview failed');
      setConfirm({ open: true, nodeType, targetId: id, preview: { counts: data.counts || {}, nodes: data.nodes || [] }, loading: false, error: null });
    } catch (e) {
      setConfirm(c => ({ ...c, loading: false, error: 'Could not load orphan preview. You can still proceed.' }));
    }
  };

  const performDelete = async () => {
    if (!confirm.targetId || !confirm.nodeType) { setConfirm({ open: false, preview: null, loading: false, error: null }); return; }
    try {
      const url = new URL(`http://localhost:8000/graph/${confirm.nodeType}/${encodeURIComponent(confirm.targetId)}`);
      url.searchParams.set("product_id", productId);
      if (confirm.preview && (confirm.preview.nodes || []).length > 0) url.searchParams.set("force", "true");
      const res = await fetch(url.toString(), { method: 'DELETE', headers: auth });
      if (res.status === 409) {
        const data = await res.json().catch(() => ({}));
        const detail = data?.detail || data;
        setConfirm(c => ({ ...c, preview: { counts: detail.counts || {}, nodes: detail.nodes || [] }, loading: false }));
        return;
      }
      if (!res.ok) throw new Error(await res.text());
      await fetchDropdowns();
      await fetchFamilies();
      setConfirm({ open: false, preview: null, loading: false, error: null });
      setNotice(`${confirm.nodeType === 'job' ? 'Job' : 'Persona'} deleted`);
    } catch (e) {
      setError(`Failed to delete ${confirm.nodeType}`);
    }
  };

  const updateFamily = async (item: FamilyItem, rel?: string, lk?: number) => {
    const parts = (item.id || "").split("|");
    const pfid = parts.join("|") || "pf"; // path param is unused server-side
    const payload: any = { product_id: productId, capability_id: item.capability_id, pain_id: item.pain?.id, job_id: item.job?.id, persona_id: item.persona?.id };
    if (rel !== undefined) payload.relevance = rel;
    if (lk !== undefined) payload.likelihood = lk;
    const res = await fetch(`http://localhost:8000/graph/pain_family/${encodeURIComponent(pfid)}`, { method: "PUT", headers: auth, body: JSON.stringify(payload) });
    if (!res.ok) throw new Error(await res.text());
    fetchFamilies();
  };

  return (
    <section className="mt-12">
      <h3 className="text-lg font-semibold text-indigo-700 mb-2">Pain Family Editor</h3>
      {error && <div className="text-red-600 mb-2">{error}</div>}
      {notice && <div className="text-green-700 mb-2">{notice}</div>}

      {/* Capability selector */}
      <div className="mb-4">
        <label className="mr-2 font-medium">Capability:</label>
        <select className="border rounded px-2 py-1" value={selectedCap} onChange={e => setSelectedCap(e.target.value)}>
          {capabilities.map(c => <option key={c.id} value={c.id}>{c.label || c.id}</option>)}
        </select>
      </div>

      {/* Existing families list */}
      <div className="space-y-2 mb-6">
        {families.map(item => (
          <div key={item.id} className="p-3 border rounded bg-white">
            <div className="text-xs text-gray-400 mb-1">{item.id}</div>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3 items-end">
              <div>
                <div className="text-sm"><span className="font-medium">Pain:</span> {item.pain?.text}</div>
                <div className="text-sm"><span className="font-medium">Job:</span> {item.job?.text || "-"}</div>
                <div className="text-sm"><span className="font-medium">Persona:</span> {item.persona?.title || "-"}</div>
              </div>
              <div>
                <label className="block text-sm font-medium">Relevance</label>
                <select className="w-full border rounded px-2 py-1" value={""} onChange={e => updateFamily(item, e.target.value, undefined)}>
                  <option value="">Update…</option>
                  {REL_OPTS.map(o => <option key={o} value={o}>{o}</option>)}
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium">Likelihood: {Math.round(item.likelihood ?? 0)}</label>
                <input type="range" min={0} max={100} value={Math.round(item.likelihood ?? 0)} onChange={e => updateFamily(item, undefined, Number(e.target.value))} />
              </div>
            </div>
          </div>
        ))}
        {families.length === 0 && <div className="text-gray-400">No pain families yet for this capability.</div>}
      </div>

      {/* Manage Jobs & Personas */}
      <div className="p-4 border rounded bg-white">
        <div className="flex items-center justify-between mb-2">
          <h4 className="font-medium text-indigo-600">Manage Jobs & Personas</h4>
          <div className="text-xs text-slate-500">Delete nodes with orphan preview</div>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <div className="mb-2 text-sm font-medium text-slate-700">Jobs</div>
            <ul className="divide-y divide-slate-200 rounded-md border border-slate-200 bg-slate-50">
              {jobs.map(j => (
                <li key={j.id} className="flex items-center justify-between gap-3 px-3 py-2">
                  <div className="min-w-0">
                    <div className="truncate text-sm font-medium text-slate-800">{j.label || j.id}</div>
                    <div className="truncate text-xs text-slate-500">{j.id}</div>
                  </div>
                  <button onClick={() => previewDelete('job', j.id)} className="rounded-md border border-red-900 bg-red-800 px-2 py-1 text-xs font-medium text-white hover:bg-red-900">Delete</button>
                </li>
              ))}
              {jobs.length === 0 && (<li className="px-3 py-2 text-sm text-slate-500">No jobs.</li>)}
            </ul>
          </div>
          <div>
            <div className="mb-2 text-sm font-medium text-slate-700">Personas</div>
            <ul className="divide-y divide-slate-200 rounded-md border border-slate-200 bg-slate-50">
              {personas.map(p => (
                <li key={p.id} className="flex items-center justify-between gap-3 px-3 py-2">
                  <div className="min-w-0">
                    <div className="truncate text-sm font-medium text-slate-800">{p.label || p.id}</div>
                    <div className="truncate text-xs text-slate-500">{p.id}</div>
                  </div>
                  <button onClick={() => previewDelete('persona', p.id)} className="rounded-md border border-red-900 bg-red-800 px-2 py-1 text-xs font-medium text-white hover:bg-red-900">Delete</button>
                </li>
              ))}
              {personas.length === 0 && (<li className="px-3 py-2 text-sm text-slate-500">No personas.</li>)}
            </ul>
          </div>
        </div>
      </div>

      {/* Create new family */}
      <div className="p-4 border rounded bg-white">
        <h4 className="font-medium text-indigo-600 mb-2">Add Pain Family</h4>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          {/* Pain */}
          <div>
            <label className="block text-sm font-medium">Pain</label>
            <select className="w-full border rounded px-2 py-1" value={newPain.mode === "existing" ? (newPain.id || "") : "+"} onChange={e => e.target.value === "+" ? setNewPain({ mode: "new", value: "" }) : setNewPain({ mode: "existing", id: e.target.value, value: "" })}>
              <option value="">Select pain…</option>
              {pains.map(p => <option key={p.id} value={p.id}>{p.label || p.id}</option>)}
              <option value="+">+ Add New Pain</option>
            </select>
            {newPain.mode === "new" && (
              <input className="mt-2 w-full border rounded px-2 py-1" placeholder="Enter pain text" value={newPain.value} onChange={e => setNewPain({ ...newPain, value: e.target.value })} />
            )}
          </div>
          {/* Job */}
          <div>
            <label className="block text-sm font-medium">Job</label>
            <select className="w-full border rounded px-2 py-1" value={newJob.mode === "existing" ? (newJob.id || "") : "+"} onChange={e => e.target.value === "+" ? setNewJob({ mode: "new", value: "" }) : setNewJob({ mode: "existing", id: e.target.value, value: "" })}>
              <option value="">Select job…</option>
              {jobs.map(j => <option key={j.id} value={j.id}>{j.label || j.id}</option>)}
              <option value="+">+ Add New Job</option>
            </select>
            {newJob.mode === "new" && (
              <input className="mt-2 w-full border rounded px-2 py-1" placeholder="Enter job description" value={newJob.value} onChange={e => setNewJob({ ...newJob, value: e.target.value })} />
            )}
          </div>
          {/* Persona */}
          <div>
            <label className="block text-sm font-medium">Persona</label>
            <select className="w-full border rounded px-2 py-1" value={newPersona.mode === "existing" ? (newPersona.id || "") : "+"} onChange={e => e.target.value === "+" ? setNewPersona({ mode: "new" }) : setNewPersona({ mode: "existing", id: e.target.value })}>
              <option value="">Select persona…</option>
              {personas.map(pr => <option key={pr.id} value={pr.id}>{pr.label || pr.id}</option>)}
              <option value="+">+ Add New Persona</option>
            </select>
            {newPersona.mode === "new" && (
              <div className="mt-2 space-y-2">
                <input className="w-full border rounded px-2 py-1" placeholder="Title" value={newPersona.title || ""} onChange={e => setNewPersona({ ...newPersona, title: e.target.value })} />
                <input className="w-full border rounded px-2 py-1" placeholder="Department" value={newPersona.department || ""} onChange={e => setNewPersona({ ...newPersona, department: e.target.value })} />
                <input className="w-full border rounded px-2 py-1" placeholder="Seniority" value={newPersona.seniority || ""} onChange={e => setNewPersona({ ...newPersona, seniority: e.target.value })} />
                <input className="w-full border rounded px-2 py-1" placeholder="LinkedIn URL (optional)" value={newPersona.linkedin_url || ""} onChange={e => setNewPersona({ ...newPersona, linkedin_url: e.target.value })} />
              </div>
            )}
          </div>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mt-3">
          <div>
            <label className="block text-sm font-medium">Relevance</label>
            <select className="w-full border rounded px-2 py-1" value={relevance} onChange={e => setRelevance(e.target.value)}>
              <option value="">Select…</option>
              {REL_OPTS.map(o => <option key={o} value={o}>{o}</option>)}
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium">Likelihood: {likelihood}</label>
            <input type="range" min={0} max={100} value={likelihood} onChange={e => setLikelihood(Number(e.target.value))} />
          </div>
        </div>
        <div className="mt-3">
          <button onClick={addFamily} className="px-3 py-1 rounded bg-indigo-600 text-white text-sm">Add Pain Family</button>
        </div>
    </div>
    {/* Modal */}
    {confirm.open && (
      <div className="fixed inset-0 z-50 flex items-center justify-center">
        <div className="absolute inset-0 bg-slate-900/50 backdrop-blur-sm" onClick={() => setConfirm({ open: false, preview: null, loading: false, error: null })} />
        <div className="relative mx-4 w-full max-w-2xl overflow-hidden rounded-xl border border-slate-200 bg-white shadow-2xl">
          <div className="border-b border-slate-200 bg-slate-50 px-5 py-3">
            <h4 className="text-lg font-semibold text-slate-800">Delete {confirm.nodeType}</h4>
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
