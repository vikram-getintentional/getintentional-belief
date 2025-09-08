import { useEffect, useMemo, useState } from "react";

type NodeItem = {
  id: string;
  label: string;
  title?: string;
  department?: string;
  seniority?: string;
  description?: string;
  linkedin_url?: string;
};

export default function PersonaJobAnchor({ productId }: { productId: string }) {
  const token = localStorage.getItem("token");
  const auth = useMemo(() => ({ Authorization: `Bearer ${token}`, "Content-Type": "application/json" }), [token]);

  const [personas, setPersonas] = useState<NodeItem[]>([]);
  const [jobs, setJobs] = useState<NodeItem[]>([]);
  const [saving, setSaving] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  const isLikelyLinkedIn = (url?: string) => !!url && /https?:\/\/(www\.)?linkedin\.com\//i.test(url);

  const fetchData = async () => {
    setError(null);
    try {
      const [perRes, jobRes] = await Promise.all([
        fetch(`http://localhost:8000/graph/nodes/persona?product_id=${encodeURIComponent(productId)}`, { headers: auth }),
        fetch(`http://localhost:8000/graph/nodes/job?product_id=${encodeURIComponent(productId)}`, { headers: auth }),
      ]);
      const [perData, jobData] = await Promise.all([perRes.json(), jobRes.json()]);
      setPersonas(perData.nodes || []);
      setJobs(jobData.nodes || []);
    } catch (_e) {
      setError("Failed to load anchors");
    }
  };

  useEffect(() => { fetchData(); /* eslint-disable-next-line */ }, [productId]);

  const updatePersona = async (p: NodeItem) => {
    setSaving(p.id);
    setError(null);
    try {
      setNotice(null);
      const res = await fetch(`http://localhost:8000/graph/persona/${encodeURIComponent(p.id)}` ,{
        method: "PUT",
        headers: auth,
        body: JSON.stringify({ product_id: productId, linkedin_url: p.linkedin_url })
      });
      if (!res.ok) throw new Error(await res.text());
      setNotice("Persona anchor saved");
    } catch (_e) {
      setError("Failed to save persona anchor");
    } finally {
      setSaving(null);
    }
  };

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

  const setPersonaField = (id: string, value: string) => setPersonas(prev => prev.map(p => p.id === id ? { ...p, linkedin_url: value } : p));
  const setJobField = (id: string, value: string) => setJobs(prev => prev.map(j => j.id === id ? { ...j, linkedin_url: value } : j));

  return (
    <section className="mt-12">
      <h3 className="text-lg font-semibold text-indigo-700 mb-2">Persona & Job Anchors</h3>
      {error && <div className="text-red-600 mb-2">{error}</div>}
      {notice && <div className="text-green-700 mb-2">{notice}</div>}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <h4 className="font-medium text-gray-700 mb-2">Personas</h4>
          <div className="space-y-2">
            {personas.map(p => (
              <div key={p.id} className="p-3 border rounded bg-white">
                <div className="text-xs text-gray-400 mb-1">{p.id}</div>
                <div className="text-sm font-medium">{p.title || p.label}</div>
                <div className="text-xs text-gray-500">{[p.department, p.seniority].filter(Boolean).join(" • ")}</div>
                <div className="mt-2">
                  <label className="block text-sm">LinkedIn URL</label>
                  <input className={`w-full border rounded px-2 py-1 ${p.linkedin_url && !isLikelyLinkedIn(p.linkedin_url) ? 'border-red-400' : ''}`} placeholder="https://www.linkedin.com/in/..." value={p.linkedin_url || ""} onChange={e => setPersonaField(p.id, e.target.value)} />
                  {p.linkedin_url && <a className="text-indigo-600 text-xs" href={p.linkedin_url} target="_blank" rel="noreferrer">Preview</a>}
                </div>
                <div className="mt-2">
                  <button onClick={() => updatePersona(p)} disabled={saving === p.id} className="px-3 py-1 rounded bg-indigo-600 text-white text-sm">{saving === p.id ? "Saving…" : "Save"}</button>
                </div>
              </div>
            ))}
            {personas.length === 0 && <div className="text-gray-400">No personas yet</div>}
          </div>
        </div>
        <div>
          <h4 className="font-medium text-gray-700 mb-2">Jobs</h4>
          <div className="space-y-2">
            {jobs.map(j => (
              <div key={j.id} className="p-3 border rounded bg-white">
                <div className="text-xs text-gray-400 mb-1">{j.id}</div>
                <div className="text-sm font-medium">{j.label}</div>
                <div className="mt-2">
                  <label className="block text-sm">LinkedIn URL</label>
                  <input className={`w-full border rounded px-2 py-1 ${j.linkedin_url && !isLikelyLinkedIn(j.linkedin_url) ? 'border-red-400' : ''}`} placeholder="https://www.linkedin.com/in/..." value={j.linkedin_url || ""} onChange={e => setJobField(j.id, e.target.value)} />
                  {j.linkedin_url && <a className="text-indigo-600 text-xs" href={j.linkedin_url} target="_blank" rel="noreferrer">Preview</a>}
                </div>
                <div className="mt-2">
                  <button onClick={() => updateJob(j)} disabled={saving === j.id} className="px-3 py-1 rounded bg-indigo-600 text-white text-sm">{saving === j.id ? "Saving…" : "Save"}</button>
                </div>
              </div>
            ))}
            {jobs.length === 0 && <div className="text-gray-400">No jobs yet</div>}
          </div>
        </div>
      </div>
    </section>
  );
}
