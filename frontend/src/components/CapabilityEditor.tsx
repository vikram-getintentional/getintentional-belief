import { useEffect, useMemo, useState } from "react";

type CapabilityNode = {
  id: string;
  label: string;
  name?: string;
  description?: string;
  coreness?: number | string;
  coreness_label?: string;
  buying_likelihood?: number;
};

type CapRow = {
  id: string;
  name: string;
  description: string;
  coreness: string; // label
  buyingLikelihood: number; // 0-100
};

const CORENESS_OPTIONS = ["Critical", "Core", "Supportive", "Ancillary"] as const;

const numericToCorenessLabel = (n?: number | null): string => {
  if (n == null) return "";
  if (n >= 0.99) return "Critical";
  if (n >= 0.79) return "Core";
  if (n >= 0.59) return "Supportive";
  if (n > 0) return "Ancillary";
  return "";
};

export default function CapabilityEditor({ productId }: { productId: string }) {
  const token = localStorage.getItem("token");
  const auth = useMemo(() => ({ Authorization: `Bearer ${token}` }), [token]);

  const [caps, setCaps] = useState<CapRow[]>([]);
  const [mergeTarget, setMergeTarget] = useState<Record<string, string>>({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState<string | null>(null);
  const [justSaved, setJustSaved] = useState<Record<string, boolean>>({});

  const fetchCapabilities = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(
        `http://localhost:8000/graph/nodes/capability?product_id=${encodeURIComponent(
          productId
        )}`,
        { headers: auth }
      );
      const data = await res.json();
      const rows: CapRow[] = (data.nodes || []).map((n: CapabilityNode) => {
        const name = (n as any).name || n.label || "";
        const description = (n as any).description || "";
        const buying = typeof n.buying_likelihood === "number" ? n.buying_likelihood : 0;
        const label = (n.coreness_label as string) || numericToCorenessLabel(typeof n.coreness === "number" ? n.coreness : undefined);
        return {
          id: n.id,
          name,
          description,
          coreness: label || "",
          buyingLikelihood: buying,
        };
      });
      setCaps(rows);
    } catch (e) {
      setError("Failed to load capabilities");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (productId) fetchCapabilities();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [productId]);

  const updateField = (id: string, field: keyof CapRow, value: any) => {
    setCaps((prev) => prev.map((c) => (c.id === id ? { ...c, [field]: value } : c)));
  };

  const saveCapability = async (cap: CapRow) => {
    setError(null);
    try {
      setSaving(cap.id);
      const res = await fetch(`http://localhost:8000/graph/capability/${cap.id}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json", ...auth },
        body: JSON.stringify({
          product_id: productId,
          name: cap.name,
          description: cap.description,
          coreness: cap.coreness || undefined,
          buying_likelihood: cap.buyingLikelihood,
        }),
      });
      if (!res.ok) throw new Error(await res.text());
      setJustSaved(prev => ({ ...prev, [cap.id]: true }));
      setTimeout(() => setJustSaved(prev => ({ ...prev, [cap.id]: false })), 1500);
    } catch (e) {
      setError("Failed to save capability");
    } finally {
      setSaving(null);
    }
  };

  const addCapability = async () => {
    setError(null);
    try {
      const res = await fetch(`http://localhost:8000/graph/capability`, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...auth },
        body: JSON.stringify({ product_id: productId, name: "", description: "" }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data?.detail || "Create failed");
      setCaps((prev) => [
        ...prev,
        {
          id: data.id,
          name: data.name || "",
          description: data.description || "",
          coreness: (data.coreness as string) || "",
          buyingLikelihood: Number(data.buying_likelihood) || 0,
        },
      ]);
    } catch (e) {
      setError("Failed to add capability");
    }
  };

  const deleteCapability = async (id: string) => {
    setError(null);
    try {
      const url = new URL(`http://localhost:8000/graph/capability/${id}`);
      url.searchParams.set("product_id", productId);
      const res = await fetch(url, { method: "DELETE", headers: auth });
      if (!res.ok) throw new Error(await res.text());
      setCaps((prev) => prev.filter((c) => c.id !== id));
    } catch (e) {
      setError("Failed to delete capability");
    }
  };

  const mergeCapability = async (sourceId: string) => {
    const targetId = mergeTarget[sourceId];
    if (!targetId || targetId === sourceId) return;
    setError(null);
    try {
      const res = await fetch(`http://localhost:8000/graph/capability/merge`, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...auth },
        body: JSON.stringify({ product_id: productId, source_id: sourceId, target_id: targetId }),
      });
      if (!res.ok) throw new Error(await res.text());
      // refresh list
      fetchCapabilities();
    } catch (e) {
      setError("Failed to merge capabilities");
    }
  };

  return (
    <div className="mt-12">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-xl font-semibold text-slate-800">Capability Editor</h3>
          <p className="text-sm text-slate-500">Review, edit, and refine your product capabilities. Use coreness to indicate strategic importance.</p>
        </div>
        <button onClick={addCapability} className="inline-flex items-center gap-2 rounded-md bg-indigo-600 px-3 py-2 text-sm font-medium text-white shadow-sm hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500">
          <span className="text-lg leading-none">＋</span> Add Capability
        </button>
      </div>
      {loading && <div className="text-slate-500">Loading…</div>}
      {error && <div className="mb-3 rounded border border-red-200 bg-red-50 px-3 py-2 text-red-700">{error}</div>}
      <div className="grid grid-cols-1 gap-4">
        {caps.map((cap) => (
          <div key={cap.id} className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm transition hover:shadow-md">
            <div className="mb-2 flex items-center justify-between">
              <div className="text-xs text-slate-400">{cap.id}</div>
              {cap.coreness && (
                <span className="inline-flex items-center rounded-full bg-slate-100 px-2 py-0.5 text-xs font-medium text-slate-700">
                  {cap.coreness}
                </span>
              )}
            </div>
            <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
              <div className="md:col-span-1">
                <label className="mb-1 block text-sm font-medium text-slate-700">Name</label>
                <input className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500" value={cap.name} onChange={(e) => updateField(cap.id, "name", e.target.value)} />
              </div>
              <div className="md:col-span-1">
                <label className="mb-1 block text-sm font-medium text-slate-700">Coreness</label>
                <select className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500" value={cap.coreness} onChange={(e) => updateField(cap.id, "coreness", e.target.value)}>
                  <option value="">Select…</option>
                  {CORENESS_OPTIONS.map((o) => (
                    <option key={o} value={o}>{o}</option>
                  ))}
                </select>
              </div>
              <div className="md:col-span-1">
                <label className="mb-1 block text-sm font-medium text-slate-700">Buying Likelihood <span className="ml-1 text-xs text-slate-400">({cap.buyingLikelihood})</span></label>
                <input type="range" min={0} max={100} value={cap.buyingLikelihood} onChange={(e) => updateField(cap.id, "buyingLikelihood", Number(e.target.value))} className="w-full accent-indigo-600" />
                <div className="mt-1 h-1 w-full rounded bg-slate-200">
                  <div className="h-1 rounded bg-gradient-to-r from-sky-400 via-indigo-500 to-fuchsia-500" style={{ width: `${cap.buyingLikelihood}%` }} />
                </div>
              </div>
              <div className="md:col-span-3">
                <label className="mb-1 block text-sm font-medium text-slate-700">Description</label>
                <textarea className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500" rows={2} value={cap.description} onChange={(e) => updateField(cap.id, "description", e.target.value)} />
              </div>
            </div>
            <div className="mt-4 -mx-4 rounded-b-lg border-t border-slate-200 bg-slate-50 px-4 py-3">
              <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => saveCapability(cap)}
                    disabled={saving === cap.id}
                    className={[
                      "rounded-md px-3 py-2 text-sm font-medium !text-white shadow-sm focus:outline-none focus:ring-2 focus:ring-offset-2 transition",
                      saving === cap.id
                        ? "!bg-green-500 cursor-wait focus:ring-green-300"
                        : justSaved[cap.id]
                          ? "!bg-emerald-600 hover:!bg-emerald-700 focus:ring-emerald-500"
                          : "!bg-green-700 hover:!bg-green-800 focus:ring-green-600",
                      saving === cap.id ? "opacity-90" : "",
                      "disabled:opacity-60 disabled:cursor-not-allowed",
                    ].join(" ")}
                  >
                    {saving === cap.id ? "Saving…" : justSaved[cap.id] ? "Saved" : "Save"}
                  </button>
                  <button
                    type="button"
                    onClick={() => deleteCapability(cap.id)}
                    className="appearance-none rounded-md border border-red-900 !bg-red-800 px-3 py-2 text-sm font-medium !text-white shadow-sm transition-colors hover:!bg-red-900 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-red-800"
                  >
                    Delete
                  </button>
                </div>
                <div className="flex items-center gap-2">
                  <label className="text-sm text-slate-600">Merge into</label>
                  <select
                    className="rounded-md border border-slate-300 px-2 py-1 text-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500"
                    value={mergeTarget[cap.id] || ""}
                    onChange={(e) => setMergeTarget((m) => ({ ...m, [cap.id]: e.target.value }))}
                  >
                    <option value="">Select target…</option>
                    {caps
                      .filter((c) => c.id !== cap.id)
                      .map((c) => (
                        <option key={c.id} value={c.id}>
                          {c.name || c.id}
                        </option>
                      ))}
                  </select>
                  <button
                    onClick={() => mergeCapability(cap.id)}
                    className="inline-flex items-center gap-2 rounded-md bg-slate-700 px-3 py-1.5 text-sm font-medium text-white shadow-sm hover:bg-slate-800 focus:outline-none focus:ring-2 focus:ring-slate-600 transition"
                  >
                    <svg className="h-4 w-4" viewBox="0 0 20 20" fill="currentColor">
                      <path d="M5 3a1 1 0 00-1 1v4a3 3 0 003 3h2v2a1 1 0 102 0v-2h2a3 3 0 003-3V4a1 1 0 10-2 0v4a1 1 0 01-1 1h-4a1 1 0 01-1-1V4a1 1 0 00-1-1H5z" />
                    </svg>
                    Merge
                  </button>
                </div>
              </div>
            </div>
          </div>
        ))}
        {caps.length === 0 && !loading && (
          <div className="rounded-lg border border-dashed border-slate-300 bg-slate-50 p-6 text-center text-slate-500">No capabilities yet. Click “Add Capability” to get started.</div>
        )}
      </div>
    </div>
  );
}
