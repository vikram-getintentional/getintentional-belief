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
    } catch (e) {
      setError("Failed to save capability");
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
    <div className="mt-10">
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-lg font-semibold text-indigo-700">Capability Editor</h3>
        <button onClick={addCapability} className="px-3 py-1 rounded bg-indigo-600 text-white text-sm">+ Add Capability</button>
      </div>
      {loading && <div className="text-gray-500">Loading…</div>}
      {error && <div className="text-red-600 mb-2">{error}</div>}
      <div className="space-y-3">
        {caps.map((cap) => (
          <div key={cap.id} className="p-3 border rounded bg-white">
            <div className="text-xs text-gray-400 mb-1">{cap.id}</div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              <div>
                <label className="block text-sm font-medium">Name</label>
                <input className="w-full border rounded px-2 py-1" value={cap.name} onChange={(e) => updateField(cap.id, "name", e.target.value)} />
              </div>
              <div>
                <label className="block text-sm font-medium">Coreness</label>
                <select className="w-full border rounded px-2 py-1" value={cap.coreness} onChange={(e) => updateField(cap.id, "coreness", e.target.value)}>
                  <option value="">Select…</option>
                  {CORENESS_OPTIONS.map((o) => (
                    <option key={o} value={o}>{o}</option>
                  ))}
                </select>
              </div>
              <div className="md:col-span-2">
                <label className="block text-sm font-medium">Description</label>
                <textarea className="w-full border rounded px-2 py-1" rows={2} value={cap.description} onChange={(e) => updateField(cap.id, "description", e.target.value)} />
              </div>
              <div>
                <label className="block text-sm font-medium">Buying Likelihood: {cap.buyingLikelihood}</label>
                <input type="range" min={0} max={100} value={cap.buyingLikelihood} onChange={(e) => updateField(cap.id, "buyingLikelihood", Number(e.target.value))} />
              </div>
              <div className="flex items-end gap-2">
                <button onClick={() => saveCapability(cap)} className="px-3 py-1 rounded bg-green-600 text-white text-sm">Save</button>
                <button onClick={() => deleteCapability(cap.id)} className="px-3 py-1 rounded bg-red-600 text-white text-sm">Delete</button>
              </div>
              <div className="flex items-end gap-2">
                <div className="flex-1">
                  <label className="block text-sm font-medium">Merge into</label>
                  <select className="w-full border rounded px-2 py-1"
                          value={mergeTarget[cap.id] || ""}
                          onChange={(e) => setMergeTarget((m) => ({ ...m, [cap.id]: e.target.value }))}>
                    <option value="">Select target…</option>
                    {caps.filter((c) => c.id !== cap.id).map((c) => (
                      <option key={c.id} value={c.id}>{c.name || c.id}</option>
                    ))}
                  </select>
                </div>
                <button onClick={() => mergeCapability(cap.id)} className="px-3 py-1 rounded bg-gray-700 text-white text-sm">Merge</button>
              </div>
            </div>
          </div>
        ))}
        {caps.length === 0 && !loading && <div className="text-gray-400">No capabilities yet. Add one to get started.</div>}
      </div>
    </div>
  );
}
