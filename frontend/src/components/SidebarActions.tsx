// components/SidebarActions.tsx
import { useEffect, useMemo, useState } from "react";
import ReactDOM from "react-dom";

type Product = { id: string; name: string };

type ConfidenceReport = {
  graph_confidence: number;
  nodes_confidence: number;
  edges_confidence: number;
  coverage: { nodes_with_meta: number; edges_with_meta: number };
  staleness_days_avg: { nodes: number; edges: number };
  as_of_iso: string;
};

type SidebarActionsProps = {
  /** optional; defaults to localStorage.getItem('token') */
  token?: string | null;
  /** endpoint that returns { company_id } */
  meUrl?: string;
  /** endpoint that returns { products: Product[] } */
  productsUrl?: (companyId: string) => string;
  /** POST endpoint to build/update the graph */
  buildGraphUrl?: string;
  /** optional: called after successful build */
  onGraphUpdated?: () => void;
  /** button label override */
  buildButtonLabel?: string;
  /** className passthrough for styling in your sidebar */
  className?: string;
  /** endpoint returning graph confidence */
  confidenceUrl?: (productId: string) => string;
};

export default function SidebarActions({
  token = typeof window !== "undefined" ? localStorage.getItem("token") : null,
  meUrl = "http://localhost:8000/me",
  productsUrl = (cid) => `http://localhost:8000/get-products/${cid}`,
  buildGraphUrl = "http://localhost:8000/analyze/deep",
  confidenceUrl = (pid) => `http://localhost:8000/graph/confidence/${pid}`,
  onGraphUpdated,
  buildButtonLabel = "Generate Graph",
  className = "",
}: SidebarActionsProps) {
  const [products, setProducts] = useState<Product[]>([]);
  const [selectedProductId, setSelectedProductId] = useState<string>("");
  const [msg, setMsg] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [conf, setConf] = useState<ConfidenceReport | null>(null);
  const [confLoading, setConfLoading] = useState(false);
  const [showDetails, setShowDetails] = useState(false);

  const authHeader = useMemo(() => {
    if (!token) return undefined;
    return { Authorization: `Bearer ${token}` };
  }, [token]);

  // fetch company + products once
  useEffect(() => {
    (async () => {
      try {
        if (!token) return;
        const me = await fetch(meUrl, { headers: { ...authHeader } }).then((r) => r.json());
        if (!me?.company_id) return;
        const prodResp = await fetch(productsUrl(me.company_id), {
          headers: { ...authHeader },
        }).then((r) => r.json());
        if (Array.isArray(prodResp?.products) && prodResp.products.length) {
          setProducts(prodResp.products);
          setSelectedProductId(prodResp.products[0].id);
        }
      } catch (e) {
        // optional: setMsg("Error loading products");
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [token]);

  const fetchConfidence = async (pid: string) => {
    if (!pid || !token) return;
    try {
      setConfLoading(true);
      const res = await fetch(confidenceUrl(pid), { headers: { ...authHeader } });
      if (!res.ok) throw new Error("confidence fetch failed");
      const data: ConfidenceReport = await res.json();
      setConf(data);
    } catch {
      setConf(null);
    } finally {
      setConfLoading(false);
    }
  };

  // fetch confidence when product changes, and when graph updates
  useEffect(() => {
    if (selectedProductId) fetchConfidence(selectedProductId);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedProductId, token]);

  useEffect(() => {
    const onUpdated = () => {
      fetchConfidence(selectedProductId);
      onGraphUpdated?.();
    };
    window.addEventListener("graph-updated", onUpdated);
    return () => window.removeEventListener("graph-updated", onUpdated);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedProductId, token]);

  const generateGraph = async () => {
    if (!selectedProductId || !token) return;
    setLoading(true);
    setMsg("Building graph…");
    try {
      const res = await fetch(buildGraphUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...authHeader },
        body: JSON.stringify({ product_id: selectedProductId }),
      });
      if (!res.ok) throw new Error("build failed");
      setMsg("Graph built! Tabs will refresh.");
      // emit an app-wide event so any page can refetch
      window.dispatchEvent(new Event("graph-updated"));
      onGraphUpdated?.();
    } catch (e) {
      setMsg("Error building graph.");
    } finally {
      setLoading(false);
      setTimeout(() => setMsg(""), 3000);
    }
  };

  return (
    <div className={`mt-auto ${className}`}>
      {/* Only show dropdown if no product is selected */}
      {!selectedProductId ? (
        <div className="space-y-2">
          {products.length > 0 ? (
            <select
              value={selectedProductId}
              onChange={(e) => setSelectedProductId(e.target.value)}
              className="w-full border rounded px-2 py-1 text-sm"
            >
              <option value="">Select a product…</option>
              {products.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.name}
                </option>
              ))}
            </select>
          ) : (
            <div className="text-xs text-gray-500">No products found</div>
          )}
        </div>
      ) : (
        <>
         {/* Confidence Score */}
         <div
            className="rounded-lg border p-3 mb-3 cursor-pointer hover:bg-gray-100 transition"
            onClick={() => {
              setShowDetails(true);
              console.log("Clicked confidence details");
            }}
            title="Click for details"
          >
            <div className="flex items-center justify-between">
              <div className="text-xs font-medium text-gray-700">Confidence</div>
              <div className="text-lg font-bold text-indigo-700">
                {confLoading ? "…" : conf ? `${Math.round(conf.graph_confidence * 100)}%` : "—"}
              </div>
            </div>
          </div>
          {/* Confidence Modal */}
          {showDetails &&
            ReactDOM.createPortal(
              (
                <div
                  className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-30"
                  onClick={() => setShowDetails(false)}
                >
                  <div
                    className="bg-white rounded-lg shadow-lg p-6 min-w-[320px] max-w-[90vw]"
                    onClick={e => e.stopPropagation()}
                  >
                    <div className="flex justify-between items-center mb-2">
                      <div className="text-lg font-semibold">Graph Confidence Details</div>
                      <button
                        className="text-gray-400 hover:text-gray-700 text-xl"
                        onClick={() => setShowDetails(false)}
                      >
                        ×
                      </button>
                    </div>
                    <div className="mb-2">
                      <div>Overall: {conf ? Math.round(conf.graph_confidence * 100) : "—"}%</div>
                      <div>Nodes: {conf ? Math.round(conf.nodes_confidence * 100) : "—"}%</div>
                      <div>Edges: {conf ? Math.round(conf.edges_confidence * 100) : "—"}%</div>
                      <div>
                        Node Meta: {conf ? Math.round(conf.coverage.nodes_with_meta * 100) : "—"}%<br />
                        Edge Meta: {conf ? Math.round(conf.coverage.edges_with_meta * 100) : "—"}%
                      </div>
                      <div>
                        Node Age: {conf?.staleness_days_avg?.nodes ?? "—"}d<br />
                        Edge Age: {conf?.staleness_days_avg?.edges ?? "—"}d
                      </div>
                      <div className="mt-1 text-[10px] text-gray-400">
                        As of: {conf?.as_of_iso ? new Date(conf.as_of_iso).toLocaleString() : "—"}
                      </div>
                    </div>
                  </div>
                </div>
              ),
              document.body
            )
          }
          {/* Generate Graph Button */}
          <button
            onClick={generateGraph}
            className={`w-full px-3 py-2 rounded text-sm text-white ${
              loading ? "bg-indigo-400" : "bg-indigo-600 hover:bg-indigo-700"
            }`}
            disabled={!selectedProductId || loading}
            title="Rebuilds the belief graph and updates confidence"
          >
            {loading ? "Working…" : buildButtonLabel}
          </button>
          {msg && <div className="text-xs text-gray-600">{msg}</div>}
        </>
      )}
    </div>
  );
}
