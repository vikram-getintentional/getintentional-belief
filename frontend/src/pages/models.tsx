import { useEffect, useState } from "react";
import CRMWinModels from "../components/model_ui/crm_win_models";

const Models = () => {
  const [companyId, setCompanyId] = useState<string | null>(null);
  const [products, setProducts] = useState<{ id: string; name: string }[]>([]);
  const [selectedProductId, setSelectedProductId] = useState<string | null>(null);
  const [statusMsg, setStatusMsg] = useState("");
  const [rebuildStatus, setRebuildStatus] = useState<string>("");
  const token = localStorage.getItem("token");

  // 1. Get company ID on mount
  useEffect(() => {
    const fetchCompanyAndProducts = async () => {
      try {
        const meRes = await fetch("http://localhost:8000/me", {
          headers: { Authorization: `Bearer ${token}` },
        });
        const meData = await meRes.json();
        setCompanyId(meData.company_id);

        // 2. Get product IDs for this company
        const prodRes = await fetch(
          `http://localhost:8000/get-products/${meData.company_id}`,
          {
            headers: { Authorization: `Bearer ${token}` },
          }
        );
        const prodData = await prodRes.json();
        if (!prodData.products || prodData.products.length === 0) {
          setStatusMsg("No products found. Please run Value Prop first.");
          return;
        }
        console.log("✅ Product IDs set:", prodData.products.map((p: any) => p.id));
        setProducts(prodData.products);

        // 3. If only one product, select it automatically
        if (prodData.products.length === 1) {
          console.log("✅ Automatically selecting single product:", prodData.products[0].id);
          setSelectedProductId(prodData.products[0].id);
        }
      } catch (err) {
        setStatusMsg("Error fetching company or products.");
      }
    };
    fetchCompanyAndProducts();
  }, [token]);

  const triggerGlobalThesisRebuild = async () => {
    if (!token) {
      setRebuildStatus("Missing auth token.");
      return;
    }
    if (!selectedProductId) {
      setRebuildStatus("Select a product first.");
      return;
    }
    try {
      setRebuildStatus("Rebuilding global thesis…");
      const res = await fetch(
        `http://localhost:8000/journey/rebuild-global-thesis/${selectedProductId}`,
        {
          method: "POST",
          headers: {
            Authorization: `Bearer ${token}`,
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ window: 1000 }),
        }
      );
      if (!res.ok) {
        const txt = await res.text();
        throw new Error(txt || "Failed to rebuild thesis");
      }
      const data = await res.json();
      const episodes = data?.thesis?.meta?.num_episodes ?? "?";
      setRebuildStatus(`Global thesis rebuilt (episodes analyzed: ${episodes}).`);
    } catch (err: any) {
      setRebuildStatus(err?.message || "Global thesis rebuild failed.");
    }
  };

  return (
    <div className="p-8">
      {products.length > 1 && (
        <div className="mb-4">
          <label className="mr-2 font-semibold">Select Product:</label>
          <select
            value={selectedProductId || ""}
            onChange={e => setSelectedProductId(e.target.value)}
            className="border rounded px-2 py-1"
          >
            <option value="" disabled>Select a product</option>
            {products.map(p => (
              <option key={p.id} value={p.id}>{p.name}</option>
            ))}
          </select>
        </div>
      )}
      {statusMsg && <div className="mb-4 text-red-600">{statusMsg}</div>}
      <div className="mb-6 flex items-center gap-3">
        <button
          className="bg-indigo-600 text-white px-3 py-2 rounded disabled:opacity-50"
          onClick={triggerGlobalThesisRebuild}
          disabled={!selectedProductId}
        >
          Rebuild Journey Thesis
        </button>
        {rebuildStatus && (
          <span className="text-sm text-slate-600">{rebuildStatus}</span>
        )}
      </div>
      <CRMWinModels productId={selectedProductId ?? ""} token={token ?? ""} />

    </div>
  );
};

export default Models;
