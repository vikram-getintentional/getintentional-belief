import { useEffect, useState } from "react";
import RCSConfigurator from "../components/RCSConfigurator";
import RCSTabs from "../components/RCSTabs";

const ReverseCaseStudies = () => {
  const [companyId, setCompanyId] = useState<string | null>(null);
  const [products, setProducts] = useState<{ id: string; name: string }[]>([]);
  const [selectedProductId, setSelectedProductId] = useState<string | null>(null);
  const [rcsData, setRcsData] = useState<any>(null);
  const token = localStorage.getItem("token");
  const [rcs, setRcs] = useState<any | null>(null);
  const [rcsGraph, setRcsGraph] = useState<any | null>(null);

  const handleRcsFetched = (data: any) => {
    // keep the full payload so normalizers/components can prefer frozen_strategy etc.
    console.log("RCS fetched (full payload):", data);
    setRcs(data);
    // graph may live in a few places depending on backend version
    setRcsGraph(data.graph || data.graph_for_ui || data?.output?.graph || null);
  };

  useEffect(() => {
    (async () => {
      const meRes = await fetch("http://localhost:8000/me", {
        headers: { Authorization: `Bearer ${token}` },
      });
      const meData = await meRes.json();
      setCompanyId(meData.company_id);

      const prodRes = await fetch(`http://localhost:8000/get-products/${meData.company_id}`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      const prodData = await prodRes.json();
      setProducts(prodData.products || []);
      if (prodData.products?.length === 1) setSelectedProductId(prodData.products[0].id);
    })();
  }, [token]);

  return (
    <div className="p-8">
      <h2 className="text-xl font-semibold mb-3">Reverse Case Studies</h2>

      {/* product dropdown */}
      {products.length > 1 && (
        <select
          className="border rounded px-2 py-1 mb-3"
          value={selectedProductId || ""}
          onChange={e => setSelectedProductId(e.target.value)}
        >
          <option value="">Select a product</option>
          {products.map(p => <option key={p.id} value={p.id}>{p.name}</option>)}
        </select>
      )}

      {/* ICP + engaged picker */}
      <RCSConfigurator
        selectedProductId={selectedProductId ?? ""}
        token={token ?? ""}
        onRcsFetched={handleRcsFetched}
      />

      {/* Render tabs only when data exists */}
      <RCSTabs rcs={rcs ?? null} rcsGraph={rcsGraph ?? null} />
    </div>
  );
};

export default ReverseCaseStudies;
