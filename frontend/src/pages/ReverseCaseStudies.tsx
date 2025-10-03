import { useEffect, useState } from "react";
import RCSArchetypePicker from "../components/RCSArchetypePicker";
import ConversionLikelihood from "../components/ConversionLikelihood";
import ReverseCaseStudyFlow from "../components/ReverseCaseStudyFlow";
import PersonaCard from "../components/PersonaCard";

const ReverseCaseStudies = () => {
  const [companyId, setCompanyId] = useState<string | null>(null);
  const [products, setProducts] = useState<{ id: string; name: string }[]>([]);
  const [selectedProductId, setSelectedProductId] = useState<string | null>(null);
  const [statusMsg, setStatusMsg] = useState("");
  const [rcsData, setRcsData] = useState<any>(null);
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

  // Example: fetch RCS data when product changes
  useEffect(() => {
    if (!selectedProductId) return;
    (async () => {
      const res = await fetch(`http://localhost:8000/get-reverse-case-study/${selectedProductId}`, {
        method: "POST",
        headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
        body: JSON.stringify({}),
      });
      if (res.ok) {
        const data = await res.json();
        setRcsData(data);
        console.log("✅ Fetched RCS data for product:", data);
      }
    })();
  }, [selectedProductId, token]);

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
      <RCSArchetypePicker selectedProductId={selectedProductId ?? ""} token={token ?? ""} />
      {rcsData && (
        <div className="mt-8">
          <ConversionLikelihood likelihood={rcsData.graph_win_likelihood ?? 0} />
      
        </div>
      )}
    </div>
  );
};

export default ReverseCaseStudies;