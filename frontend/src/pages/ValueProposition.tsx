import { useEffect, useState } from "react";
import ValuePropCard from "../components/ValuePropCard";
import CapabilityCard from "../components/CapabilityCard";
import CapabilityEditor from "../components/CapabilityEditor";
import PainFamilyEditor from "../components/PainFamilyEditor";
import PersonaJobAnchor from "../components/PersonaJobAnchor";

type Capability = {
  coreness: number;
  centrality: number;
  name: string;
  description?: string;
};

type ProductData = {
  product_node_id: string;
  summary: string;
  plg_flag?: boolean;
  capabilities: Capability[];
};

const ValueProposition = () => {
  const [companyId, setCompanyId] = useState<string | null>(null);
  const [products, setProducts] = useState<{ id: string; name: string }[]>([]);
  const [selectedProductId, setSelectedProductId] = useState<string | null>(null);
  const [productData, setProductData] = useState<ProductData | null>(null);
  const [statusMsg, setStatusMsg] = useState("");
  const token = localStorage.getItem("token");

  // Fetch company and products on mount
  useEffect(() => {
    const fetchCompanyAndProducts = async () => {
      try {
        const meRes = await fetch("http://localhost:8000/me", {
          headers: { Authorization: `Bearer ${token}` },
        });
        const meData = await meRes.json();
        setCompanyId(meData.company_id);

        const prodRes = await fetch(
          `http://localhost:8000/get-products/${meData.company_id}`,
          { headers: { Authorization: `Bearer ${token}` } }
        );
        const prodData = await prodRes.json();
        console.log("Backend Products data:", prodData);
        if (!prodData.products || prodData.products.length === 0) {
          setStatusMsg("No products found. Please run Value Prop first.");
          return;
        }
        setProducts(prodData.products);

        console.log("Products fetched length:", prodData.products.length);

        // Auto-select if only one product
        if (prodData.products.length === 1) {
          setSelectedProductId(prodData.products[0].id);
          console.log("Auto selection of product ID")
        }
      } catch (err) {
        setStatusMsg("Error fetching company or products.");
      }
    };
    fetchCompanyAndProducts();
  }, [token]);

  // Fetch product capabilities when a product is selected
  useEffect(() => {
    if (!selectedProductId) return;
    const fetchProductData = async () => {
      setStatusMsg("Loading product data...");
      try {
        const res = await fetch(
          `http://localhost:8000/get-product-capabilities/${selectedProductId}`,
          { headers: { Authorization: `Bearer ${token}` } }
        );
        const data = await res.json();
        console.log("Backend Product data:", data);
        if (!data || !data.product_node_id) {
          setProductData(null);
          setStatusMsg("No product data found.");
          return;
        }
        setProductData(data);
        setStatusMsg("");
      } catch (err) {
        setStatusMsg("Error fetching product data.");
        setProductData(null);
      }
    };
    fetchProductData();
  }, [selectedProductId, token]);

  return (
    <div className="p-8">
      <h2 className="text-2xl font-bold mb-6">Product Overview</h2>

      {/* Product selector */}
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

      {/* Status message */}
      {statusMsg && <div className="mb-4 text-red-600">{statusMsg}</div>}

      {/* ValuePropCard and Capabilities */}
      {productData && (
        <div className="mb-8">
          <ValuePropCard
            product_node_id={productData.product_node_id}
            summary={productData.summary}
            plg_flag={productData.plg_flag}
          />
          <div className="mt-6">
            <h4 className="font-semibold text-indigo-600 mb-2">Capabilities</h4>
            {productData.capabilities && productData.capabilities.length > 0 ? (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {productData.capabilities.map((cap, idx) => (
                  <CapabilityCard key={idx} name={cap.name} description={cap.description} coreness={cap.coreness} centrality={cap.centrality} />
                ))}
              </div>
            ) : (
              <div className="text-gray-400">No capabilities found.</div>
            )}
          </div>

          {/* Editor */}
          {selectedProductId && (
            <CapabilityEditor productId={selectedProductId} />
          )}
          {selectedProductId && (
            <PainFamilyEditor productId={selectedProductId} />
          )}
          {selectedProductId && (
            <PersonaJobAnchor productId={selectedProductId} />
          )}
        </div>
      )}
    </div>
  );
};

export default ValueProposition;
