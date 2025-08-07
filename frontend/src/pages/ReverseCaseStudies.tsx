import { useEffect, useState } from "react";


const ReverseCaseStudies = () => {
  const [companyId, setCompanyId] = useState<string | null>(null);
  const [products, setProducts] = useState<{ id: string; name: string }[]>([]);
  const [selectedProductId, setSelectedProductId] = useState<string | null>(null);
  const [reversecasestudy, setReverseCaseStudy] = useState([]);
  const [statusMsg, setStatusMsg] = useState("");
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

  // 4. When a product is selected, fetch personas
  useEffect(() => {
    console.log("🔄 Working with product:", selectedProductId);
    if (!selectedProductId) {
      console.log("❌ No product selected, skipping persona fetch.");
      return;
    }
    const fetchPersonas = async () => {
      try {
        console.log("🔄 Fetching reverse case studies for product:", selectedProductId);
        const reverseCaseStudyRes = await fetch(
          `http://localhost:8000/get-reverse-case-studies/${selectedProductId}`,
          {
            headers: { Authorization: `Bearer ${token}` },
          }
        );
        const reverseCaseStudyData = await reverseCaseStudyRes.json();
        console.log("Backend reverse case studies response:", reverseCaseStudyData);

        // 5. Handle backend messages
        if (reverseCaseStudyData?.detail === "No Summaries or Capabilities Mapped") {
          setStatusMsg("No summaries or capabilities mapped. Please run Value Prop first.");
          setReverseCaseStudy([]);
          return;
        }

        setStatusMsg("");
        setReverseCaseStudy(reverseCaseStudyData || []);

        // 6. If no reverse case studies, prompt to infer
        if (!reverseCaseStudyData || reverseCaseStudyData.length === 0) {
          setStatusMsg("No reverse case studies found..");
        }
      } catch (err) {
        setStatusMsg("Error fetching reverse case studies.");
        console.error("Error fetching reverse case studies:", err);
      }
    };
    fetchPersonas();
  }, [selectedProductId]);

  return (
    <div className="p-8">
      
      {/* 3. If multiple products, let user select */}
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

      {/* 4. If no products, show message */}
      {statusMsg && <div className="mb-4 text-red-600">{statusMsg}</div>}

      {/* 5. Show reverse case studies if they exist */}
      {reversecasestudy.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          Reverse Case Study Data in Terminal
        </div>
      )}
    </div>
  );
};

export default ReverseCaseStudies;