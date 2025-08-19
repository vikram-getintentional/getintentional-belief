import { useEffect, useState } from "react";
import ZmotCard from "../components/ZmotCard";
import ZmotIcpBuilder from "../components/ZmotICPBuilder";
import IcpCard from "../components/IcpCard";

const ZmotIcp = () => {
  const [companyId, setCompanyId] = useState<string | null>(null);
  const [products, setProducts] = useState<{ id: string; name: string }[]>([]);
  const [selectedProductId, setSelectedProductId] = useState<string | null>(null);
  // Define a type for ZMOT ICP items (adjust fields as needed)
  type ZmotIcpItem = {
    icp?: any;
    zmot?: any;
  };
  const [zmotIcp, setZmotIcp] = useState<ZmotIcpItem[]>([]);
  const [showBuilder, setShowBuilder] = useState(false);
  const [statusMsg, setStatusMsg] = useState("");
  const token = localStorage.getItem("token");
  const [archetypeVisibleCount, setArchetypeVisibleCount] = useState(5);
  const [zmotVisibleCount, setZmotVisibleCount] = useState(5);

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
      console.log("❌ No product selected, skipping ZMOT fetch.");
      return;
    }
    const fetchZmotsIcps = async () => {
      try {
        console.log("🔄 Fetching ZMOTs and ICPs for product:", selectedProductId);
        const zmotIcpRes = await fetch(
          `http://localhost:8000/get-zmot-icp/${selectedProductId}`,
          {
            headers: { Authorization: `Bearer ${token}` },
          }
        );
        const zmotIcpData = await zmotIcpRes.json();
        console.log("Backend ZMOTs and ICPs response:", zmotIcpData);

        // 6. If no ICPs, prompt to infer
        if (!zmotIcpData || (Array.isArray(zmotIcpData) && zmotIcpData.length === 0)) {
          setZmotIcp([]);
          setStatusMsg("No ICPs or ZMOTs found. Click 'Infer ICPs' to generate.");
          return;
        }

        // 5. Handle backend messages
        if (zmotIcpData?.detail === "No pains found for this product") {
          setStatusMsg("No pains mapped yet. Please run Personas first.");
          setZmotIcp([]);
          return;
        }

        setStatusMsg("");
        console.log("zmotIcpData in setting block:", zmotIcpData);
        if (Array.isArray(zmotIcpData)) {
          setZmotIcp(zmotIcpData);
        } else if (zmotIcpData && typeof zmotIcpData === "object") {
          setZmotIcp([zmotIcpData]);
        } else {
          setZmotIcp(zmotIcpData);
        }
        
        
      } catch (err) {
        setStatusMsg("Error fetching ZMOTs and ICPs.");
        console.error("Error fetching ZMOTs and ICPs:", err);
      }
    };
    fetchZmotsIcps();
  }, [selectedProductId]);

  return (
    <div className="p-8">
      <h2 className="text-2xl font-bold mb-6">Saved Personas</h2>

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
    
      {/* 5. Show ZMOTs if they exist */}
      {zmotIcp.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {zmotIcp.map((z, i) => (
            z && z.title ? (
              <ZmotCard
                key={i}
                zmot={z}
                isSelected={true}
                onToggle={() => {}}
              />
            ) : null
          ))}
        </div>
      )}

     {/* ICP Cards Section */}
      {zmotIcp.length > 0 && Array.isArray(zmotIcp[0].icp) && zmotIcp[0].icp.length > 0 && (
        <section className="mb-10">
          <h2 className="text-2xl font-bold mb-4">ICP Archetypes</h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6">
            {zmotIcp[0].icp
              .slice(0, archetypeVisibleCount)
              .map((icpItem, idx) => (
                <IcpCard key={idx} icp={icpItem} />
              ))}
          </div>
          {archetypeVisibleCount < zmotIcp[0].icp.length && (
            <button
              className="mt-4 px-4 py-2 bg-gray-200 rounded"
              onClick={() => setArchetypeVisibleCount(archetypeVisibleCount + 5)}
            >
              Show More Archetypes
            </button>
          )}
        </section>
      )}

      {/* ZMOT Cards Section */}
      {zmotIcp.length > 0 && zmotIcp[0].zmot && (
        <section>
          <h2 className="text-2xl font-bold mb-4">ZMOTs & Triggers</h2>
          {Array.isArray(zmotIcp[0].zmot) && zmotIcp[0].zmot.length > 0 ? (
            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6">
              {zmotIcp[0].zmot
                .slice(0, zmotVisibleCount)
                .map((zmotItem, idx) => (
                  <ZmotCard key={idx} zmot={zmotItem} />
                ))}
            </div>
          ) : (
            <div className="text-gray-400">No ZMOTs found.</div>
          )}
          {zmotVisibleCount < zmotIcp[0].zmot.length && (
            <button
              className="mt-4 px-4 py-2 bg-gray-200 rounded"
              onClick={() => setZmotVisibleCount(zmotVisibleCount + 5)}
            >
              Show More ZMOTs
            </button>
          )}
        </section>
      )}

      {/* 7. Show Infer ZMOT button if no ZMOTs are available */}
      {!showBuilder && selectedProductId && !statusMsg.includes("Inferring") && (
        <button
          className="mt-8 px-4 py-2 bg-indigo-600 text-white rounded hover:bg-indigo-700"
          onClick={() => setShowBuilder(true)}
        >
          {zmotIcp.length === 0 ? "Infer ZMOTs" : "Re-infer ZMOTs"}
        </button>
      )}

      {/* 8. Show ZmotIcpBuilder when button is clicked */}
      {showBuilder && selectedProductId && (
        <div className="mt-8">
          <ZmotIcpBuilder
            productId={selectedProductId}
            onSave={() => {
              setShowBuilder(false);
              setStatusMsg("ZMOTs inferred! Refresh to see new ZMOTs.");
            }}
          />
        </div>
      )}

      
    </div>
  );
};

export default ZmotIcp;