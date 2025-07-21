import { useEffect, useState } from "react";
import PersonaCard from "../components/PersonaCard";
import PersonaBuilder from "../components/PersonaBuilder";

const Personas = () => {
  const [companyId, setCompanyId] = useState<string | null>(null);
  const [products, setProducts] = useState<{ id: string; name: string }[]>([]);
  const [selectedProductId, setSelectedProductId] = useState<string | null>(null);
  const [personas, setPersonas] = useState([]);
  const [showBuilder, setShowBuilder] = useState(false);
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
        console.log("🔄 Fetching personas for product:", selectedProductId);
        const personaRes = await fetch(
          `http://localhost:8000/get-personas/${selectedProductId}`,
          {
            headers: { Authorization: `Bearer ${token}` },
          }
        );
        const personaData = await personaRes.json();
        console.log("Backend personas response:", personaData);

        // 5. Handle backend messages
        if (personaData?.detail === "No Summaries or Capabilities Mapped") {
          setStatusMsg("No summaries or capabilities mapped. Please run Value Prop first.");
          setPersonas([]);
          return;
        }

        setStatusMsg("");
        setPersonas(personaData || []);

        // 6. If no personas, prompt to infer
        if (!personaData || personaData.length === 0) {
          setStatusMsg("No personas found. Click 'Infer Personas' to generate.");
        }
      } catch (err) {
        setStatusMsg("Error fetching personas.");
        console.error("Error fetching personas:", err);
      }
    };
    fetchPersonas();
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
    
      {/* 5. Show personas if they exist */}
      {personas.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {personas.map((p, i) => (
            p && p.persona_title ? (
              <PersonaCard
                key={i}
                persona={p}
                isSelected={true}
                onToggle={() => {}}
              />
            ) : null
          ))}
        </div>
      )}

     {personas.length > 0 && (
        <button
          className="mt-6 px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700"
          onClick={async () => {
            if (!selectedProductId) return;
            setStatusMsg("Inferring Hop+ Personas...");
            try {
              const res = await fetch("http://localhost:8000/analyze/hop_plus", {
                method: "POST",
                headers: {
                  "Content-Type": "application/json",
                  Authorization: `Bearer ${token}`,
                },
                body: JSON.stringify({ product_id: selectedProductId }),
              });
              if (!res.ok) throw new Error("Failed to infer Hop+ personas");
              setStatusMsg("Hop+ Personas inferred! Refresh to see updates.");
              // Optionally, refresh personas here by calling fetchPersonas()
            } catch (err) {
              setStatusMsg("Error inferring Hop+ personas.");
              console.error(err);
            }
          }}
        >
          Infer Hop+ Personas
        </button>
      )}

      {/* 7. Show Infer Personas button if no personas and product is selected */}
      {!showBuilder && selectedProductId && !statusMsg.includes("Value Prop") && (
        <button
          className="mt-8 px-4 py-2 bg-indigo-600 text-white rounded hover:bg-indigo-700"
          onClick={() => setShowBuilder(true)}
        >
          {personas.length === 0 ? "Infer Personas" : "Re-infer Personas"}
        </button>
      )}
      

      {/* 8. Show PersonaBuilder when button is clicked */}
      {showBuilder && selectedProductId && (
        <div className="mt-8">
          <PersonaBuilder
            product_id={selectedProductId}
            onSave={() => {
              setShowBuilder(false);
              setStatusMsg("Personas inferred! Refresh to see new personas.");
            }}
          />
        </div>
      )}
    </div>
  );
};

export default Personas;