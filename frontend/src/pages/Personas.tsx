import { useEffect, useState, Dispatch, SetStateAction } from "react";
import { useNavigate } from 'react-router-dom';
import PersonaCard from "../components/PersonaCard";
import PersonaBuilder from "../components/PersonaBuilder";
import * as Api from "../api";
import SpinnerIcon from "../utils/SpinnerIcon";

type Props = { token: string | undefined, progress: number, setPersonaTaskID: Dispatch<SetStateAction<string | null>>, suggestedPersonas: any[] }
const Personas = ({ token, progress, setPersonaTaskID, suggestedPersonas }: Props) => {
  const [products, setProducts] = useState<{ id: string; name: string }[]>([]);
  const [selectedProductId, setSelectedProductId] = useState<string | null>(null);
  const [personas, setPersonas] = useState<any[]>([]);
  const [showBuilder, setShowBuilder] = useState(progress < 100 && progress > 0);
  const [statusMsg, setStatusMsg] = useState("");
  const navigate = useNavigate();

  useEffect(() => {

    if (!token) {
      navigate("/login");
      return;
    }

    const fetchCompanyAndProducts = async () => {
      try {
        // 1. Get company ID on mount
	const { companyID } = await Api.me(token);

        // 2. Get product IDs for this company
	const { products } = await Api.products(token, companyID);
        if (products.length === 0) {
          setStatusMsg("No products found. Please run Value Prop first.");
          return;
        }
        console.log("✅ Product IDs set:", products.map((p: any) => p.id));
        setProducts(products);

        // 3. If only one product, select it automatically
        if (products.length === 1) {
          console.log("✅ Automatically selecting single product:", products[0].id);
          setSelectedProductId(products[0].id);
        }

	console.log("🔄 Working with product:", selectedProductId);
	if (!selectedProductId) {
	  console.log("❌ No product selected, skipping persona fetch.");
	  return;
	}

        console.log("🔄 Fetching personas for product:", selectedProductId);
	let personas;
	try {
	  personas = await Api.personas(token, selectedProductId);
          // 6. If no personas, prompt to infer
          if (personas.length === 0 && progress === 0) {
            setStatusMsg("No personas found. Click 'Infer Personas' to generate.");
          } else {
            setStatusMsg("");
            setPersonas(personas);
	  }

	} catch(e: any) {
	  if (e.message === "No Summaries or Capabilities Mapped") {
            setStatusMsg("No summaries or capabilities mapped. Please run Value Prop first.");
            setPersonas([]);
            return;
	  } else {
            setStatusMsg("Error fetching personas.");
            console.error("Error fetching personas:", e);
	  }
	}
        console.log("Backend personas response:", personas);
      } catch (e: any) {
        setStatusMsg(`Error: ${e.message}`);
      }
    };
    fetchCompanyAndProducts();
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
    
      {!showBuilder && selectedProductId && !statusMsg.includes("Value Prop") && (
        <button
          className="my-8 px-4 py-2 bg-indigo-600 text-white rounded hover:bg-indigo-700"
          onClick={() => setShowBuilder(true)}
        >
          {personas.length === 0 ? "Infer Personas" : "Re-infer Personas"}
        </button>
      )}

      {progress > 0 && progress < 100 && (
        <div className="mt-4 bg-blue-50 border border-blue-200 rounded-lg p-4">
          <div className="flex items-center space-x-3">
            <SpinnerIcon />
            <div className="flex-1">
              <p className="text-blue-800 font-medium">Analyzing personas...</p>
              {progress > 0 && (
                <div className="mt-2">
                  <div className="bg-blue-200 rounded-full h-2">
                    <div 
                      className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                      style={{ width: `${progress}%` }}
                    ></div>
                  </div>
                  <p className="text-blue-600 text-xs mt-1">{progress}% complete</p>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {showBuilder && selectedProductId && (
        <div className="mt-8">
          <PersonaBuilder
            product_id={selectedProductId}
	    token={token!!}
	    progress={progress}
	    setTaskID={setPersonaTaskID}
            onSave={() => {
              setShowBuilder(false);
              setStatusMsg("Personas inferred! Refresh to see new personas.");
            }}
          />
        </div>
      )}
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
	      const json = await res.json();
	      setPersonaTaskID(json.task_id);
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

    </div>
  );
};

export default Personas;
