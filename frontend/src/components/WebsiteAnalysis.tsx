import { useEffect, useState } from "react";
import { useLocation } from "react-router-dom";

type Props = {
  url: string;
  scrapedText: string;
  plgCta?: boolean;
  footerFeatures?: string[];
};

type Capability = {
  node_id?: string;
  name: string;
  description: string;
};

const WebsiteAnalysis = ({ scrapedText, url, plgCta = false, footerFeatures = [] }: Props) => {
  const [summary, setSummary] = useState("");
  const [capabilities, setCapabilities] = useState<Capability[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [editedSummary, setEditedSummary] = useState(false);
  const [editedCapabilities, setEditedCapabilities] = useState<Capability[]>([]);
  const [newCapabilities, setNewCapabilities] = useState<Capability[]>([]);
  const [companyId, setCompanyId] = useState<string | null>(null);
  const [productId, setProductId] = useState<string | null>(null);
  const [companyIdLoading, setCompanyIdLoading] = useState(true);
  const location = useLocation();

  useEffect(() => {
    // Save when the route changes
    return () => {
      handleSave();
    };
    // Only run when the location changes
  }, [location]);

  // Fetch company ID
  useEffect(() => {
    const token = localStorage.getItem("token");
    fetch("http://localhost:8000/me", {
      headers: { Authorization: `Bearer ${token}` },
    })
      .then((res) => res.json())
      .then((data) => {
        setCompanyId(data.company_id);
        setCompanyIdLoading(false);
        console.log("✅ Company ID set:", data.company_id);
      })
      .catch(() => {
        console.error("❌ Failed to fetch company ID.");
        setCompanyIdLoading(false);
      });
  }, []);

  // Analyze website content
  useEffect(() => {
    if (companyIdLoading || !scrapedText || !url || !companyId) return;

    setLoading(true);
    const token = localStorage.getItem("token");
    console.log("🚀 Sending to /analyze:", { companyId, scrapedText });
    fetch("http://localhost:8000/analyze", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify({
        url,
        company_id: companyId,
        text: scrapedText,
        plg_cta_found: plgCta,
        footer_features: footerFeatures,
      }),
    })
      .then((res) => res.json())
      .then((data) => {
        setProductId(data.product_node_id || "");
        setSummary(data.summary || "");
        setCapabilities(data.capabilities || []);
        setLoading(false);
      })
      .catch(() => {
        setError("Failed to analyze content.");
        setLoading(false);
      });
  }, [scrapedText, url, companyId, companyIdLoading]);

  // Handle summary change
  const handleSummaryChange = (value: string) => {
    setSummary(value);
    setEditedSummary(true);
    //debouncedSave();
  };

  // Handle capability change
  const handleCapabilityChange = (index: number, field: "name" | "description", value: string) => {
    const updated = [...capabilities];
    updated[index][field] = value;
    setCapabilities(updated);

    const edited = [...editedCapabilities];
    const capNodeId = updated[index].node_id;
    const existingIdx = edited.findIndex((cap) => cap.node_id === capNodeId);
    if (existingIdx !== -1) {
      edited[existingIdx] = { ...updated[index] };
    } else {
      edited.push({ ...updated[index] });
    }
    setEditedCapabilities(edited);

    //debouncedSave();
  };

  // Handle adding a new capability
  const handleAddCapability = () => {
    const newCapability = { name: "", description: "" };
    setCapabilities([...capabilities, newCapability]);
    setNewCapabilities([...newCapabilities, newCapability]);
  };

  // Save changes
  const handleSave = () => {
    const token = localStorage.getItem("token");

    if (editedSummary) {
      console.log("🚀 Sending summary update:", { summary });
      fetch("http://localhost:8000/save-summary", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          company_id: companyId,
          product_id: productId,
          summary,
        }),
      })
        .then((res) => res.json())
        .then((data) => {
          console.log("✅ Summary saved:", data);
          setEditedSummary(false);
        })
        .catch((err) => {
          console.error("❌ Summary save failed:", err);
        });
    }

    if (editedCapabilities.length > 0) {
      console.log("🚀 Sending capability updates:", { capabilities: editedCapabilities });
      fetch("http://localhost:8000/update-capabilities", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          company_id: companyId,
          product_id: productId,
          capabilities: editedCapabilities.map(cap => ({
            node_id: cap.node_id, // must be present!
            name: cap.name,
            description: cap.description
          })),
        }),
      })
        .then((res) => res.json())
        .then((data) => {
          console.log("✅ Capabilities updated:", data);
          setEditedCapabilities([]);
        })
        .catch((err) => {
          console.error("❌ Capability update failed:", err);
        });
    }

    if (newCapabilities.length > 0) {
      console.log("🚀 Sending new capabilities:", { capabilities: newCapabilities });
      fetch("http://localhost:8000/add-capabilities", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          company_id: companyId,
          product_id: productId,
          capabilities: newCapabilities,
        }),
      })
        .then((res) => res.json())
        .then((data) => {
          console.log("✅ New capabilities added:", data);
          setCapabilities(prev =>
            prev.map(cap =>
              // If cap has no node_id, try to match by name/description and update with node_id from backend
              cap.node_id
                ? cap
                : data.capabilities.find(
                  (c: any) => c.name === cap.name && c.description === cap.description
                  ) || cap
            )
          );
          setNewCapabilities([]); // Clear new capabilities if you track them separately
        })
        .catch((err) => {
          console.error("❌ New capability save failed:", err);
        });
    }
  };

  // Save changes before exiting
  useEffect(() => {
    const handleBeforeUnload = () => {
      handleSave();
    };

    window.addEventListener("beforeunload", handleBeforeUnload);
    return () => {
      window.removeEventListener("beforeunload", handleBeforeUnload);
    };
  }, []);

  // // Debounce the save function
  // const _ = debounce(() => {
  //   console.log("🔄 Autosave triggered");
  //   handleSave();
  // }, 1000);
  // useEffect(() => {
  //   const token = localStorage.getItem("token");
  //   fetch("http://localhost:8000/me", {
  //     headers: { Authorization: `Bearer ${token}` },
  //   })
  //     .then((res) => res.json())
  //     .then((data) => {
  //       setCompanyId(data.company_id);
  //       console.log("✅ Company ID set:", data.company_id);
  //     });
  // }, []);

  /*const handleAnalyzeClick = () => {
    const token = localStorage.getItem("token");
    setIsGenerating(true);
  
    // 🛡️ Prevent running before company_id is ready
    if (!productId) {
      alert("Product ID is not yet available. Please wait a moment and try again.");
      console.error("❌ Cannot run analysis without product_id.");
      setIsGenerating(false);
      return;
    }
  
    fetch("http://localhost:8000/analyze/deep", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify({
        product_id: productId,
      }),
    })
      .then((res) => res.json())
      .then((data) => {
        if (data.error) {
          console.error("🛑 Deep analysis failed:", data.error);
          alert("Could not generate persona analysis. Please try again.");
          setIsGenerating(false);
          return;
        }
  
        console.log("✅ Deep analysis success. Printing all data attributes", data);
        setPersonaSuggestions(data.personas || []);
        setShowAdvanced(true);
        setIsGenerating(false);
      })
      .catch((err) => {
        console.error("❌ Deep analysis error:", err);
        alert("An unexpected error occurred while generating analysis.");
        setIsGenerating(false);
      });
  };*/


  return (
    <div className="bg-white rounded-xl shadow p-6 space-y-4">
      {loading && <p className="text-blue-500">Analyzing website content...</p>}
      {error && <p className="text-red-500">{error}</p>}

      {!loading && !error && (
        <>
          <div>
            <h3 className="text-lg font-semibold">Value Prop Summary</h3>
            <textarea
              value={summary}
              onChange={(e) => handleSummaryChange(e.target.value)}
              onBlur={handleSave}
              rows={4}
              className="bg-transparent text-md font-normal text-indigo-700 w-full focus:outline-none focus:ring-0 border-none p-0 m-0"
            />
          </div>

          <div>
            <h3 className="text-lg font-semibold">Core Capabilities</h3>
            <ul className="space-y-4">
                {capabilities.map((cap, idx) => (
                    <li key={idx} className="border border-gray-200 p-4 rounded-lg bg-white shadow-sm space-y-2">
                        <input
                            type="text"
                            value={cap.name}
                            onChange={(e) => handleCapabilityChange(idx, "name", e.target.value)}
                            onBlur={handleSave}
                            className="w-full text-indigo-700 font-semibold bg-transparent border-b border-gray-300 focus:outline-none"
                        />
                        <textarea
                            value={cap.description}
                            onChange={(e) => handleCapabilityChange(idx, "description", e.target.value)}
                            onBlur={handleSave}
                            rows={2}
                            className="w-full text-sm text-gray-600 bg-transparent border-none focus:outline-none"
                        />
                    </li>
                ))}
            </ul>

            <button
              onClick={handleAddCapability}
              className="mt-2 text-blue-600 hover:underline text-sm"
            >
              + Add Capability
            </button>
          </div>



        </>
        )}
    </div>
    );
}

export default WebsiteAnalysis;
