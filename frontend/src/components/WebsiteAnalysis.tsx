import { useEffect, useState } from 'react';
import PersonaBuilder from './PersonaBuilder';
import SpinnerIcon from "../utils/SpinnerIcon";



type Props = {
    url: string;
    scrapedText: string;
    plgCta?: boolean;
    footerFeatures?: string[];
};

type OrgTypes = {
  industries: string[];
  regions: string[];
  revenue_ranges: string[];
};

type Capability = {
    name: string;
    description: string;
  };


  
const WebsiteAnalysis = ({ scrapedText, url, plgCta = false, footerFeatures = [] }: Props) => {
  const [personaSuggestions, setPersonaSuggestions] = useState<PersonaPainCombo[]>([]);
  const [summary, setSummary] = useState('');
  const [capabilities, setCapabilities] = useState<Capability[]>([]);
  const [originalSummary, setOriginalSummary] = useState('');
  const [originalCapabilities, setOriginalCapabilities] = useState<Capability[]>([]);


  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [saved, setSaved] = useState(false);
  const [edited, setEdited] = useState(false);
  const [userEmail, setUserEmail] = useState("");
  const [companyId, setCompanyId] = useState<string | null>(null);
  const [isSaving, setIsSaving] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false); 


  useEffect(() => {
    const token = localStorage.getItem("token");
    fetch("http://localhost:8000/me", {
      headers: { Authorization: `Bearer ${token}` },
    })
      .then((res) => res.json())
      .then((data) => {
        setCompanyId(data.company_id);
        setUserEmail(data.email || "unknown");
        console.log("✅ Company ID set:", data.company_id);
      });

  }, []);

  useEffect(() => {
    if (!scrapedText || !url) return;

    setLoading(true);
    const token = localStorage.getItem('token');
    console.log("🚀 Sending to /analyze:", { companyId, scrapedText });
    fetch('http://localhost:8000/analyze', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify({
        url: url,
        company_id: companyId,
        text: scrapedText,
        plg_cta_found: plgCta,
        footer_features: footerFeatures,
      }),
    })
    .then((res) => res.json())
    .then((data) => {
    setSummary(data.summary || '');
    setCapabilities(data.capabilities || []);
    setOriginalSummary(data.summary || '');
    setOriginalCapabilities(data.capabilities || []);
    setLoading(false);
    })
    .catch(() => {
    setError('Failed to analyze content.');
    setLoading(false);
    });
  }, [scrapedText, url, companyId]);

  const handleSave = () => {
    setSaved(true);
    const token = localStorage.getItem('token');  

    setTimeout(() => setSaved(false), 1500);
  };

  const handleAnalyzeClick = () => {
    const token = localStorage.getItem("token");
    setIsGenerating(true);
  
    // 🛡️ Prevent running before company_id is ready
    if (!companyId) {
      alert("Company ID is not yet available. Please wait a moment and try again.");
      console.error("❌ Cannot run analysis without company_id.");
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
        company_id: companyId,
        url,
        summary,
        capabilities,
        texT: scrapedText,
        plg_cta_found: plgCta,
        footer_features: footerFeatures,
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
      
  };
  

  const handleCapabilityChange = (index: number, value: string) => {
    const updated = [...capabilities];
    updated[index] = value;
    setCapabilities(updated);
    setEdited(true);
  };

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
              onChange={(e) => {
                setSummary(e.target.value);
                setEdited(true);
              }}
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
                            onChange={(e) => {
                            const updated = [...capabilities];
                            updated[idx].name = e.target.value;
                            setCapabilities(updated);
                            setEdited(true);
                            }}
                            className="w-full text-indigo-700 font-semibold bg-transparent border-b border-gray-300 focus:outline-none"
                        />
                        <textarea
                            value={cap.description}
                            onChange={(e) => {
                            const updated = [...capabilities];
                            updated[idx].description = e.target.value;
                            setCapabilities(updated);
                            setEdited(true);
                            }}
                            rows={2}
                            className="w-full text-sm text-gray-600 bg-transparent border-none focus:outline-none"
                        />
                    </li>
                ))}
            </ul>

            <button
              onClick={() => {
                setCapabilities([...capabilities, { name: '', description: '' }]);
                setEdited(true);
              }}
              className="mt-2 text-blue-600 hover:underline text-sm"
            >
              + Add Capability
            </button>
          </div>

          {!showAdvanced && (
            <button
              onClick={handleAnalyzeClick}
              disabled={isGenerating}
              className={`px-4 py-2 rounded text-white flex items-center gap-2 ${
                isGenerating ? "bg-gray-400 cursor-not-allowed" : "bg-indigo-600 hover:bg-indigo-700"
              }`}
            >
              {isGenerating ? <><SpinnerIcon /> Analyzing...</> : "Get Primary Personas"}
            </button>
          )}

          {saved && <p className="text-green-600 text-sm">Insights saved!</p>}
          

          {showAdvanced && (
            <PersonaBuilder
                url={url}
                summary={summary}
                capabilities={capabilities}
                userEmail={userEmail}
                initialSuggestions={personaSuggestions}
        
                onSave={(finalCombos) => {
                    console.log("✅ Final approved combos:", finalCombos);
                    setSaved(true);
                    setTimeout(() => setSaved(false), 2000);
                }}
                onComplete={() => {
                console.log("✅ Persona + pain analysis complete");
                }}
            />
            )}  


        </>
        )}
    </div>
    );
}

export default WebsiteAnalysis;
