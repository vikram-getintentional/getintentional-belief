import { useEffect, useState } from "react";
import { useLocation } from "react-router-dom";

type Props = {
  url: string;
  scrapedText: string;
  plgCta?: boolean;
  footerFeatures?: string[];
  onAnalyzeSuccess?: (data: any) => void;
};

type Capability = {
  node_id?: string;
  name: string;
  description: string;
};

export default function WebsiteAnalysis({
  url,
  scrapedText,
  plgCta = false,
  footerFeatures = [],
  onAnalyzeSuccess,
}: Props) {
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
        if (typeof onAnalyzeSuccess === "function") onAnalyzeSuccess(data);
        setLoading(false);
      })
      .catch(() => {
        setError("Failed to analyze content.");
        setLoading(false);
      });
  }, [scrapedText, url, companyId, companyIdLoading]);

  

  return (
    <div className="bg-white rounded-xl shadow p-6 space-y-4">
      {loading && <p className="text-blue-500">Analyzing website content...</p>}
      {error && <p className="text-red-500">{error}</p>}

      {!loading && !error && (
        <>
          <div>
            <h3 className="text-lg font-semibold">Value Prop Summary</h3>
            <p className="text-indigo-700 whitespace-pre-wrap">{summary}</p>
          </div>

          <div>
            <h3 className="text-lg font-semibold">Core Capabilities</h3>
            <ul className="space-y-4">
              {capabilities.map((cap, idx) => (
                <li key={idx} className="border border-gray-200 p-4 rounded-lg bg-white shadow-sm">
                  <div className="font-semibold text-indigo-700">{cap.name}</div>
                  <div className="text-sm text-gray-600">{cap.description}</div>
                </li>
              ))}
            </ul>
          </div>
        </>
      )}
    </div>
  );
}