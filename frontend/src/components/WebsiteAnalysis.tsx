import { useEffect, useMemo, useState, useCallback } from "react";
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

  // ---- draft key used for local autosave (safe even if backend isn't ready)
  const draftKey = useMemo(() => {
    const c = companyId ?? "unknownCompany";
    const p = productId ?? "unknownProduct";
    return `gi:draft:website-analysis:${c}:${p}`;
  }, [companyId, productId]);

  // ---- define handleSave so cleanup has something to call
  const handleSave = useCallback(() => {
    // only save if anything changed
    const hasChanges =
      editedSummary || editedCapabilities.length > 0 || newCapabilities.length > 0;

    if (!hasChanges) return;

    try {
      const payload = {
        company_id: companyId,
        product_id: productId,
        summary,
        // Merge current capabilities + edits + new ones (dedupe by name)
        capabilities: (() => {
          const all = [...capabilities, ...editedCapabilities, ...newCapabilities];
          const seen = new Set<string>();
          return all.filter((c) => {
            const key = (c.name || "").toLowerCase().trim();
            if (!key || seen.has(key)) return false;
            seen.add(key);
            return true;
          });
        })(),
        savedAt: new Date().toISOString(),
      };

      // 1) Always store a draft locally so we never crash / lose edits
      localStorage.setItem(draftKey, JSON.stringify(payload));

      // 2) (Optional) If you have a backend route, you can uncomment this:
      // const token = localStorage.getItem("token");
      // fetch("http://localhost:8000/save_value_prop", {
      //   method: "POST",
      //   headers: {
      //     "Content-Type": "application/json",
      //     Authorization: `Bearer ${token}`,
      //   },
      //   body: JSON.stringify(payload),
      // }).catch(() => {
      //   // Swallow errors here—draft is already safe in localStorage
      // });

      // reset edit buffers after saving
      setEditedCapabilities([]);
      setNewCapabilities([]);
      setEditedSummary(false);
      // console.log("💾 Draft saved");
    } catch (e) {
      // Never throw from cleanup
      // console.error("Draft save failed", e);
    }
  }, [
    companyId,
    productId,
    summary,
    capabilities,
    editedSummary,
    editedCapabilities,
    newCapabilities,
    draftKey,
  ]);

  // Save when the route changes (component unmount or location change)
  useEffect(() => {
    return () => {
      // guard: don't blow up on teardown
      try {
        handleSave();
      } catch {
        /* no-op */
      }
    };
  }, [location, handleSave]);

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

  // Load any existing draft once we know the draftKey (after companyId/productId available)
  useEffect(() => {
    try {
      const raw = localStorage.getItem(draftKey);
      if (!raw) return;
      const draft = JSON.parse(raw);
      if (draft?.summary) setSummary(draft.summary);
      if (Array.isArray(draft?.capabilities)) setCapabilities(draft.capabilities);
    } catch {
      /* ignore */
    }
  }, [draftKey]);

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
        // fresh analyze result replaces any local draft
        try {
          localStorage.removeItem(draftKey);
        } catch {
          /* ignore */
        }
      })
      .catch(() => {
        setError("Failed to analyze content.");
        setLoading(false);
      });
  }, [scrapedText, url, companyId, companyIdLoading, plgCta, footerFeatures, onAnalyzeSuccess, draftKey]);

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
                <li
                  key={`${cap.node_id ?? cap.name}-${idx}`}
                  className="border border-gray-200 p-4 rounded-lg bg-white shadow-sm"
                >
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
