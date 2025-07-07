import { useEffect, useState } from "react";
import PersonaCard from "./PersonaCard";
import SpinnerIcon from "../utils/SpinnerIcon";

// Types
export type PersonaSuggestion = {
  persona: {
    title: string;
    department: string;
    seniority: string;
    hop?: number; // optional for now, but available
  };
  relevance: number;
  jobs: {
    description: string;
    pains: string[];
  }[];
  capabilities: string[];
};

type Props = {
  product_id: string;
  onSave: (confirmed: PersonaSuggestion[]) => void;
};

const PersonaBuilder = ({ product_id, onSave }: Props) => {
  const [personaSuggestions, setPersonaSuggestions] = useState<PersonaSuggestion[]>([]);
  const [selected, setSelected] = useState<Record<string, boolean>>({});
  const [loading, setLoading] = useState(false);
  const [fetchError, setFetchError] = useState("");

  useEffect(() => {
    if (!product_id) return;
    const token = localStorage.getItem("token");
    setLoading(true);

    fetch("http://localhost:8000/analyze/deep", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify({ product_id }),
    })
      .then((res) => res.json())
      .then((data) => {
        console.log("Raw API response in PersonaBuilder:", data);
        const aggregated = Array.isArray(data.hop_0_results?.aggregated_personas) ? data.hop_0_results.aggregated_personas : [];
        console.log("✅ Loaded aggregated persona cards:", aggregated);

        setPersonaSuggestions(aggregated);

        const toggles: Record<string, boolean> = {};
        aggregated.forEach((p) => {
          const key = `${p.persona.title}__${p.persona.department}__${p.persona.seniority}`;
          toggles[key] = true;
        });
        setSelected(toggles);
        setLoading(false);
      })
      .catch((err) => {
        console.error("Fetch error:", err);
        setFetchError("Could not load personas.");
        setLoading(false);
      });
  }, [product_id]);

  const togglePersona = (key: string) => {
    setSelected((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  const handleSave = async () => {
    const final = personaSuggestions.filter((p) => {
      const key = `${p.persona.title}__${p.persona.department}__${p.persona.seniority}`;
      return selected[key];
    });

    if (final.length === 0) return;

    // Prepare payload for /analyze/hop_plus
    // Flatten jobs and pains for each persona
    const results = final.flatMap((p) =>
      p.jobs.flatMap((jobObj) =>
        jobObj.pains.map((pain) => ({
          persona: p.persona,
          job: jobObj.description,
          pain,
          capability: p.capabilities[0] || "", // or adjust as needed
          relevance: p.relevance,
        }))
      )
    );

    const token = localStorage.getItem("token");
    setLoading(true);
    setFetchError("");

    try {
      const response = await fetch("http://localhost:8000/analyze/hop_plus", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          results,
          product_id,
        }),
      });

      const data = await response.json();
      setLoading(false);

      // You can now use data.hop_results to display Hop+ results to the user
      // For example, you might want to call a prop like onHopResults(data.hop_results)
      console.log("Hop+ results:", data.hop_results);

      // Optionally, call onSave if you want to keep the old behavior
      onSave(final);

    } catch (err) {
      setLoading(false);
      setFetchError("Could not analyze dependencies.");
      console.error("Hop+ fetch error:", err);
    }
  };

  return (
    <div className="space-y-4">
      {loading && <p className="text-blue-600">Analyzing personas...</p>}
      {fetchError && <p className="text-red-600">{fetchError}</p>}

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        {!loading &&
          personaSuggestions.map((p, idx) => {
            const key = `${p.persona.title}__${p.persona.department}__${p.persona.seniority}`;
            return (
              <PersonaCard
                key={idx}
                persona={p}
                isSelected={!!selected[key]}
                onToggle={() => togglePersona(key)}
              />
            );
          })}
      </div>

      <div className="flex gap-4 mt-6">
        <button
          onClick={handleSave}
          disabled={Object.values(selected).every((v) => !v)}
          className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700"
        >
          ✅ Infer Hop+ Dependencies
        </button>
      </div>
    </div>
  );
};

export default PersonaBuilder;
