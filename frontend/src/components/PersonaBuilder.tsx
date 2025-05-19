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
  url: string;
  summary: string;
  capabilities: { name: string; description: string }[];
  onSave: (confirmed: PersonaSuggestion[]) => void;
  userEmail: string;
};

const PersonaBuilder = ({ url, summary, capabilities, userEmail, onSave }: Props) => {
  const [personaSuggestions, setPersonaSuggestions] = useState<PersonaSuggestion[]>([]);
  const [selected, setSelected] = useState<Record<string, boolean>>({});
  const [loading, setLoading] = useState(false);
  const [fetchError, setFetchError] = useState("");

  useEffect(() => {
    if (!summary || !capabilities.length) return;
    const token = localStorage.getItem("token");
    setLoading(true);

    fetch("http://localhost:8000/analyze/deep", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify({ summary, capabilities }),
    })
      .then((res) => res.json())
      .then((data) => {
        console.log("Raw API response in PersonaBuilder:", data);
        const aggregated = Array.isArray(data.aggregated_personas) ? data.aggregated_personas : [];
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
  }, [summary, capabilities]);

  const togglePersona = (key: string) => {
    setSelected((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  const handleSave = () => {
    const final = personaSuggestions.filter((p) => {
      const key = `${p.persona.title}__${p.persona.department}__${p.persona.seniority}`;
      return selected[key];
    });

    if (final.length > 0) {
      onSave(final);
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
          ✅ Save & Analyze
        </button>
      </div>
    </div>
  );
};

export default PersonaBuilder;
