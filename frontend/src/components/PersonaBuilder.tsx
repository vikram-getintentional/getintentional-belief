import { useEffect, useState, Dispatch, SetStateAction } from "react";
import PersonaCard from "./PersonaCard";
import { taskService, TaskStatus } from "../services/taskService";

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
  token: string;
  progress: number,
  setTaskID: Dispatch<SetStateAction<string | null>>,
  onSave: (confirmed: PersonaSuggestion[]) => void;
};

const PersonaBuilder = ({ token, progress, setTaskID, product_id, onSave }: Props) => {
  const [personaSuggestions, setPersonaSuggestions] = useState<PersonaSuggestion[]>([]);
  const [selected, setSelected] = useState<Record<string, boolean>>({});
  const [fetchError, setFetchError] = useState("");
  const [statusMessage, setStatusMessage] = useState("");

  useEffect(() => {
    (async function () {
      if (!product_id) return;
      await startDeepAnalysis(product_id);
    }());
  }, [product_id]);

  const startDeepAnalysis = async (product_id: string) => {
    setFetchError("");
    setStatusMessage("Starting Hop0 graph generation...");

    try {
      // Start the async task
      const response = await taskService.startDeepAnalysis(product_id, token);
      console.log("✅ Task started:", response);
      setStatusMessage(response.message);
      setTaskID(response.task_id)
    } catch (startError: any) { // Add explicit type for startError
      console.error("❌ Failed to start deep analysis:", startError);
      setFetchError("Could not start analysis.");
      setStatusMessage("");
    }
  };

  const handleTaskSuccess = (result: any) => {
    console.log("Raw task result:", result);
    const aggregated = Array.isArray(result.hop_0_results?.personas) ? result.hop_0_results.personas : [];
    console.log("✅ Loaded aggregated persona cards:", aggregated);

    setPersonaSuggestions(aggregated);

    const toggles: Record<string, boolean> = {};
    aggregated.forEach((p) => {
      const key = `${p.persona.title}__${p.persona.department}__${p.persona.seniority}`;
      toggles[key] = true;
    });
    setSelected(toggles);
    setStatusMessage("Analysis completed successfully!");
    // setProgress(100);
  };

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

      // You can now use data.hop_results to display Hop+ results to the user
      // For example, you might want to call a prop like onHopResults(data.hop_results)
      console.log("Hop+ results:", data.hop_results);

      // Optionally, call onSave if you want to keep the old behavior
      onSave(final);

    } catch (err) {
      setFetchError("Could not analyze dependencies.");
      console.error("Hop+ fetch error:", err);
    }
  };

  return (
    <div className="space-y-4">
      {fetchError && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <p className="text-red-800">{fetchError}</p>
          <button 
            onClick={startDeepAnalysis}
            className="mt-2 text-red-600 hover:text-red-800 underline text-sm"
          >
            Try again
          </button>
        </div>
      )}

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        {progress === 0 &&
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
    </div>
  );
};

export default PersonaBuilder;
