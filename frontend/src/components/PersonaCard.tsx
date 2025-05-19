import React from "react";

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
  persona: PersonaSuggestion;
  isSelected: boolean;
  onToggle: () => void;
};

const relevanceLabel = (score: number): string => {
  if (score >= 0.8) return "🔥 Very High";
  if (score >= 0.6) return "💡 High";
  if (score >= 0.3) return "🧐 Moderate";
  return "🤷 Low";
};

const PersonaCard = ({ persona, isSelected, onToggle }: Props) => {
  return (
    <div
      className={`border p-4 rounded-lg shadow-sm transition cursor-pointer ${
        isSelected ? "ring-2 ring-indigo-500 bg-indigo-50" : "bg-white"
      }`}
      onClick={onToggle}
    >
      <div className="mb-2">
        <h4 className="font-semibold text-indigo-700">{persona.persona.title}</h4>
        <p className="text-sm text-gray-500">
          {persona.persona.seniority} • {persona.persona.department}
        </p>
        <p className="text-xs text-indigo-600 font-medium mt-1">
          Relevance: {persona.relevance}
        </p>
        {persona.hop && (
          <p className="text-sm text-gray-500">Hop: {persona.hop}</p>
        )}
      </div>

      <div className="space-y-2 mt-3">
        {persona.jobs.map((job, jdx) => (
          <div key={jdx}>
            <p className="text-sm font-medium text-gray-700">🛠 {job.description}</p>
            <ul className="list-disc list-inside ml-2 text-sm text-gray-600">
              {job.pains.map((pain, idx) => (
                <li key={idx}>{pain}</li>
              ))}
            </ul>
          </div>
        ))}
      </div>

      <div className="mt-3 text-xs text-gray-500">
        Capabilities matched:
        <ul className="list-disc list-inside ml-4 mt-1">
          {persona.capabilities.map((cap, i) => (
            <li key={i}>{cap}</li>
          ))}
        </ul>
      </div>
    </div>
  );
};

export default PersonaCard;
