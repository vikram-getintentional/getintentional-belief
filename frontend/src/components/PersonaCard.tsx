import React from "react";

export type PersonaSuggestion = {
  persona_title: string;
  persona_departments: string[];
  persona_seniority: string[];
  persona_ids: string[];
  max_relevance: number;
  jobs: string[];
  pains: string[];
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
  if (!persona) return null;
  return (
    <div
      className={`border p-4 rounded-lg shadow-sm transition cursor-pointer ${
        isSelected ? "ring-2 ring-indigo-500 bg-indigo-50" : "bg-white"
      }`}
      onClick={onToggle}
    >
      <div className="mb-2">
        <h4 className="font-semibold text-indigo-700">{persona.persona_title}</h4>
        <p className="text-sm text-gray-500">
          {persona.persona_seniority.join(", ")} • {persona.persona_departments.join(", ")}
        </p>
        <p className="text-xs text-indigo-600 font-medium mt-1">
          Relevance: {persona.max_relevance}
          Relevance Label: {relevanceLabel(persona.max_relevance)}
        </p>
      </div>

      {persona.jobs.length > 0 && (
        <div className="space-y-2 mt-3">
          <strong>Jobs:</strong>
          <ul className="list-disc list-inside ml-4 mt-1 text-sm text-gray-700">
            {persona.jobs.map((job, jdx) => (
              <li key={jdx}>{job}</li>
            ))}
          </ul>
        </div>
      )}

      {persona.pains.length > 0 && (
        <div className="mt-3">
          <strong>Pains:</strong>
          <ul className="list-disc list-inside ml-4 mt-1 text-sm text-gray-600">
            {persona.pains.map((pain, idx) => (
              <li key={idx}>{pain}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

export default PersonaCard;