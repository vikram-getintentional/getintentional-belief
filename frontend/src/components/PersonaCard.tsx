import React, { useState } from "react";
import { getCardClass } from "./interfaceElements/cardUtils";
import Pill from "./interfaceElements/pillbox";

export type PersonaSuggestion = {
  persona_title: string;
  persona_departments: string[];
  persona_seniority: string[];
  persona_ids: string[];
  max_relevance: number;
  jobs: string[];   // Each is a JSON string: {description, relevance}
  pains: string[];  // Each is a JSON string: {description, relevance}
};

type Props = {
  persona: PersonaSuggestion;
  isSelected: boolean;
  onToggle: () => void;
};

const relevanceLabel = (score: number): string => {
  if (score >= 0.8) return "★★★★★";
  if (score >= 0.6) return "★★☆☆☆";
  if (score >= 0.3) return "★☆☆☆☆";
  return "☆☆☆☆☆";
};

const PersonaCard = ({ persona, isSelected, onToggle }: Props) => {
  const [showAllJobs, setShowAllJobs] = useState(false);
  const [showAllPains, setShowAllPains] = useState(false);

  if (!persona) return null;

  // Helper to parse job/pain strings
  const parseItem = (str: string) => {
    try {
      return JSON.parse(str);
    } catch {
      return { description: str, relevance: 0 };
    }
  };

  const jobsToShow = showAllJobs ? persona.jobs : persona.jobs.slice(0, 3);
  const painsToShow = showAllPains ? persona.pains : persona.pains.slice(0, 3);

  return (
    <div
      className={getCardClass(persona.max_relevance)}
    >
      <div className="mb-2">
        <h4 className="font-semibold text-indigo-700">{persona.persona_title}</h4>
        <p className="text-sm text-gray-500">
          {persona.persona_seniority.join(", ")} • {persona.persona_departments.join(", ")}
        </p>
        <p className="text-xs text-indigo-600 font-medium mt-1">
          Relevance Label: {relevanceLabel(persona.max_relevance)}
        </p>
      </div>

      {persona.jobs.length > 0 && (
        <div className="space-y-2 mt-3">
          <strong>Jobs:</strong>
          <div className="flex flex-wrap mt-1">
            {jobsToShow.map((jobStr, jdx) => {
              const jobObj = parseItem(jobStr);
              return (
                <Pill key={jdx} label={jobObj.description} score={jobObj.relevance} />
              );
            })}
            {!showAllJobs && persona.jobs.length > 3 && (
              <button
                className="text-xs text-indigo-600 underline ml-2"
                onClick={e => {
                  e.stopPropagation();
                  setShowAllJobs(true);
                }}
              >
                <span className="text-lg font-bold">▼</span>
              </button>
            )}
            {showAllJobs && persona.jobs.length > 3 && (
              <button
                className="text-xs text-indigo-600 underline ml-2"
                onClick={e => {
                  e.stopPropagation();
                  setShowAllJobs(false);
                }}
              >
                <span className="text-lg font-bold">▲</span>
              </button>
            )}
          </div>
        </div>
      )}

      {persona.pains.length > 0 && (
        <div className="mt-3">
          <strong>Pains:</strong>
          <div className="flex flex-wrap mt-1">
            {painsToShow.map((painStr, jdx) => {
              const painObj = parseItem(painStr);
              return (
                <Pill key={jdx} label={painObj.description} score={painObj.relevance} />
              );
            })}
            {!showAllPains && persona.pains.length > 3 && (
              <button
                className="text-xs text-indigo-600 underline ml-2"
                onClick={e => {
                  e.stopPropagation();
                  setShowAllPains(true);
                }}
              >
                <span className="text-lg font-bold">▼</span>
              </button>
            )}
            {showAllPains && persona.pains.length > 3 && (
              <button
                className="text-xs text-indigo-600 underline ml-2"
                onClick={e => {
                  e.stopPropagation();
                  setShowAllPains(false);
                }}
              >
                <span className="text-lg font-bold">▲</span>
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default PersonaCard;