import React from "react";

type ZmotCardProps = {
  zmot: {
    trigger_event: string;
    trigger_event_relevance: number;
    observable_moments: {
      observable_moment: string;
      obs_relevance: number;
      keywords: [string, number][];
    }[];
  };
};

const ZmotCard = ({ zmot }: { zmot: any }) => (
  <div className="mb-4 p-4 border rounded bg-white shadow">
    <h3 className="font-bold text-lg mb-2">
      {zmot.trigger_event}{" "}
      <span className="text-gray-500 text-sm">
        ({zmot.trigger_event_relevance?.toFixed(2)})
      </span>
    </h3>
    <ul className="mb-2 space-y-1">
      {(!zmot.observable_moments || zmot.observable_moments.length === 0) ? (
        <li className="text-gray-400">No observable moments</li>
      ) : (
        zmot.observable_moments.map((moment: any, idx: number) => {
          // Handle both array and object forms
          const [label, score, keywords] = Array.isArray(moment)
            ? moment
            : [moment.observable_moment, moment.obs_relevance, moment.keywords];
          return (
            <li key={label}>
              <b>{label}</b> ({score?.toFixed(2)})
              <ul className="ml-4 list-disc">
                {!keywords || keywords.length === 0 ? (
                  <li className="text-gray-400 text-sm">No keywords</li>
                ) : (
                  keywords.map(([kw, rel]: [string, number], kidx: number) => (
                    <li key={kidx} className="text-sm">
                      {kw} ({rel?.toFixed(2)})
                    </li>
                  ))
                )}
              </ul>
            </li>
          );
        })
      )}
    </ul>
  </div>
);

export default ZmotCard;