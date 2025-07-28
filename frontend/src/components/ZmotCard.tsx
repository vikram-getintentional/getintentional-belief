import React from "react";
import { getCardClass } from "./interfaceElements/cardUtils";
import Pill from "./interfaceElements/pillbox";

type ZmotCardProps = {
  zmot: {
    trigger_event: string;
    trigger_event_relevance: number;
    observable_moments: [string, number][];
    trigger_keywords: [string, number][];
  };
};


const ZmotCard = ({ zmot }: { zmot: ZmotCardProps["zmot"] }) => (
  <div className={getCardClass(zmot.trigger_event_relevance)}>
    <h3 className="font-bold text-lg mb-2">
      {zmot.trigger_event}{" "}
    </h3>
    <div className="mb-2">
      <b>Likely Intent Signals:</b>
      <div className="flex flex-wrap mt-1">
        {zmot.observable_moments && zmot.observable_moments.length > 0 ? (
          zmot.observable_moments.map(([moment, rel], idx) => (
            <Pill key={idx} label={moment} score={rel} />
          ))
        ) : (
          <span className="text-gray-400 text-sm">No observable moments</span>
        )}
      </div>
    </div>
    <div>
      <b>Keywords:</b>
      <div className="flex flex-wrap mt-1">
        {zmot.trigger_keywords && zmot.trigger_keywords.length > 0 ? (
          zmot.trigger_keywords.map(([kw, rel], idx) => (
            <Pill key={idx} label={kw} score={rel} />
            
          ))
        ) : (
          <span className="text-gray-400 text-sm">No keywords</span>
        )}
      </div>
    </div>
  </div>
);

export default ZmotCard;