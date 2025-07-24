import React from "react";

type Props = {
  trigger: any; // Replace 'any' with a more specific type if you know the structure
};

const ZmotTriggerCard = ({ trigger }: Props) => (
  <div className="border rounded p-4 shadow bg-white">
    <h4 className="font-semibold mb-2">{trigger.theme || "Trigger Theme"}</h4>
    <pre className="text-xs text-gray-700">{JSON.stringify(trigger, null, 2)}</pre>
  </div>
);

export default ZmotTriggerCard;