import React from "react";
import { getCardClass } from "./interfaceElements/cardUtils"; // adjust path as needed

type CapabilityCardProps = {
  name: string;
  description?: string;
  coreness: number;
  centrality: number;
};

const CapabilityCard: React.FC<CapabilityCardProps> = ({ name, description, coreness, centrality }) => (
  <div className={getCardClass(coreness)}>
    <div className="font-semibold text-indigo-700">{name}</div>
    {description && <div className="text-gray-600 text-sm mt-1">{description}</div>}
  </div>
);

export default CapabilityCard;