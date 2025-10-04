import React from "react";

type PillBoxProps = {
  label: string;
  score: number;
};

export const getPillClass = (score: number) => {
  if (score > 0.8) {
    return "inline-block bg-blue-500 text-white border border-blue-600 text-xs font-semibold mr-2 mb-2 px-3 py-1 rounded-[25px]";
  } 
  else if (score > 0.6) {
    return "inline-block bg-blue-300 text-white border border-blue-400 text-xs font-semibold mr-2 mb-2 px-3 py-1 rounded-[25px]";
  } 
  else if (score > 0.4) {
    return "inline-block bg-blue-200 text-white border border-blue-300 text-xs font-semibold mr-2 mb-2 px-3 py-1 rounded-[25px]";
  } 
  else {
    return "inline-block bg-blue-100 text-blue-300 border border-blue-200 text-xs font-semibold mr-2 mb-2 px-3 py-1 rounded-[25px]";
  }
};

const PillBox = ({ label, score }: PillBoxProps) => (
  <span className={getPillClass(score)}>
    {label}
  </span>
);

export default PillBox;