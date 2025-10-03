import React from "react";

export default function ReverseCaseStudyFlow({ rcsData }: { rcsData: any }) {
  const concerns = rcsData?.graph_concerns_summary ?? [];
  const marketingPlan = rcsData?.marketing_execution_plan ?? [];
  // You can format this paragraph using rcsData fields
  return (
    <div style={{ margin: "1em 0", padding: "1em", background: "#f9f9f9", borderRadius: "8px" }}>
      <h2>Baseline Report</h2>
      <h3 className="mt-4 mb-2 font-semibold">Graph Concerns Summary</h3>
      <ul>
        {concerns.map((item: any, idx: number) => (
          <li key={idx} style={{ marginBottom: "0.5em" }}>
            <b>Persona:</b> {item.persona} | <b>Stage:</b> {item.stage} | <b>ROI:</b> {item.ROI?.toExponential?.() ?? item.ROI} | <b>Lift:</b> {item.lift?.toExponential?.() ?? item.lift}
          </li>
        ))}
      </ul>
      <h3 className="mt-4 mb-2 font-semibold">Marketing Execution Plan</h3>
      <ul>
        {marketingPlan.map((item: any, idx: number) => (
          <li key={idx} style={{ marginBottom: "0.5em" }}>
            <b>Persona:</b> {item.persona} | <b>Stage:</b> {item.stage} | <b>ROI:</b> {item.roi?.toExponential?.() ?? item.roi} | <b>Lift:</b> {item.lift?.toExponential?.() ?? item.lift}
          </li>
        ))}
      </ul>
      {/* You can add more dynamic content here using rcsData */}
    </div>
  );
}