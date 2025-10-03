import React from "react";
import AssetRecommendation from "./AssetRecommendation"

export default function PersonaCard({ concern }: { concern: any }) {
  return (
    <div style={{
      border: "1px solid #eee",
      borderRadius: "8px",
      padding: "1em",
      background: "#fff"
    }}>
      <div style={{ fontWeight: 700, marginBottom: "0.5em" }}>{concern.persona}</div>
      <div style={{ marginBottom: "0.5em" }}>
        <b>Stage:</b> {concern.stage}
      </div>
      <div style={{ marginBottom: "0.5em" }}>
        <b>ROI of Engagement:</b> {concern.ROI?.toFixed(2)}
      </div>
      <div style={{ marginBottom: "0.5em" }}>
        <b>Concern:</b> {concern.description?.upstream_pains?.join(", ") || ""}
      </div>
      <AssetRecommendation plays={concern.plays} />
    </div>
  );
}