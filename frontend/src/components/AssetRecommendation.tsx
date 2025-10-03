import React from "react";

export default function AssetRecommendation({ plays }: { plays: any[] }) {
  if (!plays || plays.length === 0) return null;
  return (
    <div style={{ marginTop: "0.5em" }}>
      <b>Recommended Engagements:</b>
      <ul>
        {plays.map((play, idx) => (
          <li key={idx}>
            {play.asset?.name && <span>Asset: <b>{play.asset.name}</b></span>}
            {play.channel?.name && <span> &mdash; Channel: <b>{play.channel.name}</b></span>}
            {play.note && <span>{play.note}</span>}
          </li>
        ))}
      </ul>
    </div>
  );
}