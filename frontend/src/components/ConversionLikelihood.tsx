import React from "react";

export default function ConversionLikelihood({ likelihood }: { likelihood: number }) {
  const percent = Math.round(likelihood * 100);
  return (
    <span
      style={{
        display: "inline-block",
        padding: "0.5em 1em",
        borderRadius: "999px",
        background: "#e0f7fa",
        color: "#00796b",
        fontWeight: 600,
        fontSize: "1rem",
        marginBottom: "1em"
      }}
    >
      Conversion Likelihood&nbsp;
      <span style={{ fontWeight: 700 }}>{percent}%</span>
    </span>
  );
}