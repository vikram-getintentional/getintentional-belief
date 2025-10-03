import React, { useMemo } from "react";
import { Box, Typography } from "@mui/material";
import {
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Label
} from "recharts";
import { zNormalizePersonas, domainOf, PersonaMetric } from "../utils/rcs_metrics";

type Persona = PersonaMetric;

const CustomTooltip = ({ active, payload }: any) => {
  if (active && payload && payload.length > 0) {
    const p = payload[0].payload;
    return (
      <Box sx={{ bgcolor: "white", p: 1, border: "1px solid #ccc", borderRadius: 1 }}>
        <Typography variant="subtitle2">{p.label}</Typography>
        <Typography variant="body2">Involvement (raw): {p.x.toFixed(3)}</Typography>
        <Typography variant="body2">Activation (raw): {p.y.toFixed(3)}</Typography>
        <Typography variant="body2">Involvement (z): {p.zx.toFixed(2)}</Typography>
        <Typography variant="body2">Activation (z): {p.zy.toFixed(2)}</Typography>
      </Box>
    );
  }
  return null;
};

export default function RCSPersonaInvolvmentVisual({ personas }: { personas: Persona[] }) {
  const data = useMemo(() => zNormalizePersonas(personas), [personas]);

  // raw ranges if you ever want to toggle axes
  const xRawDomain = useMemo(() => domainOf(data.map(d => d.x)), [data]);
  const yRawDomain = useMemo(() => domainOf(data.map(d => d.y)), [data]);

  return (
    <Box sx={{ width: "100%", height: 400 }}>
      <Typography variant="h6" sx={{ mb: 2 }}>
        Persona Involvement vs Activation (z-normalized)
      </Typography>
      <ResponsiveContainer width="100%" height="100%">
        <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
          <CartesianGrid />
          <XAxis
            type="number"
            dataKey="zx"
            name="Involvement (z)"
            domain={[-1, 1]} // compact view; make [-2,2] if you prefer
            allowDataOverflow
          >
            <Label value="Involvement (z)" position="insideBottom" offset={-10} />
          </XAxis>
          <YAxis
            type="number"
            dataKey="zy"
            name="Activation (z)"
            domain={[-1, 1]}
            allowDataOverflow
          >
            <Label value="Activation (z)" angle={-90} position="insideLeft" offset={-10} />
          </YAxis>
          <Tooltip content={<CustomTooltip />} cursor={{ strokeDasharray: "3 3" }} />
          <Scatter name="Personas" data={data} fill="#1976d2" />
        </ScatterChart>
      </ResponsiveContainer>
    </Box>
  );
}
