import React, { useMemo } from "react";
import { Box, Grid, Typography } from "@mui/material";
import {
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Label,
} from "recharts";

type Persona = {
  id: string;
  label: string;
  involvement: number; // 0..1
  activation: number;  // 0..1
  perceptibility?: number; // 0..1
  proximity?: number;      // 0..1
};

function zNorm(values: number[]) {
  const m = values.reduce((a, b) => a + b, 0) / (values.length || 1);
  const v = values.reduce((a, b) => a + (b - m) * (b - m), 0) / (values.length || 1);
  const s = Math.sqrt(v || 1e-9);
  return (x: number) => (x - m) / (s || 1e-9);
}

const Tip = ({ active, payload, xLabel, yLabel }: any) => {
  if (!active || !payload?.length) return null;
  const p = payload[0].payload;
  return (
    <Box sx={{ bgcolor: "white", p: 1, border: "1px solid #ccc", borderRadius: 1 }}>
      <Typography variant="subtitle2">{p.label}</Typography>
      <Typography variant="body2">{xLabel}: {p.xDisp}</Typography>
      <Typography variant="body2">{yLabel}: {p.yDisp}</Typography>
    </Box>
  );
};

export default function RCSPersonaInvolvmentVisual({
  personas = [],
  rcs,
}: {
  personas?: Persona[];
  rcs?: any;
}) {
  const P: Persona[] = useMemo(() => personas, [personas]);

  // Chart 1: Involvement vs Activation (z-normalized)
  const zData = useMemo(() => {
    const invs = P.map(p => p.involvement || 0);
    const acts = P.map(p => p.activation || 0);
    const zI = zNorm(invs);
    const zA = zNorm(acts);
    return P.map(p => ({
      id: p.id, label: p.label,
      x: zI(p.involvement || 0), y: zA(p.activation || 0),
      xDisp: (p.involvement ?? 0).toFixed(3),
      yDisp: (p.activation ?? 0).toFixed(3),
    }));
  }, [P]);

  // Chart 2: Perceptibility vs Proximity (raw 0..1)
  const ppData = useMemo(() => {
    return P.map(p => ({
      id: p.id, label: p.label,
      x: Number(p.proximity ?? 0),
      y: Number(p.perceptibility ?? 0),
      xDisp: `${Math.round(Number(p.proximity ?? 0) * 100)}%`,
      yDisp: `${Math.round(Number(p.perceptibility ?? 0) * 100)}%`,
    }));
  }, [P]);

  return (
    <Box sx={{ p: 2 }}>
      <Grid container spacing={3}>
        <Grid size={{ xs: 12, md: 6 }}>
          <Typography variant="h6" sx={{ mb: 1 }}>Persona Involvement vs Activation (z-normalized)</Typography>
          <Box sx={{ width: "100%", height: 360 }}>
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                <CartesianGrid />
                <XAxis type="number" dataKey="x" domain={[-1, 1]}>
                  <Label value="Involvement (z)" position="insideBottom" offset={-10} />
                </XAxis>
                <YAxis type="number" dataKey="y" domain={[-1, 1]}>
                  <Label value="Activation (z)" angle={-90} position="insideLeft" offset={-10} />
                </YAxis>
                <Tooltip content={<Tip xLabel="Involvement (raw)" yLabel="Activation (raw)" />} cursor={{ strokeDasharray: "3 3" }} />
                <Scatter name="Personas" data={zData} fill="#1976d2" />
              </ScatterChart>
            </ResponsiveContainer>
          </Box>
        </Grid>

        <Grid size={{ xs: 12, md: 6 }}>
          <Typography variant="h6" sx={{ mb: 1 }}>Perceptibility vs Proximity</Typography>
          <Box sx={{ width: "100%", height: 360 }}>
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                <CartesianGrid />
                <XAxis type="number" dataKey="x" domain={[0, 1]}>
                  <Label value="Proximity (to product)" position="insideBottom" offset={-10} />
                </XAxis>
                <YAxis type="number" dataKey="y" domain={[0, 1]}>
                  <Label value="Perceptibility (externally visible)" angle={-90} position="insideLeft" offset={-10} />
                </YAxis>
                <Tooltip content={<Tip xLabel="Proximity" yLabel="Perceptibility" />} cursor={{ strokeDasharray: "3 3" }} />
                <Scatter name="Personas" data={ppData} fill="#2e7d32" />
              </ScatterChart>
            </ResponsiveContainer>
          </Box>
        </Grid>
      </Grid>
    </Box>
  );
}
