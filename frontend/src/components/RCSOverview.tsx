// ============================
// Component: RCSOverview.tsx
// ============================
import React from "react";
import {
  Box, Card, CardContent, Chip, Divider, Grid, Stack, Typography,
  Tooltip
} from "@mui/material";

import { bucketByRole, zNormalizePersonas, classifyRole, PersonaMetric } from "../utils/rcs_metrics";
import { pickPreferredSource } from "../utils/rcs_normalize";



type PersonaRow = { id: string; label?: string; involvement?: number; activation?: number };

function roleOf(p: PersonaRow) {
  const inv = p.involvement ?? 0;
  const act = p.activation ?? 0;
  if (inv >= 0.5 && act >= 0.5) return "Potential Champion";
  if (inv >= 0.5 && act < 0.5)  return "Blocker";
  if (inv < 0.5 && act >= 0.5)  return "Operator";
  return "Passive Influencer";
}

function pct(n?: number) { return `${Math.round(100 * (n ?? 0))}%`; }

export default function RCSOverview({ rcs }: { rcs: any }) {
  if (!rcs) return null;

  const preferred = pickPreferredSource(rcs) || {};
  const personasSrc = preferred?.frozen_persona_pool || preferred?.all_personas || rcs?.all_personas || rcs?.top_N_personas || [];
  const personas: PersonaMetric[] = (Array.isArray(personasSrc) ? personasSrc.map((p: any) => ({
    id: p?.id || p?.persona || p?.label || String(p),
    label: p?.persona_label || p?.label || p?.id || String(p),
    involvement: Number(p?.involvement ?? p?.I ?? 0),
    activation: Number(p?.activation ?? p?.A ?? 0),
  })) : []);
 
  const byRole = bucketByRole(personas, { mode: "z", zxCut: 0, zyCut: 0 });
   personas.forEach(p => byRole[roleOf(p)].push(p));
 

  const win =
    Number(preferred?.graph_win_likelihood ?? preferred?.baseline?.win_likelihood) ||
    Number(rcs?.graph_win_likelihood) ||
    Number(rcs?.baseline?.win_likelihood) ||
    Number(rcs?.concern_sequences?.[0]?.final_win) ||
    0;
 
  const coalitions =
    preferred?.concern_coalitions ||
    preferred?.coalitions ||
    rcs?.concern_coalitions ||
    rcs?.coalitions ||
    [];

  return (
    <Stack spacing={2}>
      {/* Meta / Stats */}
      <Card variant="outlined">
        <CardContent>
          <Typography variant="h6">Graph Snapshot</Typography>
          <Divider sx={{ my: 1.5 }} />
          <Stack direction="row" spacing={4} flexWrap="wrap">
            <Stack>
              <Typography variant="body2" color="text.secondary">Graph win likelihood</Typography>
              <Typography variant="h5">{(win as number).toFixed(3)}</Typography>
            </Stack>
            <Stack>
              <Typography variant="body2" color="text.secondary">Personas analyzed</Typography>
              <Typography variant="h5">{personas.length}</Typography>
            </Stack>
            <Stack>
              <Typography variant="body2" color="text.secondary">Coalition ideas</Typography>
              <Typography variant="h5">{coalitions.length}</Typography>
            </Stack>
          </Stack>
        </CardContent>
      </Card>

      {/* Persona roles */}
      <Card variant="outlined">
        <CardContent>
          <Typography variant="h6">Persona Roles (Involvement × Activation)</Typography>
          <Divider sx={{ my: 1.5 }} />
          <Grid container spacing={2}>
            {Object.entries(byRole).map(([role, list]) => (
              <Grid size={{ xs: 12, md: 6, lg: 6 }} key={role}>
                <Card variant="outlined">
                  <CardContent>
                    <Stack spacing={1}>
                      <Stack direction="row" spacing={1} alignItems="center">
                        <Chip
                          color={
                            role === "Potential Champion" ? "success" :
                            role === "Blocker" ? "error" :
                            role === "Operator" ? "info" : "default"
                          }
                          label={role}
                          size="small"
                        />
                        <Typography variant="body2" color="text.secondary">
                          {list.length} persona{list.length !== 1 ? "s" : ""}
                        </Typography>
                      </Stack>

                      <Stack spacing={0.5}>
                        {list.slice(0, 6).map(p => (
                          <Tooltip
                            key={p.id}
                            title={`Involvement ${pct(p.involvement)} · Activation ${pct(p.activation)}`}
                          >
                            <Typography variant="body2">• {p.label || p.id}</Typography>
                          </Tooltip>
                        ))}
                        {list.length > 6 && (
                          <Typography variant="caption" color="text.secondary">
                            +{list.length - 6} more…
                          </Typography>
                        )}
                      </Stack>
                    </Stack>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </CardContent>
      </Card>

      {/* Coalitions */}
      <Card variant="outlined">
        <CardContent>
          <Typography variant="h6">Potential Persona Coalitions</Typography>
          <Divider sx={{ my: 1.5 }} />
          {!coalitions.length ? (
            <Typography color="text.secondary">No coalitions available.</Typography>
          ) : (
            <Stack spacing={1.5}>
              {coalitions.slice(0, 8).map((c: any, i: number) => (
                <Card key={i} variant="outlined">
                  <CardContent>
                    <Stack direction="row" spacing={2} alignItems="center" flexWrap="wrap">
                      <Chip color="primary" size="small" label={`Lift ${(c.lift ?? 0).toFixed(3)}`} />
                      <Chip size="small" label={`Synergy ${(c.synergy ?? 0).toFixed(3)}`} />
                      <Divider orientation="vertical" flexItem sx={{ mx: 1 }} />
                      <Stack direction="row" spacing={1} flexWrap="wrap">
                        {(c.concerns || []).map((cc: any) => (
                          <Chip
                            key={`${cc.persona}-${cc.concern_id}`}
                            label={`${cc.persona?.split(":").slice(-1)[0]} · ${cc.concern_label || cc.concern_id}`}
                            size="small"
                            variant="outlined"
                          />
                        ))}
                      </Stack>
                    </Stack>
                  </CardContent>
                </Card>
              ))}
            </Stack>
          )}
        </CardContent>
      </Card>
    </Stack>
  );
}
