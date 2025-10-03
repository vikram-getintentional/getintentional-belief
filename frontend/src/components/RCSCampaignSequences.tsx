// ============================
// Component: RCSCampaignSequences.tsx
// ============================
import React from "react";
import {
  Box, Card, CardContent, Chip, Divider, Grid, Stack, Typography,
  Tooltip, LinearProgress
} from "@mui/material";

type StepItem = {
  step: number;
  persona: string;
  concern_id: string;
  stage?: string;
  concern_label?: string;
  delta_lift?: number;
  win_likelihood?: number;
};

type Seq = {
  sequence: { persona: string; concern_id: string; stage?: string; concern_label?: string }[];
  steps: StepItem[];
  base_win?: number;
  final_win?: number;
  total_lift?: number;
};

type Asset = {
  asset_id?: string;
  asset_name?: string;
  channel?: { id?: string; name?: string } | null;
  fitness?: number;
  why?: string;
  note?: string; // when no plays exist
};

type AssetsByPersona = Record<string, Record<string, Asset[]>>;

function StageChip({ stage }: { stage?: string }) {
  const s = (stage || "").toLowerCase();
  const color =
    s === "solution" ? "success" : s === "pain" ? "warning" : "default";
  return <Chip size="small" label={stage || "—"} color={color as any} />;
}

function FitnessBar({ fitness }: { fitness?: number }) {
  const v = Math.max(0, Math.min(1, fitness ?? 0));
  return (
    <Box sx={{ minWidth: 120 }}>
      <Tooltip title={`Fitness ${(v * 100).toFixed(0)}%`}>
        <LinearProgress variant="determinate" value={v * 100} />
      </Tooltip>
    </Box>
  );
}

// pick the best available asset or return a “gap” note
function bestAsset(assets?: Asset[]): Asset | null {
  if (!assets || assets.length === 0) return null;
  const withFitness = [...assets].sort((a, b) => (b.fitness ?? 0) - (a.fitness ?? 0));
  return withFitness[0];
}

/**
 * Props:
 * - sequences: rcs.concern_sequences (or report.sequences from backend)
 * - labelMap:
 *    personaLabelById[id] = human label
 *    concernLabelById[id] = human label (fallback handled)
 * - assetsByPersona: { [personaId]: { [concernId]: Asset[] } }
 */
export default function RCSCampaignSequences({
  sequences,
  labelMap,
  assetsByPersona,
}: {
  sequences: Seq[];
  labelMap: {
    personaLabelById: Record<string, string>;
    concernLabelById: Record<string, string>;
  };
  assetsByPersona: AssetsByPersona;
}) {
  if (!sequences?.length) {
    return <Typography color="text.secondary">No sequences available.</Typography>;
  }

  return (
    <Stack spacing={2}>
      {sequences.map((seq, idx) => (
        <Card key={idx} variant="outlined">
          <CardContent>
            <Stack direction="row" justifyContent="space-between" alignItems="baseline">
              <Typography variant="h6">Sequence #{idx + 1}</Typography>
              <Stack direction="row" spacing={2} alignItems="center">
                <Typography variant="body2" color="text.secondary">
                  Base win: {(seq.base_win ?? 0).toFixed(3)}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Final win: {(seq.final_win ?? 0).toFixed(3)}
                </Typography>
                <Chip
                  label={`Total Δlift ${(seq.total_lift ?? 0).toFixed(3)}`}
                  color="primary"
                  size="small"
                />
              </Stack>
            </Stack>

            <Divider sx={{ my: 2 }} />

            <Grid container spacing={2}>
              {seq.steps.map((st) => {
                const personaLabel =
                  labelMap.personaLabelById[st.persona] || st.persona;
                const concernLabel =
                  st.concern_label ||
                  labelMap.concernLabelById[st.concern_id] ||
                  st.concern_id;

                const assets =
                  assetsByPersona?.[st.persona]?.[st.concern_id] ?? [];
                const top = bestAsset(assets);

                const coverage =
                  assets?.length &&
                  !("note" in (assets[0] || {})) &&
                  (assets.some(a => (a.fitness ?? 0) >= 0.25));

                return (
                  <Grid size={{ xs: 12 }} key={`${st.persona}-${st.concern_id}-${st.step}`}>
                    <Card variant="outlined" sx={{ borderLeft: 4, borderLeftColor: "primary.main" }}>
                      <CardContent>
                        <Stack spacing={1}>
                          <Stack direction="row" spacing={1} alignItems="center">
                            <Chip label={`Step ${st.step}`} size="small" />
                            <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
                              {personaLabel}
                            </Typography>
                            <StageChip stage={st.stage} />
                            {coverage ? (
                              <Chip size="small" color="success" label="COVERED" />
                            ) : (
                              <Chip size="small" color="default" label="ASSET GAP" />
                            )}
                          </Stack>

                          <Typography variant="body1">
                            Solve: <strong>{concernLabel}</strong>
                          </Typography>

                          <Stack direction="row" spacing={3} alignItems="center">
                            <Chip
                              size="small"
                              color="info"
                              label={`Δlift ${(st.delta_lift ?? 0).toFixed(3)}`}
                            />
                            <Chip
                              size="small"
                              label={`Cum win ${(st.win_likelihood ?? 0).toFixed(3)}`}
                            />
                          </Stack>

                          {/* Best asset block */}
                          <Box sx={{ mt: 1 }}>
                            {top && !top.note ? (
                              <Stack spacing={0.5}>
                                <Typography variant="body2" sx={{ fontWeight: 600 }}>
                                  Best Asset: {top.asset_name || top.asset_id}
                                </Typography>
                                <Stack direction="row" spacing={2} alignItems="center">
                                  <Typography variant="caption" color="text.secondary">
                                    Channel: {top.channel?.name || "—"}
                                  </Typography>
                                  <FitnessBar fitness={top.fitness} />
                                </Stack>
                                {top.why && (
                                  <Typography variant="caption" color="text.secondary">
                                    Why: {top.why}
                                  </Typography>
                                )}
                              </Stack>
                            ) : (
                              <Typography variant="caption" color="text.secondary">
                                No mapped plays. Create a targeted asset for this step.
                              </Typography>
                            )}
                          </Box>
                        </Stack>
                      </CardContent>
                    </Card>
                  </Grid>
                );
              })}
            </Grid>
          </CardContent>
        </Card>
      ))}
    </Stack>
  );
}
