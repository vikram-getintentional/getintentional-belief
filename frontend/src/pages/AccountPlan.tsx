import React, { useEffect, useMemo, useState } from "react";
import {
  Alert,
  Box,
  Breadcrumbs,
  Button,
  Card,
  CardActionArea,
  CardContent,
  Chip,
  CircularProgress,
  Divider,
  FormControl,
  Grid,
  IconButton,
  InputLabel,
  Link as MUILink,
  MenuItem,
  Paper,
  Select,
  Stack,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Tooltip,
  Typography,
} from "@mui/material";
import InfoOutlinedIcon from "@mui/icons-material/InfoOutlined";
import { Link as RouterLink, useNavigate, useParams } from "react-router-dom";
import PersonaEngagementCadence, {
  type PersonaEngagementPlan,
  type PersonaEngagementPerson,
} from "../components/PersonaEngagementCadence";
import AssetCadenceTable, {
  type AssetCadenceRow,
} from "../components/AssetCadenceTable";

type JourneyStep = {
  bucket?: string | null;
  observed?: string | null;
  predicted?: Array<string>;
  t?: number;
};

type TypicalPathStep = {
  step: number;
  persona: string;
  highlight?: string;
  impact?: string;
  confidence?: number;
  reason?: string;
};

type PersonaLikelihoodEntry = {
  persona_id: string;
  persona_label?: string | null;
  probability?: number | null;
  band?: string | null;
  belief_level?: number | null;
  belief_band?: string | null;
  phase_probs?: Record<string, number> | null;
  dominant_phase?: string | null;
  journey_phase?: string | null;
  committee_probability?: number | null;
  top_people?: PersonaEngagementPerson[];
};

type PersonLikelihoodEntry = {
  person_id?: string | null;
  display_name?: string | null;
  probability?: number | null;
  role_band?: string | null;
  belief_level?: number | null;
  phase_probs?: Record<string, number> | null;
  dominant_phase?: string | null;
  persona_id?: string | null;
  persona_label?: string | null;
};

type ExpectedNextPersona = {
  persona_id: string;
  persona_label?: string | null;
  probability?: number | null;
  band?: string | null;
  journey_stage?: string | null;
  reason?: string | null;
  top_people?: PersonaEngagementPerson[];
};

type ExpectedNextPerson = {
  persona_id?: string | null;
  persona_label?: string | null;
  person_id?: string | null;
  display_name?: string | null;
  committee_probability?: number | null;
  role_band?: string | null;
};

type Play = {
  id?: string;
  name?: string;
  asset?: {
    name?: string | null;
    title?: string | null;
  };
  asset_label?: string;
  asset_type_label?: string;
  channel_label?: string;
  channel?: {
    name?: string | null;
    channel_type_label?: string | null;
  };
  expected_delta_bp?: number | null;
  avg_confidence?: number | null;
  avg_duration_days?: number | null;
};

type PersonMatch = {
  match_id?: string;
  person_id?: string | null;
  person_name?: string | null;
  display_name?: string | null;
};

type PersonaPathEntry = {
  id: string;
  probability?: number | null;
  personas: Array<{
    id?: string | null;
    label?: string | null;
    people_names?: string[];
    matched_people?: PersonMatch[];
    belief_level?: number | null;
    priority_score?: number | null;
    expected_next_prob?: number | null;
  }>;
};

type ConversionSequenceEntry = {
  timeline_index: number;
  timeline_label?: string | null;
  timeline_days?: number | null;
  persona_descriptor?: string | null;
  expected_delta_bp?: number | null;
  avg_confidence?: number | null;
  avg_belief_conversion?: number | null;
  effort_pct?: number | null;
  time_to_effect_days?: number | null;
  segment_filters?: Record<string, string>;
  segment_summary?: string | null;
  expected_outcome_summary?: string | null;
  belief_transition_meta?: {
    stage_label?: string | null;
    narrative?: string | null;
    pain?: { label?: string | null };
    problem?: { label?: string | null };
    resolution?: { label?: string | null };
  };
  plays?: Play[];
};

type AccountPlan = {
  account_id: string;
  account_name: string;
  meta?: Record<string, unknown>;
  prediction: {
    persona_paths: PersonaPathEntry[];
    expected_next?: Array<{
      persona?: string;
      persona_label?: string;
      prob?: number;
    }>;
    expected_next_personas?: ExpectedNextPersona[];
    journey: {
      steps: JourneyStep[];
      total_steps: number;
    };
  };
  execution?: {
    plays?: Play[];
    conversion_sequence?: ConversionSequenceEntry[];
    persona_engagements?: PersonaEngagementPlan[];
    asset_cadence?: AssetCadenceRow[];
  };
  thesis?: {
    persona_likelihoods?: PersonaLikelihoodEntry[];
    person_likelihoods?: PersonLikelihoodEntry[];
    expected_next_personas?: ExpectedNextPersona[];
    expected_next_people?: ExpectedNextPerson[];
  };
  enrichment?: {
    summary?: {
      coverage_pct?: number;
      coverage_ratio?: number;
      matched_people?: number;
      required_personas?: number;
    };
  };
};

type MarketingPlan = {
  generated_at?: string;
  summary: {
    generated_at?: string;
    typical_path?: TypicalPathStep[];
    canonical_persona_path?: Array<{
      id?: string | null;
      persona_id?: string | null;
      label?: string | null;
    }>;
  };
  accounts: AccountPlan[];
  canonical_journey?: {
    typical_path?: TypicalPathStep[];
  };
  product_insights?: {
    journey_structure?: {
      typical_path?: TypicalPathStep[];
    };
  };
};

type PersonaRequirement = {
  persona_id: string;
  persona_label?: string | null;
  stage_label?: string | null;
  expected_in_deal?: number | null;
  expected_in_deal_pct?: number | null;
  people_names?: string[];
  matched_people?: Array<{
    person_id?: string;
    person_name?: string;
    display_name?: string | null;
    engagement_count?: number | null;
    last_seen_at?: string | null;
  }>;
};

type AccountEnrichmentPayload = {
  account_id: string;
  product_id: string;
  blueprint_generated_at?: string | null;
  summary?: {
    required_personas?: number;
    personas_with_matches?: number;
    matched_people?: number;
    coverage_pct?: number;
    coverage_ratio?: number;
  };
  persona_requirements?: PersonaRequirement[];
};

type PulseLine = {
  label: string;
  value: string;
};

type PulseEntry = {
  title: string;
  tone: "critical" | "positive" | "neutral" | "warning";
  lines: PulseLine[];
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const fmtPercent = (value?: number | null, fraction = false) => {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  const scaled = fraction ? value * 100 : value;
  const rounded = Number(scaled).toFixed(scaled >= 10 ? 0 : 1);
  return `${rounded.replace(/\.0$/, "")}%`;
};

const formatFrequency = (value?: number | null) => {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  const precision = value >= 1 ? 1 : 2;
  return `${value.toFixed(precision)} / wk`;
};

const fmtBasisPoints = (value?: number | null) => {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return `${value.toFixed(1)} bp`;
};

const fmtDays = (value?: number | null) => {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  if (value < 1) return "<1 day";
  if (value === 1) return "1 day";
  return `${Math.round(value)} days`;
};

const uniqueStrings = (values: Array<string | null | undefined>) =>
  Array.from(new Set(values.filter(Boolean) as string[]));

const titleize = (value?: string | null) => {
  if (!value) return "";
  return value
    .split(/[\s_-]+/)
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1).toLowerCase())
    .join(" ");
};

const segmentChipsFromFilters = (filters?: Record<string, string>) => {
  if (!filters) return [];
  return Object.entries(filters).filter(([, value]) => value && value !== "any");
};

const formatRelativeTime = (iso?: string | null) => {
  if (!iso) return "Updated just now";
  const parsed = new Date(iso);
  if (Number.isNaN(parsed.getTime())) return "Updated just now";
  const diffMs = Date.now() - parsed.getTime();
  if (diffMs < 0) return "Updated just now";
  const minutes = Math.floor(diffMs / 60000);
  if (minutes < 1) return "Updated just now";
  if (minutes < 60) return `Updated ${minutes} min${minutes > 1 ? "s" : ""} ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `Updated ${hours} hr${hours > 1 ? "s" : ""} ago`;
  const days = Math.floor(hours / 24);
  return `Updated ${days} day${days > 1 ? "s" : ""} ago`;
};

const dedupePlays = (plays: Play[] = []) => {
  const seen = new Map<string, Play>();
  plays.forEach((play) => {
    const key = play.id || play.name || play.asset_label || JSON.stringify(play);
    if (!seen.has(key)) {
      seen.set(key, play);
    }
  });
  return Array.from(seen.values());
};

const renderPersonaPath = (path?: PersonaPathEntry) => {
  if (!path || !path.personas.length) {
    return (
      <Typography variant="body2" color="text.secondary">
        Path will appear after you run journey replay for this account.
      </Typography>
    );
  }
  return (
    <Stack spacing={1}>
      <Stack direction="row" spacing={1} alignItems="center">
        <Typography variant="body2" color="text.secondary">
          Probability {fmtPercent(path.probability, true)}
        </Typography>
      </Stack>
      <Stack spacing={0.75}>
        {path.personas.map((persona, idx) => (
          <Paper key={`${path.id}-${idx}`} variant="outlined" sx={{ p: 1.25 }}>
            <Stack direction="row" spacing={1} alignItems="center">
              <Chip size="small" label={`Step ${idx + 1}`} />
              <Typography variant="subtitle2">
                {persona.label || persona.id || "Belief persona"}
              </Typography>
            </Stack>
          </Paper>
        ))}
      </Stack>
    </Stack>
  );
};

const renderTypicalPathSteps = (
  steps: TypicalPathStep[],
  placeholder = "Global path will appear when insights are available."
) => {
  if (!steps.length) {
    return (
      <Typography variant="body2" color="text.secondary">
        {placeholder}
      </Typography>
    );
  }
  return (
    <Stack spacing={1}>
      {steps.map((step) => {
        const tooltipLines = [
          step.impact ? `Impact: ${step.impact}` : null,
          step.reason ? `Why: ${step.reason}` : null,
          typeof step.confidence === "number"
            ? `Confidence ${fmtPercent(step.confidence, true)}`
            : null,
        ].filter(Boolean) as string[];
        const tooltip =
          tooltipLines.length > 0 ? (
            <Box>
              {tooltipLines.map((line, idx) => (
                <Typography key={`${step.persona}-detail-${idx}`} variant="body2">
                  {line}
                </Typography>
              ))}
            </Box>
          ) : null;
        return (
          <Paper key={`${step.step}-${step.persona}`} variant="outlined" sx={{ p: 1.25 }}>
            <Stack direction="row" spacing={1} alignItems="center">
              <Chip size="small" label={`Step ${step.step}`} />
              <Typography variant="subtitle2">{step.persona}</Typography>
              {tooltip && (
                <Tooltip title={tooltip} arrow>
                  <IconButton size="small">
                    <InfoOutlinedIcon fontSize="inherit" />
                  </IconButton>
                </Tooltip>
              )}
            </Stack>
            <Typography variant="caption" color="text.secondary">
              {step.highlight || step.reason || "Key activation in the canonical path"}
            </Typography>
          </Paper>
        );
      })}
    </Stack>
  );
};

// ---------------------------------------------------------------------------
// Belief pulse synthesis
// ---------------------------------------------------------------------------

const buildBeliefPulse = (
  account: AccountPlan | null,
  enrichment: AccountEnrichmentPayload | undefined,
  topPlay: Play | null,
  personaLabelLookup: Record<string, string>
): PulseEntry[] => {
  if (!account) {
    return [
      {
        title: "No belief telemetry yet",
        tone: "neutral",
        lines: [
          {
            label: "Status",
            value: "Run Account Plan to generate belief insights for this account.",
          },
        ],
      },
    ];
  }

  const steps = account.prediction?.journey?.steps || [];
  const requirements = enrichment?.persona_requirements || [];
  const personaEngagements = account.execution?.persona_engagements || [];
  const labelForPersona = (pid: string | null | undefined) => {
    if (!pid) return "Unknown persona";
    if (personaLabelLookup[pid]) return personaLabelLookup[pid];
    const matchFromReq = requirements.find((req) => req.persona_id === pid);
    if (matchFromReq?.persona_label) return matchFromReq.persona_label;
    for (const path of account.prediction?.persona_paths || []) {
      const persona = path.personas.find((p) => p.id === pid);
      if (persona?.label) return persona.label;
    }
    return pid;
  };
  const describePersonaEntry = (entry?: any) => {
    if (!entry) return null;
    return (
      entry.persona_label ||
      labelForPersona(entry.persona_id || (entry as any).persona)
    );
  };
  const getEngagementForPersona = (personaId?: string | null) =>
    personaEngagements.find((entry) => entry.persona_id === personaId);

  const riskPersona = requirements
    .slice()
    .sort(
      (a, b) =>
        (b.expected_in_deal_pct || 0) - (a.expected_in_deal_pct || 0)
    )
    .find((req) => !req.matched_people || req.matched_people.length === 0);

  const accelerationStep = steps.find((step) => {
    const bucket = (step.bucket || "").toLowerCase();
    return (
      bucket === "near_path" ||
      bucket === "jump_ahead" ||
      bucket === "skip_hit"
    );
  });

  const expectedNext =
    account.thesis?.expected_next_personas?.[0] ||
    account.prediction?.expected_next_personas?.[0] ||
    account.prediction?.expected_next?.[0] ||
    null;
  const expectedNextProbability =
    typeof (expectedNext as any)?.probability === "number"
      ? (expectedNext as any).probability
      : typeof (expectedNext as any)?.prob === "number"
      ? (expectedNext as any).prob
      : null;
  const expectedNextLabel = describePersonaEntry(expectedNext);
  const topRecommendedPlay = topPlay;

  const progressBuckets = new Set(["on_path", "near_path", "jump_ahead", "skip_hit"]);
  const lastProgressIndex = (() => {
    for (let idx = steps.length - 1; idx >= 0; idx -= 1) {
      const bucket = (steps[idx].bucket || "").toLowerCase();
      if (progressBuckets.has(bucket)) {
        return idx;
      }
    }
    return -1;
  })();
  const stepsSinceProgress =
    lastProgressIndex === -1 ? steps.length : Math.max(0, steps.length - lastProgressIndex - 1);
  const hasRecentMovement = stepsSinceProgress <= 1;

  const pulses: PulseEntry[] = [];

  if (riskPersona) {
    const personaDisplay = labelForPersona(riskPersona.persona_id);
    const riskEngagement = getEngagementForPersona(riskPersona.persona_id);
    const lastTouchDays = riskEngagement?.engagement_snapshot?.last_touch_days ?? null;
    const touchDensity = riskEngagement?.engagement_snapshot?.touch_density_per_week ?? null;
    const riskWhyBits = [
      lastTouchDays !== null && lastTouchDays !== undefined ? `Last touch ${fmtDays(lastTouchDays)} ago` : null,
      touchDensity ? `${touchDensity.toFixed(1)} touches/wk policy` : null,
    ].filter(Boolean);
    pulses.push({
      title: "🚨 Critical Belief Risk",
      tone: "critical",
      lines: [
        {
          label: "Persona",
          value: `${personaDisplay} · No mapped champion yet`,
        },
        {
          label: "Risk",
          value: `High — ${fmtPercent(
            typeof riskPersona.expected_in_deal === "number"
              ? riskPersona.expected_in_deal
              : (riskPersona.expected_in_deal_pct || 0) / 100,
            true
          )} of wins depend on this persona.`,
        },
        {
          label: "Why",
          value: riskWhyBits.length ? riskWhyBits.join(" · ") : "No engagement telemetry yet for this persona.",
        },
        {
          label: "Action",
          value: topRecommendedPlay
            ? `Map a ${personaDisplay} contact and trigger ${
                topRecommendedPlay.name || topRecommendedPlay.asset_label || "the top play"
              } via ${topRecommendedPlay.channel_label || "primary channel"}.`
            : `Map a ${personaDisplay} contact and assign first-touch owner.`,
        },
      ],
    });
  }

  if (accelerationStep?.observed) {
    const predictedNextLabel = accelerationStep.predicted?.length
      ? labelForPersona(accelerationStep.predicted[0])
      : "model expectation";
    const bucketLabel = titleize(accelerationStep.bucket) || "Journey update";
    const nextPersonaLabel = expectedNextLabel;
    pulses.push({
      title: "⚡ Surprising Acceleration",
      tone: "positive",
      lines: [
        {
          label: "Persona",
          value: `${labelForPersona(accelerationStep.observed)} jumped ahead of ${predictedNextLabel}.`,
        },
        {
          label: "Trigger",
          value: `${bucketLabel}${typeof accelerationStep.t === "number" ? ` at t=${accelerationStep.t}` : ""}`,
        },
        {
          label: "Impact",
          value: nextPersonaLabel
            ? `Next priority: ${nextPersonaLabel}. Re-sequence conversion focus accordingly.`
            : "Re-sequence remaining personas to capitalize on the signal.",
        },
      ],
    });
  }

  if (expectedNext) {
    const nextConfidence =
      expectedNextProbability !== null
        ? fmtPercent(expectedNextProbability, true)
        : null;
    pulses.push({
      title: "🟣 Hidden Opportunity (Latent)",
      tone: "neutral",
      lines: [
        {
          label: "Persona",
          value: `${expectedNextLabel || "Priority persona"} is the next best believer to activate.`,
        },
        {
          label: "Why it matters",
          value:
            expectedNext.reason ||
            (expectedNext.journey_stage && nextConfidence
              ? `Model expects them to reach ${expectedNext.journey_stage} (${nextConfidence})`
              : nextConfidence
              ? `Model confidence ${nextConfidence}`
              : "Model flagged this persona based on recent signals."),
        },
      ],
    });
  }

  if (topRecommendedPlay) {
    pulses.push({
      title: "🟢 Highest ROI Move Right Now",
      tone: "positive",
      lines: [
        {
          label: "Play",
          value: topRecommendedPlay.name || topRecommendedPlay.asset_label || "Recommended play",
        },
        {
          label: "Channel",
          value: topRecommendedPlay.channel_label || "Preferred outreach channel",
        },
        {
          label: "Expected Impact",
          value: `${fmtBasisPoints(topRecommendedPlay.expected_delta_bp)} in ${fmtDays(
            topRecommendedPlay.avg_duration_days
          )}`,
        },
      ],
    });
  }

  pulses.push({
    title: "🔴 Likely Stall in Current Tactic",
    tone: "warning",
    lines: [
      {
        label: "Status",
        value: hasRecentMovement
          ? "On-path momentum detected within the last step."
          : stepsSinceProgress > 0
          ? `No on-path signals for ${stepsSinceProgress} logged step${
              stepsSinceProgress === 1 ? "" : "s"
            }.`
          : "Awaiting first on-path signal.",
      },
      {
        label: "Recommended",
        value: topRecommendedPlay
          ? `Queue ${topRecommendedPlay.name || topRecommendedPlay.asset_label || "top play"} via ${
              topRecommendedPlay.channel_label || "primary channel"
            }.`
          : describePersonaEntry(expectedNext)
          ? `Increase touches for ${describePersonaEntry(expectedNext)}`
          : "Increase touches for unmatched personas.",
      },
    ],
  });

  return pulses.slice(0, 5);
};

// ---------------------------------------------------------------------------
// Rendering helpers
// ---------------------------------------------------------------------------

const renderPlaysSummary = (plays: Play[] | undefined) => {
  const items = dedupePlays(plays || []).sort(
    (a, b) => (b.expected_delta_bp || 0) - (a.expected_delta_bp || 0)
  );
  if (!items.length) {
    return (
      <Typography variant="body2" color="text.secondary">
        Plays will populate once engagements are processed for this account.
      </Typography>
    );
  }

  return (
    <TableContainer component={Paper} variant="outlined">
      <Table size="small">
        <TableHead>
          <TableRow>
            <TableCell>Play</TableCell>
            <TableCell>Channel</TableCell>
            <TableCell align="right">Δ (bps)</TableCell>
            <TableCell align="right">Confidence</TableCell>
            <TableCell align="right">Avg Time</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {items.map((play) => {
            const assetName = play.asset?.name || play.asset_label;
            const channelName =
              play.channel?.name || play.channel_label || play.channel?.channel_type_label;
            const name = play.name || assetName || channelName || "Recommended play";
            const tooltipLines: string[] = [];
            if (play.asset_type_label) tooltipLines.push(`Asset: ${play.asset_type_label}`);
            if (channelName) tooltipLines.push(`Channel: ${channelName}`);
            const tooltip =
              tooltipLines.length > 0 ? (
                <Box>
                  {tooltipLines.map((line, idx) => (
                    <Typography key={`${name}-detail-${idx}`} variant="body2">
                      {line}
                    </Typography>
                  ))}
                </Box>
              ) : null;
            return (
              <TableRow key={play.id || play.name || play.asset_label || name}>
                <TableCell>
                  <Stack direction="row" spacing={1} alignItems="center">
                    <Typography variant="body2" fontWeight={600}>
                      {name}
                    </Typography>
                    {tooltip && (
                      <Tooltip title={tooltip} arrow>
                        <IconButton size="small">
                          <InfoOutlinedIcon fontSize="inherit" />
                        </IconButton>
                      </Tooltip>
                    )}
                  </Stack>
                </TableCell>
                <TableCell>{channelName || "—"}</TableCell>
                <TableCell align="right">
                  {typeof play.expected_delta_bp === "number"
                    ? fmtBasisPoints(play.expected_delta_bp)
                    : "—"}
                </TableCell>
                <TableCell align="right">{fmtPercent(play.avg_confidence, true)}</TableCell>
                <TableCell align="right">
                  {typeof play.avg_duration_days === "number" ? fmtDays(play.avg_duration_days) : "—"}
                </TableCell>
              </TableRow>
            );
          })}
        </TableBody>
      </Table>
    </TableContainer>
  );
};

const renderConversionSequence = (entries: ConversionSequenceEntry[] | undefined) => {
  if (!entries?.length) {
    return (
      <Typography variant="body2" color="text.secondary">
        Conversion sequence will populate once replay data is available for this account.
      </Typography>
    );
  }

  return (
    <Stack spacing={1.5}>
      {entries.map((entry) => {
        const chips: Array<{ label: string; outlined?: boolean }> = [];
        if (typeof entry.expected_delta_bp === "number") {
          chips.push({ label: `Δ ${fmtBasisPoints(entry.expected_delta_bp)}` });
        }
        if (typeof entry.avg_confidence === "number") {
          chips.push({
            label: `Confidence ${fmtPercent(entry.avg_confidence, true)}`,
            outlined: true,
          });
        }
        if (typeof entry.avg_belief_conversion === "number") {
          chips.push({
            label: `Belief ${fmtPercent(entry.avg_belief_conversion, true)}`,
            outlined: true,
          });
        }
        if (typeof entry.effort_pct === "number") {
          chips.push({ label: `Effort ${fmtPercent(entry.effort_pct, true)}`, outlined: true });
        }
        if (typeof entry.time_to_effect_days === "number") {
          chips.push({ label: `Time ${fmtDays(entry.time_to_effect_days)}`, outlined: true });
        }
        const description =
          entry.belief_transition_meta?.narrative ||
          entry.belief_transition_meta?.pain?.label ||
          entry.belief_transition_meta?.problem?.label ||
          entry.belief_transition_meta?.resolution?.label ||
          "Belief progression";
        const stageLabel = entry.belief_transition_meta?.stage_label
          ? `${entry.belief_transition_meta?.stage_label}: `
          : "";
        const tooltipLines: string[] = [];
        if (entry.expected_outcome_summary) {
          tooltipLines.push(entry.expected_outcome_summary);
        }
        if (stageLabel || description) {
          tooltipLines.push(`${stageLabel}${description}`);
        }
        if (entry.segment_summary) {
          tooltipLines.push(entry.segment_summary);
        }
        segmentChipsFromFilters(entry.segment_filters).forEach(([key, value]) => {
          tooltipLines.push(`${titleize(key)}: ${value}`);
        });
        const tooltip =
          tooltipLines.length > 0 ? (
            <Box>
              {tooltipLines.map((line, idx) => (
                <Typography key={`sequence-${entry.timeline_index}-detail-${idx}`} variant="body2">
                  {line}
                </Typography>
              ))}
            </Box>
          ) : null;
        return (
          <Paper key={`sequence-${entry.timeline_index}`} variant="outlined" sx={{ p: 2 }}>
            <Stack
              direction={{ xs: "column", md: "row" }}
              spacing={1}
              alignItems={{ xs: "flex-start", md: "center" }}
            >
              <Chip
                size="small"
                color="primary"
                label={entry.timeline_label || `T+${entry.timeline_index}`}
              />
              <Stack direction="row" spacing={1} alignItems="center">
                <Typography variant="subtitle2">
                  Step {entry.timeline_index + 1}:{" "}
                  {entry.persona_descriptor || "Target persona"}
                </Typography>
                {tooltip && (
                  <Tooltip title={tooltip} arrow>
                    <IconButton size="small">
                      <InfoOutlinedIcon fontSize="inherit" />
                    </IconButton>
                  </Tooltip>
                )}
              </Stack>
            </Stack>
            <Typography variant="body2" sx={{ mt: 0.5 }}>
              {stageLabel}
              {description}
            </Typography>
            {chips.length ? (
              <Stack direction="row" spacing={0.75} sx={{ flexWrap: "wrap", mt: 1 }}>
                {chips.map((chip, idx) => (
                  <Chip
                    key={`${entry.timeline_index}-chip-${idx}`}
                    label={chip.label}
                    size="small"
                    variant={chip.outlined ? "outlined" : "filled"}
                  />
                ))}
              </Stack>
            ) : null}
          </Paper>
        );
      })}
    </Stack>
  );
};

const renderPulseCard = (pulse: PulseEntry) => {
  const borderColor =
    pulse.tone === "critical"
      ? "error.light"
      : pulse.tone === "positive"
      ? "success.light"
      : pulse.tone === "warning"
      ? "warning.light"
      : "divider";
  const background =
    pulse.tone === "critical"
      ? "rgba(211,47,47,0.05)"
      : pulse.tone === "positive"
      ? "rgba(56,142,60,0.05)"
      : pulse.tone === "warning"
      ? "rgba(237,108,2,0.05)"
      : "background.paper";

  return (
    <Paper
      key={pulse.title}
      variant="outlined"
      sx={{ p: 2, borderColor, backgroundColor: background }}
    >
      <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 0.75 }}>
        {pulse.title}
      </Typography>
      <Stack spacing={0.5}>
        {pulse.lines.map((line) => (
          <Typography key={`${pulse.title}-${line.label}`} variant="body2">
            <strong>{line.label}:</strong> {line.value}
          </Typography>
        ))}
      </Stack>
    </Paper>
  );
};

type FourQuestionsAccountProps = {
  account: AccountPlan;
  personaMatches: Record<string, string[]>;
  canonicalTypicalPath: TypicalPathStep[];
};

const FourQuestionsAccount: React.FC<FourQuestionsAccountProps> = ({
  account,
  personaMatches,
  canonicalTypicalPath,
}) => {
  const primaryPath = account.prediction?.persona_paths?.[0] || null;
  const plays = account.execution?.plays || [];
  const thesis = account.thesis;
  const personaLikelihoods = thesis?.persona_likelihoods ?? [];
  const personLikelihoods = thesis?.person_likelihoods ?? [];
  const expectedNextPersonas =
    thesis?.expected_next_personas ?? account.prediction?.expected_next_personas ?? [];
  const personaCadence = account.execution?.persona_engagements || [];

  const heading = (label: string, helper: string) => (
    <Stack direction="row" spacing={0.75} alignItems="center">
      <Typography variant="h6">{label}</Typography>
      <Tooltip title={helper}>
        <InfoOutlinedIcon fontSize="small" color="action" />
      </Tooltip>
    </Stack>
  );

  const formatChipLabel = (band?: string | null, label?: string | null, prob?: number | null) => {
    const parts: string[] = [];
    if (band) parts.push(titleize(band));
    parts.push(label || "Persona");
    if (typeof prob === "number") {
      parts.push(fmtPercent(prob, true));
    }
    return parts.join(" · ");
  };

  return (
    <Stack spacing={3}>
      <Box>
        {heading(
          "1. Who is likely to be involved?",
          "Posterior committee personas plus the real people mapped to each role."
        )}
        <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
          The personas most likely to participate in this account's win committee.
        </Typography>
        {personaLikelihoods.length ? (
          <Stack direction="row" spacing={0.5} sx={{ flexWrap: "wrap", mb: 1 }}>
            {personaLikelihoods.slice(0, 10).map((persona) => (
              <Chip
                key={`persona-likelihood-${persona.persona_id}`}
                size="small"
                label={formatChipLabel(persona.band, persona.persona_label, persona.probability)}
              />
            ))}
          </Stack>
        ) : null}
        <Typography variant="body2" color="text.secondary">
          People mapped to these personas (highest probability first):
        </Typography>
        {personLikelihoods.length ? (
          <Stack direction="row" spacing={0.5} sx={{ flexWrap: "wrap", mt: 1 }}>
            {personLikelihoods.slice(0, 10).map((person) => (
              <Chip
                key={`person-likelihood-${person.person_id || person.display_name}`}
                variant="outlined"
                size="small"
                label={formatChipLabel(person.role_band, person.display_name, person.probability)}
              />
            ))}
          </Stack>
        ) : (
          <Stack spacing={0.5} sx={{ mt: 1 }}>
            {(primaryPath?.personas || []).map((persona) => {
              const pid = persona.id || "";
              const people = personaMatches[pid] || [];
              if (!people.length) return null;
              return (
                <Stack key={`${pid}-people`} direction="row" spacing={0.5} alignItems="center">
                  <Chip label={persona.label || pid || "Persona"} size="small" />
                  <Stack direction="row" spacing={0.5} sx={{ flexWrap: "wrap" }}>
                    {people.map((name) => (
                      <Chip key={`${pid}-${name}`} label={name} size="small" variant="outlined" />
                    ))}
                  </Stack>
                </Stack>
              );
            })}
          </Stack>
        )}
        <Box sx={{ mt: 1.5 }}>
          {primaryPath ? (
            <>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 0.5 }}>
                Conversion path context:
              </Typography>
              {renderPersonaPath(primaryPath)}
            </>
          ) : (
            renderTypicalPathSteps(
              canonicalTypicalPath,
              "Using canonical path until this account accumulates journey telemetry."
            )
          )}
        </Box>
        {expectedNextPersonas.length ? (
          <Box sx={{ mt: 2 }}>
            <Typography variant="subtitle2" sx={{ mb: 0.5 }}>
              Expected next personas to activate
            </Typography>
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Persona</TableCell>
                  <TableCell align="right">Probability</TableCell>
                  <TableCell>Stage</TableCell>
                  <TableCell>Why</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {expectedNextPersonas.slice(0, 6).map((entry) => (
                  <TableRow key={`expected-next-${entry.persona_id}`}>
                    <TableCell>{entry.persona_label || entry.persona_id}</TableCell>
                    <TableCell align="right">{fmtPercent(entry.probability, true)}</TableCell>
                    <TableCell>{entry.journey_stage || "Journey"}</TableCell>
                    <TableCell>{entry.reason || "Belief graph projection"}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </Box>
        ) : null}
      </Box>

      <Divider />

      <Box>
        {heading(
          "2. What is their current belief level?",
          "Belief level, dominant phase, and committee probability per persona."
        )}
        {personaCadence.length ? (
          <Table size="small" sx={{ mt: 1 }}>
            <TableHead>
              <TableRow>
                <TableCell>Persona</TableCell>
                <TableCell align="right">Belief</TableCell>
                <TableCell>Phase</TableCell>
                <TableCell align="right">Committee</TableCell>
                <TableCell align="right">Cadence</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {personaCadence.map((persona) => (
                <TableRow key={persona.persona_id}>
                  <TableCell>{persona.persona_label || persona.persona_id}</TableCell>
                  <TableCell align="right">
                    {persona.belief_level != null ? (
                      <Chip
                        size="small"
                        variant="outlined"
                        label={`${fmtPercent(persona.belief_level, true)} · ${titleize(
                          persona.belief_band
                        )}`}
                      />
                    ) : (
                      "—"
                    )}
                  </TableCell>
                  <TableCell>
                    {persona.dominant_phase ? (
                      <Tooltip
                        title={
                          persona.phase_probs
                            ? Object.entries(persona.phase_probs)
                                .map(([phase, prob]) => `${phase}: ${fmtPercent(prob, true)}`)
                                .join(" · ")
                            : ""
                        }
                      >
                        <Chip size="small" label={persona.dominant_phase} />
                      </Tooltip>
                    ) : (
                      "—"
                    )}
                  </TableCell>
                  <TableCell align="right">
                    {persona.committee_probability != null
                      ? fmtPercent(persona.committee_probability, true)
                      : "—"}
                  </TableCell>
                  <TableCell align="right">
                    {persona.engagement_state
                      ? `${titleize(persona.engagement_state)} · ${formatFrequency(
                          persona.base_frequency_per_week
                        )}`
                      : "—"}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        ) : (
          <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
            Belief table will appear once persona cadences are generated for this account.
          </Typography>
        )}
      </Box>

      <Divider />

      <Box>
        {heading(
          "3. How often will we engage them?",
          "Cadence adapts using belief, fatigue, and observed responses."
        )}
        <Typography variant="body2" color="text.secondary">
          U-shaped cadence tuned to each persona's belief state and fatigue level.
        </Typography>
        <Box sx={{ mt: 1.5 }}>
          <PersonaEngagementCadence
            engagements={account.execution?.persona_engagements}
            personaMatches={personaMatches}
          />
        </Box>
      </Box>

      <Divider />

      <Box>
        {heading(
          "4. What assets & channels will we use?",
          "Highest-impact plays for this account's belief stage."
        )}
        {renderPlaysSummary(plays)}
      </Box>
    </Stack>
  );
};

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

const AccountPlanPage: React.FC = () => {
  const token =
    typeof window !== "undefined" ? localStorage.getItem("token") : null;
  const { accountId } = useParams<{ accountId?: string }>();
  const navigate = useNavigate();

  const [products, setProducts] = useState<Array<{ id: string; name: string }>>([]);
  const [selectedProductId, setSelectedProductId] = useState<string>("");
  const [plan, setPlan] = useState<MarketingPlan | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [enrichmentLoading, setEnrichmentLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [enrichmentMap, setEnrichmentMap] = useState<
    Record<string, AccountEnrichmentPayload>
  >({});

  const canonicalTypicalPath = useMemo(() => {
    if (!plan) return [];
    return (
      plan.canonical_journey?.typical_path ??
      plan.product_insights?.journey_structure?.typical_path ??
      plan.summary?.typical_path ??
      []
    );
  }, [plan]);

  useEffect(() => {
    if (!token) return;
    (async () => {
      try {
        const meRes = await fetch("http://localhost:8000/me", {
          headers: { Authorization: `Bearer ${token}` },
        });
        if (!meRes.ok) throw new Error(await meRes.text());
        const me = await meRes.json();

        const prodRes = await fetch(
          `http://localhost:8000/get-products/${me.company_id}`,
          { headers: { Authorization: `Bearer ${token}` } }
        );
        if (!prodRes.ok) throw new Error(await prodRes.text());
        const prodData = await prodRes.json();
        const list = prodData.products ?? [];
        if (!list.length) {
          setError("No products found. Run Value Prop to create the first product.");
          return;
        }
        setProducts(list);
        setSelectedProductId((prev) => prev || list[0].id);
      } catch (err: unknown) {
        console.error(err);
        const message = err instanceof Error ? err.message : "Failed to load products.";
        setError(message);
      }
    })();
  }, [token]);

  useEffect(() => {
    if (!selectedProductId || !token) return;
    setLoading(true);
    setError(null);
    (async () => {
      try {
        const res = await fetch(
          `http://localhost:8000/get-comprehensive-execution-plan/${selectedProductId}`,
          { headers: { Authorization: `Bearer ${token}` } }
        );
        if (!res.ok) throw new Error(await res.text());
        const data: MarketingPlan = await res.json();
        setPlan(data);
      } catch (err: unknown) {
        console.error(err);
        const message = err instanceof Error ? err.message : "Failed to load account plan.";
        setError(message);
        setPlan(null);
      } finally {
        setLoading(false);
      }
    })();
  }, [selectedProductId, token]);

  useEffect(() => {
    if (!plan || !token || !selectedProductId) {
      setEnrichmentMap({});
      return;
    }
    let cancelled = false;
    const headers = { Authorization: `Bearer ${token}` };
    setEnrichmentLoading(true);
    Promise.all(
      plan.accounts.map(async (account) => {
        try {
          const res = await fetch(
            `http://localhost:8000/accounts/${account.account_id}/enrichment?product_id=${encodeURIComponent(
              selectedProductId
            )}`,
            { headers }
          );
          if (!res.ok) throw new Error(await res.text());
          const payload: AccountEnrichmentPayload = await res.json();
          return [account.account_id, payload] as const;
        } catch (err) {
          console.warn("Failed to load enrichment for account", account.account_id, err);
          return [account.account_id, undefined] as const;
        }
      })
    )
      .then((entries) => {
        if (cancelled) return;
        const next: Record<string, AccountEnrichmentPayload> = {};
        entries.forEach(([id, payload]) => {
          if (payload) {
            next[id] = payload;
          }
        });
        setEnrichmentMap(next);
      })
      .finally(() => {
        if (!cancelled) setEnrichmentLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [plan, selectedProductId, token]);

  const accounts = plan?.accounts ?? [];
  const activeAccount = accountId
    ? accounts.find((acct) => acct.account_id === accountId) || null
    : null;
  const activeEnrichment = accountId
    ? enrichmentMap[accountId]
    : undefined;

  const topPlay =
    activeAccount?.execution?.plays && activeAccount.execution.plays.length
      ? dedupePlays(activeAccount.execution.plays)
          .sort(
            (a, b) =>
              (b.expected_delta_bp || 0) - (a.expected_delta_bp || 0)
          )[0]
      : null;

  const personaLabelLookup = useMemo(() => {
    const lookup: Record<string, string> = {};
    plan?.summary?.canonical_persona_path?.forEach((persona) => {
      const pid = persona?.id || persona?.persona_id;
      if (pid && persona?.label) {
        lookup[pid] = persona.label;
      }
    });
    activeAccount?.prediction?.persona_paths?.forEach((path) => {
      (path.personas || []).forEach((persona) => {
        if (persona.id && persona.label) {
          lookup[persona.id] = persona.label;
        }
      });
    });
    return lookup;
  }, [plan, activeAccount]);

  const personaMatches = useMemo(() => {
    const matches: Record<string, string[]> = {};
    const pushName = (pid: string | undefined | null, name?: string | null) => {
      if (!pid || !name) return;
      if (!matches[pid]) matches[pid] = [];
      if (!matches[pid].includes(name)) {
        matches[pid].push(name);
      }
    };
    activeEnrichment?.persona_requirements?.forEach((req) => {
      const pid = req.persona_id;
      (req.matched_people || []).forEach((person) => {
        pushName(pid, person.person_name || person.display_name);
      });
      (req.people_names || []).forEach((name) => pushName(pid, name));
    });
    (activeAccount?.prediction?.persona_paths || []).forEach((path) => {
      (path.personas || []).forEach((persona) => {
        (persona.people_names || []).forEach((name) =>
          pushName(persona.id, name)
        );
      });
    });
    return matches;
  }, [activeEnrichment, activeAccount]);

  const beliefPulse = useMemo(
    () => buildBeliefPulse(activeAccount, activeEnrichment, topPlay, personaLabelLookup),
    [activeAccount, activeEnrichment, topPlay, personaLabelLookup]
  );

  const accountUpdatedAt =
    activeEnrichment?.blueprint_generated_at ||
    plan?.summary.generated_at ||
    plan?.generated_at;

  if (loading) {
    return (
      <Box sx={{ display: "flex", justifyContent: "center", mt: 8 }}>
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Box sx={{ p: 3 }}>
        <Alert severity="error">{error}</Alert>
      </Box>
    );
  }

  return (
    <Box sx={{ p: 3 }}>
      <Stack
        direction={{ xs: "column", sm: "row" }}
        alignItems={{ xs: "flex-start", sm: "center" }}
        spacing={2}
        sx={{ mb: 3 }}
      >
        <Typography variant="h5">Account Plan</Typography>
        <Box sx={{ flexGrow: 1 }} />
        {products.length > 0 && (
          <FormControl size="small" sx={{ minWidth: 220 }}>
            <InputLabel id="account-plan-product-label">Product</InputLabel>
            <Select
              labelId="account-plan-product-label"
              label="Product"
              value={selectedProductId}
              onChange={(event) => {
                navigate("/account-plan");
                setSelectedProductId(String(event.target.value));
              }}
            >
              {products.map((product) => (
                <MenuItem key={product.id} value={product.id}>
                  {product.name || product.id}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        )}
      </Stack>

      {accountId && activeAccount && (
        <Breadcrumbs sx={{ mb: 2 }}>
          <MUILink component={RouterLink} color="inherit" to="/account-plan">
            Account Plan
          </MUILink>
          <Typography color="text.primary">{activeAccount.account_name}</Typography>
        </Breadcrumbs>
      )}

      <Typography variant="subtitle1" sx={{ mb: 1.5 }}>
        Target Accounts
      </Typography>

      <Grid container spacing={2} sx={{ mb: 4 }}>
        {accounts.map((acct) => {
          const enrichment = enrichmentMap[acct.account_id];
          const coverage =
            enrichment?.summary?.coverage_pct ??
            acct.enrichment?.summary?.coverage_pct ??
            0;
          const matchedPeople =
            enrichment?.summary?.matched_people ??
            acct.enrichment?.summary?.matched_people ??
            0;
          const required =
            enrichment?.summary?.required_personas ??
            acct.enrichment?.summary?.required_personas ??
            undefined;
          const hasEngagements =
            (acct.prediction?.journey?.steps?.length || 0) > 0;

          const metaChips = uniqueStrings(
            Object.entries(acct.meta || {}).map(
              ([key, value]) =>
                `${key.replace(/[_-]/g, " ")}: ${String(value)}`
            )
          );

          const isSelected = accountId === acct.account_id;

          return (
            <Grid key={acct.account_id} size={{ xs: 12, md: 6, lg: 4 }}>
              <Card
                variant="outlined"
                sx={{
                  borderColor: isSelected ? "primary.main" : undefined,
                  boxShadow: isSelected ? 2 : undefined,
                }}
              >
                <CardActionArea
                  onClick={() => navigate(`/account-plan/${acct.account_id}`)}
                  sx={{ alignItems: "stretch" }}
                >
                  <CardContent>
                    <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
                      {acct.account_name}
                    </Typography>
                    <Stack direction="row" spacing={0.75} sx={{ flexWrap: "wrap", mt: 1 }}>
                      {metaChips.slice(0, 5).map((chip) => (
                        <Chip key={`${acct.account_id}-${chip}`} label={chip} size="small" />
                      ))}
                    </Stack>
                    <Stack direction="row" spacing={1.5} sx={{ mt: 2 }}>
                      <Stack spacing={0.25}>
                        <Typography variant="caption" color="text.secondary">
                          Enrichment
                        </Typography>
                        <Typography variant="body2">
                          {enrichmentLoading && !enrichment
                            ? "Loading…"
                            : fmtPercent(coverage)}
                        </Typography>
                        {required !== undefined ? (
                          <Typography variant="caption" color="text.secondary">
                            {matchedPeople}/{required} personas matched
                          </Typography>
                        ) : null}
                      </Stack>
                      <Stack spacing={0.25}>
                        <Typography variant="caption" color="text.secondary">
                          Engagements
                        </Typography>
                        <Typography variant="body2">
                          {hasEngagements ? "Live" : "No engagements"}
                        </Typography>
                      </Stack>
                    </Stack>
                  </CardContent>
                </CardActionArea>
              </Card>
            </Grid>
          );
        })}

        {!accounts.length && (
          <Grid size={{ xs: 12 }}>
            <Alert severity="info">
              No target accounts yet. Add accounts from the Target Accounts page to generate
              Account Plans.
            </Alert>
          </Grid>
        )}
      </Grid>

      {accountId && !activeAccount && (
        <Alert severity="warning">
          Could not locate this account.{" "}
          <Button
            component={RouterLink}
            to="/account-plan"
            size="small"
            sx={{ textTransform: "none" }}
          >
            Back to accounts
          </Button>
        </Alert>
      )}

      {accountId && activeAccount && (
        <Stack spacing={3}>
          <Card variant="outlined">
            <CardContent>
              <Stack
                direction={{ xs: "column", md: "row" }}
                spacing={1.5}
                alignItems={{ xs: "flex-start", md: "center" }}
              >
                <Typography variant="h6" sx={{ flexGrow: 1 }}>
                  Belief Pulse & Recommended Next Action
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  {formatRelativeTime(accountUpdatedAt)}
                </Typography>
              </Stack>
              <Stack spacing={1.5} sx={{ mt: 2 }}>
                {beliefPulse.map(renderPulseCard)}
              </Stack>
            </CardContent>
          </Card>

          <Card variant="outlined">
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2 }}>
                Detailed Account Strategy
              </Typography>
              <FourQuestionsAccount
                account={activeAccount}
                personaMatches={personaMatches}
                canonicalTypicalPath={canonicalTypicalPath}
              />
              <Divider sx={{ my: 3 }} />
              <Box>
                <Typography variant="subtitle2" sx={{ mb: 1 }}>
                  Conversion Sequence Blueprint
                </Typography>
                {renderConversionSequence(activeAccount.execution?.conversion_sequence)}
              </Box>
              <Divider sx={{ my: 3 }} />
              <Box>
                <Typography variant="subtitle2" sx={{ mb: 1 }}>
                  Asset Cadence Table
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Filterable mix of assets & channels aligned to funnel stages and campaign
                  themes.
                </Typography>
                <Box sx={{ mt: 1.5 }}>
                  <AssetCadenceTable
                    rows={activeAccount.execution?.asset_cadence}
                    personaLabelLookup={personaLabelLookup}
                    enableFilters
                  />
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Stack>
      )}
    </Box>
  );
};

export default AccountPlanPage;
