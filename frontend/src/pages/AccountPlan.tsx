import React, { useCallback, useEffect, useMemo, useState } from "react";
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
  LinearProgress,
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
import type { AccountPlanContract } from "../types/apiContracts";

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

type EntryPointPlay = {
  play_id?: string | null;
  stage?: string | null;
  concern?: string | null;
  asset?: string | null;
  channel?: string | null;
  mode?: string | null;
  expected_delta_bp?: number | null;
  confidence?: number | null;
};

type EntryPoint = {
  persona_id: string;
  persona_label?: string | null;
  rank?: number | null;
  entry_score?: number | null;
  combined_perceptibility?: number | null;
  combined_proximity?: number | null;
  graph_perceptibility?: number | null;
  graph_proximity?: number | null;
  intervention_reach?: number | null;
  top_people?: PersonaEngagementPerson[];
  top_plays?: EntryPointPlay[];
};

type KeystonePersona = {
  persona_id: string;
  persona_label: string;
  wolves_score?: number | null;
  wolves_delta_bp?: number | null;
  wolves_involvement_rate?: number | null;
  wolves_blocker_rate?: number | null;
  wolves_sample_size?: number | null;
  is_new_persona?: boolean;
  persona_source?: string | null;
  priority_score?: number | null;
  journey_phase?: string | null;
  matched_people_count?: number | null;
  top_people?: PersonaEngagementPerson[];
  chain_effect?: string | null;
  chain_targets?: string[];
  chain_share?: number | null;
  sample_story?: string | null;
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
  entry_points?: EntryPoint[];
  keystone_personas?: KeystonePersona[];
  wolves_metrics_updated_at?: string | null;
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
    entry_points?: EntryPoint[];
  };
  enrichment?: {
    summary?: {
      coverage_pct?: number;
      coverage_ratio?: number;
      matched_people?: number;
      required_personas?: number;
    };
  };
  interventions?: AccountIntervention[];
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

type AccountIntervention = {
  id: string;
  persona_label?: string | null;
  persona_descriptor?: string | null;
  stage_label?: string | null;
  concern_theme?: string | null;
  belief_lift_bp?: number | null;
  coverage_score?: number | null;
  coverage_state?: "strong" | "steady" | "weak" | null;
  needs_net_new?: boolean;
  current_modality?: {
    format_label?: string | null;
    channel_label?: string | null;
  } | null;
};

type EnrichmentPersonaCandidate = {
  label: string;
  normalized_label: string;
  account_occurrences: number;
  global_account_count?: number;
  wolves_score?: number | null;
  wolves_delta_bp?: number | null;
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
  persona_candidates?: EnrichmentPersonaCandidate[];
};

type TargetAccountLookupEntry = {
  id: string;
  account_name: string;
  industry?: string;
  revenue_range?: string;
  employee_range?: string;
  funding_stage?: string;
  geography?: string;
  deal_status?: string;
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

const formatList = (items: string[], conjunction = "and") => {
  const safeItems = items.filter(Boolean);
  if (!safeItems.length) return "";
  if (safeItems.length === 1) return safeItems[0];
  if (safeItems.length === 2) return `${safeItems[0]} ${conjunction} ${safeItems[1]}`;
  return `${safeItems.slice(0, -1).join(", ")}, ${conjunction} ${safeItems[safeItems.length - 1]}`;
};

const TARGET_ACCOUNT_META_FIELDS: Array<
  keyof Omit<TargetAccountLookupEntry, "id" | "account_name">
> = ["industry", "revenue_range", "employee_range", "funding_stage", "geography"];

const getAccountDisplayName = (
  account?: AccountPlan | null,
  lookup?: TargetAccountLookupEntry
) => {
  return (
    lookup?.account_name ||
    account?.account_name ||
    account?.account_id ||
    "Account"
  );
};

const getAccountMetaForChips = (
  account?: AccountPlan,
  lookup?: TargetAccountLookupEntry
): Record<string, unknown> => {
  const baseMeta: Record<string, unknown> = { ...(account?.meta || {}) };
  if (lookup) {
    TARGET_ACCOUNT_META_FIELDS.forEach((field) => {
      const existing = baseMeta[field];
      if (
        lookup[field] &&
        (!existing || (typeof existing === "string" && !existing.trim()))
      ) {
        baseMeta[field] = lookup[field];
      }
    });
  }
  return baseMeta;
};

const normalizePersonaLabel = (label?: string | null) =>
  (label || "").trim().toLowerCase();

const describeKeystoneImpact = (persona?: KeystonePersona | null) => {
  if (!persona) return null;
  if (persona.chain_effect) return persona.chain_effect;
  const chainTargets =
    persona.chain_targets && persona.chain_targets.length
      ? formatList(persona.chain_targets)
      : null;
  if (chainTargets) {
    return persona.chain_share !== null && persona.chain_share !== undefined
      ? `Unlocks ${chainTargets} (${fmtPercent(persona.chain_share, true)} of wins)`
      : `Unlocks ${chainTargets}`;
  }
  if (persona.sample_story) return persona.sample_story;
  if (typeof persona.wolves_delta_bp === "number") {
    return `Keystone impact +${fmtBasisPoints(persona.wolves_delta_bp)} when activated`;
  }
  if (typeof persona.wolves_involvement_rate === "number") {
    return `Appears in ${fmtPercent(persona.wolves_involvement_rate, true)} of wins`;
  }
  return null;
};

const classifyKeystoneRole = (phase?: string | null): string | null => {
  if (!phase) return null;
  const normalized = phase.toLowerCase();
  if (["unaware", "zmot", "discovery", "awareness", "problem"].includes(normalized)) {
    return "Keystone starter";
  }
  if (["evaluation", "pilot", "committee", "mid_funnel", "coalition"].includes(normalized)) {
    return "Bridge to exec";
  }
  if (["pre-close", "decision", "procurement", "customer", "adoption"].includes(normalized)) {
    return "Downstream operator";
  }
  return null;
};

const describeEntryPlay = (play?: EntryPointPlay | null) => {
  if (!play) return null;
  const assetLabel = play.asset || "Asset";
  const channelLabel = play.channel || "Channel";
  const stagePrefix = play.stage ? `${play.stage}: ` : "";
  const deltaLabel =
    typeof play.expected_delta_bp === "number"
      ? ` · Δ ${fmtBasisPoints(play.expected_delta_bp)}`
      : "";
  return `${stagePrefix}${assetLabel} via ${channelLabel}${deltaLabel}`;
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
  personaLabelLookup: Record<string, string>,
  keystonePersonas: KeystonePersona[] = []
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
  const keystoneList =
    (keystonePersonas && keystonePersonas.length ? keystonePersonas : account.keystone_personas) ?? [];
  const keystoneById = new Map<string, KeystonePersona>();
  const keystoneByLabel = new Map<string, KeystonePersona>();
  keystoneList.forEach((persona) => {
    if (persona.persona_id) {
      keystoneById.set(persona.persona_id, persona);
    }
    if (persona.persona_label) {
      const normalized = normalizePersonaLabel(persona.persona_label);
      if (normalized) {
        keystoneByLabel.set(normalized, persona);
      }
    }
  });
  const getKeystoneFor = (entry?: any): KeystonePersona | null => {
    if (!entry) return null;
    if (typeof entry === "string") {
      return (
        keystoneById.get(entry) ||
        keystoneByLabel.get(normalizePersonaLabel(entry)) ||
        null
      );
    }
    const personaId = entry.persona_id || entry.personaId || entry.persona?.id;
    const personaLabel =
      entry.persona_label ||
      entry.persona ||
      entry.persona_focus ||
      entry.label ||
      entry.descriptor;
    return (
      (personaId && keystoneById.get(personaId)) ||
      (personaLabel && keystoneByLabel.get(normalizePersonaLabel(personaLabel))) ||
      null
    );
  };

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
  const expectedNextKeystone = getKeystoneFor(expectedNext);
  const expectedNextNarrative = describeKeystoneImpact(expectedNextKeystone);
  const expectedNextReason = isExpectedNextPersona(expectedNext) ? expectedNext.reason : undefined;
  const expectedNextStage = isExpectedNextPersona(expectedNext)
    ? expectedNext.journey_stage
    : undefined;
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
    const riskKeystone = getKeystoneFor(riskPersona);
    const riskKeystoneNarrative = describeKeystoneImpact(riskKeystone);
    const criticalLines: PulseLine[] = [
      {
        label: "Persona",
        value: `${personaDisplay}${riskKeystone ? " (keystone)" : ""} · No mapped champion yet`,
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
    ];
    if (riskKeystoneNarrative) {
      criticalLines.push({
        label: "Chain effect",
        value: riskKeystoneNarrative,
      });
    }
    pulses.push({
      title: "🚨 Critical Belief Risk",
      tone: "critical",
      lines: criticalLines,
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
    const latentLines: PulseLine[] = [
      {
        label: "Persona",
        value: `${expectedNextLabel || "Priority persona"} is the next best believer to activate${
          expectedNextKeystone ? " (keystone)" : ""
        }.`,
      },
      {
        label: "Why it matters",
        value:
          expectedNextReason ||
          (expectedNextStage && nextConfidence
            ? `Model expects them to reach ${expectedNextStage} (${nextConfidence})`
            : nextConfidence
            ? `Model confidence ${nextConfidence}`
            : "Model flagged this persona based on recent signals."),
      },
    ];
    if (expectedNextNarrative) {
      latentLines.push({
        label: "Chain effect",
        value: expectedNextNarrative,
      });
    }
    pulses.push({
      title: "🟣 Hidden Opportunity (Latent)",
      tone: "neutral",
      lines: latentLines,
    });
  }

  if (topRecommendedPlay) {
    const moveLines: PulseLine[] = [
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
    ];
    if (expectedNextLabel) {
      moveLines.push({
        label: "Target persona",
        value: `${expectedNextLabel}${expectedNextKeystone ? " (keystone)" : ""}`,
      });
    }
    if (expectedNextNarrative) {
      moveLines.push({
        label: "Chain effect",
        value: expectedNextNarrative,
      });
    }
    pulses.push({
      title: "🟢 Highest ROI Move Right Now",
      tone: "positive",
      lines: moveLines,
    });
  }

  const stallLines: PulseLine[] = [
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
        ? `Increase touches for ${describePersonaEntry(expectedNext)}${
            expectedNextKeystone ? " (keystone)" : ""
          }`
        : "Increase touches for unmatched personas.",
    },
  ];
  if (!topRecommendedPlay && expectedNextNarrative) {
    stallLines.push({
      label: "Chain effect",
      value: expectedNextNarrative,
    });
  }
  pulses.push({
    title: "🔴 Likely Stall in Current Tactic",
    tone: "warning",
    lines: stallLines,
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
  keystonePersonas: KeystonePersona[];
  wolvesUpdatedAt?: string | null;
};

const FourQuestionsAccount: React.FC<FourQuestionsAccountProps> = ({
  account,
  personaMatches,
  canonicalTypicalPath,
  keystonePersonas,
  wolvesUpdatedAt,
}) => {
  const primaryPath = account.prediction?.persona_paths?.[0] || null;
  const plays = account.execution?.plays || [];
  const thesis = account.thesis;
  const personaLikelihoods = thesis?.persona_likelihoods ?? [];
  const personLikelihoods = thesis?.person_likelihoods ?? [];
  const expectedNextPersonas =
    thesis?.expected_next_personas ?? account.prediction?.expected_next_personas ?? [];
  const personaCadence = account.execution?.persona_engagements || [];
  const entryPoints = useMemo(() => {
    const rows = account.entry_points ?? thesis?.entry_points ?? [];
    return rows.slice(0, 5);
  }, [account.entry_points, thesis?.entry_points]);
  const formatEntryPercent = (value?: number | null) =>
    value !== null && value !== undefined ? fmtPercent(value, true) : "—";

  const heading = (label: string, helper: string) => (
    <Stack direction="row" spacing={0.75} alignItems="center">
      <Typography variant="h6">{label}</Typography>
      <Tooltip title={helper}>
        <InfoOutlinedIcon fontSize="small" color="action" />
      </Tooltip>
    </Stack>
  );
  const wolvesRefreshed = wolvesUpdatedAt ? formatRelativeTime(wolvesUpdatedAt) : null;
  const hasKeystone = keystonePersonas.length > 0;
  const keystoneLookup = useMemo(() => {
    const byId = new Map<string, KeystonePersona>();
    const byLabel = new Map<string, KeystonePersona>();
    keystonePersonas.forEach((persona) => {
      if (persona.persona_id) {
        byId.set(persona.persona_id, persona);
      }
      if (persona.persona_label) {
        const normalized = normalizePersonaLabel(persona.persona_label);
        if (normalized) {
          byLabel.set(normalized, persona);
        }
      }
    });
    return { byId, byLabel };
  }, [keystonePersonas]);
  const getKeystoneFor = useCallback(
    (personaId?: string | null, personaLabel?: string | null) => {
      if (personaId && keystoneLookup.byId.has(personaId)) {
        return keystoneLookup.byId.get(personaId)!;
      }
      if (personaLabel) {
        const normalized = normalizePersonaLabel(personaLabel);
        if (normalized && keystoneLookup.byLabel.has(normalized)) {
          return keystoneLookup.byLabel.get(normalized)!;
        }
      }
      return null;
    },
    [keystoneLookup]
  );

  const keystoneColor = (score?: number | null): "default" | "primary" | "success" => {
    if (typeof score !== "number") return "default";
    if (score >= 0.7) return "success";
    if (score >= 0.55) return "primary";
    return "default";
  };

  const formatChipLabel = (band?: string | null, label?: string | null, prob?: number | null) => {
    const parts: string[] = [];
    if (band) parts.push(titleize(band));
    parts.push(label || "Persona");
    if (typeof prob === "number") {
      parts.push(fmtPercent(prob, true));
    }
    return parts.join(" · ");
  };
  const personaLikelihoodRows = useMemo(() => {
    if (!personaLikelihoods.length) return [];
    return personaLikelihoods
      .map((entry, idx) => {
        const keystone = getKeystoneFor(entry.persona_id, entry.persona_label);
        return {
          key: entry.persona_id || entry.persona_label || `persona-${idx}`,
          persona: entry,
          keystone,
          roleLabel: classifyKeystoneRole(keystone?.journey_phase),
          wolvesNarrative: describeKeystoneImpact(keystone),
        };
      })
      .sort((a, b) => {
        const scoreA =
          typeof a.keystone?.wolves_score === "number" ? (a.keystone.wolves_score as number) : -1;
        const scoreB =
          typeof b.keystone?.wolves_score === "number" ? (b.keystone.wolves_score as number) : -1;
        if (scoreA !== scoreB) return scoreB - scoreA;
        return (
          (b.persona.committee_probability || b.persona.probability || 0) -
          (a.persona.committee_probability || a.persona.probability || 0)
        );
      });
  }, [getKeystoneFor, personaLikelihoods]);
  const coalitionStory = useMemo(() => {
    if (!primaryPath?.personas?.length) return null;
    const entries = primaryPath.personas.slice(0, 4).map((persona, idx) => {
      const label = persona.label || persona.id || `Persona ${idx + 1}`;
      const keystone = getKeystoneFor(persona.id, persona.label);
      const role = classifyKeystoneRole(keystone?.journey_phase);
      if (keystone) {
        return `${label}${role ? ` (${role})` : " (keystone)"}`;
      }
      return label;
    });
    if (!entries.length) return null;
    const segmentLabel =
      (account.meta?.industry && titleize(account.meta?.industry)) ||
      (account.meta?.segment && titleize(account.meta?.segment)) ||
      "peer accounts";
    return `Likely winning pattern (${segmentLabel}): ${entries.join(" → ")}`;
  }, [account.meta?.industry, account.meta?.segment, getKeystoneFor, primaryPath]);

  return (
    <Stack spacing={3}>
      {entryPoints.length ? (
        <Box>
          {heading(
            "Entry points to prioritize",
            "Graph perceptibility × intervention reach to decide which personas to activate first."
          )}
          <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
            Combines belief graph signals with available people and plays. Higher entry score =
            faster path to influence.
          </Typography>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>Persona</TableCell>
                <TableCell align="right">Entry score</TableCell>
                <TableCell>Signals</TableCell>
                <TableCell>Intervention plan</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {entryPoints.map((entry, idx) => (
                <TableRow key={`entry-point-${entry.persona_id}-${idx}`}>
                  <TableCell>
                    <Stack spacing={0.25}>
                      <Typography variant="body2" sx={{ fontWeight: 600 }}>
                        {entry.persona_label || entry.persona_id}
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        Rank {entry.rank ?? idx + 1}
                      </Typography>
                    </Stack>
                  </TableCell>
                  <TableCell align="right">
                    <Stack spacing={0.25} alignItems="flex-end">
                      <Typography variant="body2">
                        {formatEntryPercent(entry.entry_score)}
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        Combined perc {formatEntryPercent(entry.combined_perceptibility)} · prox{" "}
                        {formatEntryPercent(entry.combined_proximity)}
                      </Typography>
                    </Stack>
                  </TableCell>
                  <TableCell>
                    <Stack spacing={0.25}>
                      <Typography variant="caption" color="text.secondary">
                        Graph {formatEntryPercent(entry.graph_perceptibility)} perc /{" "}
                        {formatEntryPercent(entry.graph_proximity)} prox
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        Intervention reach {formatEntryPercent(entry.intervention_reach)}
                      </Typography>
                    </Stack>
                  </TableCell>
                  <TableCell>
                    <Stack direction="row" spacing={0.5} sx={{ flexWrap: "wrap" }}>
                      {(entry.top_people || []).slice(0, 2).map((person) => (
                        <Chip
                          key={`entry-point-${entry.persona_id}-person-${person.person_id || person.display_name}`}
                          size="small"
                          variant="outlined"
                          label={`${person.display_name || "Person"} · ${formatEntryPercent(
                            person.committee_probability
                          )}`}
                        />
                      ))}
                      {(entry.top_plays || []).slice(0, 2).map((play, playIdx) => (
                        <Chip
                          key={`entry-point-${entry.persona_id}-play-${play.play_id || playIdx}`}
                          size="small"
                          color="primary"
                          variant="outlined"
                          label={describeEntryPlay(play) || "Recommended play"}
                        />
                      ))}
                    </Stack>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </Box>
      ) : null}
      {hasKeystone && (
        <Box>
          {heading(
            "Keystone personas",
            "Personas with the highest wolves-impact score across observed episodes."
          )}
          <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
            These personas materially change predicted win rates when engaged.
            {wolvesRefreshed ? ` Refreshed ${wolvesRefreshed}.` : ""}
          </Typography>
          <Stack spacing={1.5}>
            {keystonePersonas.slice(0, 4).map((persona) => (
              <Paper
                key={`keystone-${persona.persona_id}`}
                variant="outlined"
                sx={{ p: 1.5, borderColor: "primary.100" }}
              >
                <Stack direction="row" justifyContent="space-between" alignItems="center">
                  <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
                    {persona.persona_label}
                  </Typography>
                  <Stack direction="row" spacing={0.5}>
                    {persona.is_new_persona && (
                      <Chip size="small" color="warning" variant="outlined" label="New persona" />
                    )}
                    <Chip
                      size="small"
                      color={keystoneColor(persona.wolves_score)}
                      variant="outlined"
                      label={`Wolves ${fmtPercent(persona.wolves_score, true)}`}
                    />
                  </Stack>
                </Stack>
                <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>
                  Δ win {fmtBasisPoints(persona.wolves_delta_bp)}
                  {" · "}Involvement {fmtPercent(persona.wolves_involvement_rate, true)}
                  {" · "}Blocker {fmtPercent(persona.wolves_blocker_rate, true)}
                  {typeof persona.wolves_sample_size === "number"
                    ? ` · Sample n=${persona.wolves_sample_size}`
                    : ""}
                </Typography>
                <Stack direction="row" spacing={0.75} sx={{ mt: 0.75, flexWrap: "wrap" }}>
                  {persona.journey_phase && (
                    <Chip size="small" label={persona.journey_phase} variant="outlined" />
                  )}
                  {typeof persona.priority_score === "number" && (
                    <Chip
                      size="small"
                      label={`Priority ${persona.priority_score.toFixed(2)}`}
                      variant="outlined"
                    />
                  )}
                  {typeof persona.matched_people_count === "number" && (
                    <Chip
                      size="small"
                      label={`${persona.matched_people_count} matched ${
                        persona.matched_people_count === 1 ? "person" : "people"
                      }`}
                      variant="outlined"
                    />
                  )}
                </Stack>
                {persona.top_people && persona.top_people.length > 0 && (
                  <Stack direction="row" spacing={0.5} sx={{ flexWrap: "wrap", mt: 0.75 }}>
                    {persona.top_people.slice(0, 3).map((person) => (
                      <Chip
                        key={`keystone-${persona.persona_id}-${person.person_id || person.display_name}`}
                        size="small"
                        variant="outlined"
                        label={`${person.display_name || "Person"} · ${fmtPercent(
                          person.committee_probability,
                          true
                        )}`}
                      />
                    ))}
                  </Stack>
                )}
              </Paper>
            ))}
          </Stack>
        </Box>
      )}
      <Box>
        {heading(
          "1. Who is likely to be involved?",
          "Posterior committee personas plus the real people mapped to each role."
        )}
        <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
          The personas most likely to participate in this account's win committee.
        </Typography>
        {personaLikelihoodRows.length ? (
          <>
            <Table size="small" sx={{ mb: 1 }}>
              <TableHead>
                <TableRow>
                  <TableCell>Persona</TableCell>
                  <TableCell>Role</TableCell>
                  <TableCell align="right">Committee</TableCell>
                  <TableCell align="right">Wolves</TableCell>
                  <TableCell>Chain effect</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {personaLikelihoodRows.slice(0, 6).map((row) => {
                  const committeeProb =
                    row.persona.committee_probability ?? row.persona.probability ?? null;
                  return (
                    <TableRow key={`persona-likelihood-${row.key}`}>
                      <TableCell>
                        <Stack spacing={0.25}>
                          <Stack direction="row" spacing={0.5} alignItems="center">
                            <Typography variant="body2">
                              {row.persona.persona_label || row.persona.persona_id || row.key}
                            </Typography>
                            {row.keystone ? (
                              <Chip size="small" color="success" label="Keystone" />
                            ) : null}
                          </Stack>
                          {row.persona.band ? (
                            <Typography variant="caption" color="text.secondary">
                              {titleize(row.persona.band)}
                            </Typography>
                          ) : null}
                        </Stack>
                      </TableCell>
                      <TableCell>{row.roleLabel || "—"}</TableCell>
                      <TableCell align="right">
                        {committeeProb !== null
                          ? fmtPercent(committeeProb, true)
                          : "—"}
                      </TableCell>
                      <TableCell align="right">
                        {row.keystone?.wolves_score != null
                          ? fmtPercent(row.keystone.wolves_score, true)
                          : "—"}
                      </TableCell>
                      <TableCell>
                        <Typography
                          variant="body2"
                          color={row.wolvesNarrative ? "success.main" : "text.secondary"}
                        >
                          {row.wolvesNarrative || "Model expects this persona to unlock the next gate."}
                        </Typography>
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
            {coalitionStory ? (
              <Typography variant="caption" color="text.secondary" sx={{ mb: 1 }}>
                {coalitionStory}
              </Typography>
            ) : null}
          </>
        ) : (
          <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
            Persona committee model will populate once this account accumulates more telemetry.
          </Typography>
        )}
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
  const [accountPlanContract, setAccountPlanContract] =
    useState<AccountPlanContract | null>(null);
  const [companyId, setCompanyId] = useState<string | null>(null);
  const [targetAccountLookup, setTargetAccountLookup] = useState<
    Record<string, TargetAccountLookupEntry>
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
        setCompanyId(me.company_id);

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
    if (!token || !companyId || !selectedProductId) {
      setTargetAccountLookup({});
      return;
    }
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch(
          `http://localhost:8000/get-target-accounts/${companyId}?product_id=${selectedProductId}`,
          { headers: { Authorization: `Bearer ${token}` } }
        );
        if (!res.ok) throw new Error(await res.text());
        const data = await res.json();
        if (cancelled) return;
        const next: Record<string, TargetAccountLookupEntry> = {};
        (data.accounts || []).forEach((entry: TargetAccountLookupEntry) => {
          if (entry.id) {
            next[entry.id] = entry;
          }
        });
        setTargetAccountLookup(next);
      } catch (err) {
        if (!cancelled) {
          console.warn("Failed to load target accounts for account plan", err);
          setTargetAccountLookup({});
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [companyId, selectedProductId, token]);

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
    if (!token || !selectedProductId || !accountId) {
      setAccountPlanContract(null);
      return;
    }
    (async () => {
      try {
        const res = await fetch(
          `http://localhost:8000/accounts/${accountId}/plan?product_id=${encodeURIComponent(
            selectedProductId
          )}`,
          { headers: { Authorization: `Bearer ${token}` } }
        );
        if (!res.ok) throw new Error(await res.text());
        const data: AccountPlanContract = await res.json();
        setAccountPlanContract(data);
      } catch (err: unknown) {
        console.warn("Failed to load account plan contract", err);
      }
    })();
  }, [selectedProductId, token, accountId]);

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
  const activeLookupEntry = activeAccount
    ? targetAccountLookup[activeAccount.account_id]
    : undefined;
  const activeAccountDisplayName = getAccountDisplayName(
    activeAccount,
    activeLookupEntry
  );

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
  const cadenceKeystoneResolver = useMemo(() => {
    const keystones = activeAccount?.keystone_personas ?? [];
    if (!keystones.length) return undefined;
    const byId = new Map<string, KeystonePersona>();
    const byLabel = new Map<string, KeystonePersona>();
    keystones.forEach((persona) => {
      if (persona.persona_id) {
        byId.set(persona.persona_id, persona);
      }
      if (persona.persona_label) {
        const normalized = normalizePersonaLabel(persona.persona_label);
        if (normalized) {
          byLabel.set(normalized, persona);
        }
      }
    });
    return (row: AssetCadenceRow) => {
      const candidate =
        (row.persona_id && byId.get(row.persona_id)) ||
        (row.persona_label && byLabel.get(normalizePersonaLabel(row.persona_label))) ||
        null;
      if (!candidate) return null;
      return {
        label: candidate.persona_label,
        wolvesScore: candidate.wolves_score,
        wolvesDeltaBp: candidate.wolves_delta_bp,
        narrative: describeKeystoneImpact(candidate),
      };
    };
  }, [activeAccount?.keystone_personas]);

  const beliefPulse = useMemo(
    () =>
      buildBeliefPulse(
        activeAccount,
        activeEnrichment,
        topPlay,
        personaLabelLookup,
        activeAccount?.keystone_personas ?? []
      ),
    [activeAccount, activeAccount?.keystone_personas, activeEnrichment, topPlay, personaLabelLookup]
  );

  const accountInterventions = useMemo(
    () => (activeAccount?.interventions ?? []).slice(0, 3),
    [activeAccount?.interventions]
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
        {accountPlanContract?.account && (
          <Typography variant="caption" color="text.secondary">
            Win probability {fmtPercent(accountPlanContract.account.predictedWinPct, true)} ·
            Stall in {accountPlanContract.account.predictedStallInDays} days
          </Typography>
        )}
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
          <Typography color="text.primary">{activeAccountDisplayName}</Typography>
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
          const lookupEntry = targetAccountLookup[acct.account_id];
          const accountName = getAccountDisplayName(acct, lookupEntry);
          const metaChips = uniqueStrings(
            Object.entries(getAccountMetaForChips(acct, lookupEntry)).map(
              ([key, value]) => {
                if (value === undefined || value === null) return null;
                return `${key.replace(/[_-]/g, " ")}: ${String(value)}`;
              }
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
                      {accountName}
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
          {accountInterventions.length ? (
            <Card variant="outlined">
              <CardContent>
                <Stack
                  direction={{ xs: "column", md: "row" }}
                  spacing={1}
                  justifyContent="space-between"
                  alignItems={{ xs: "flex-start", md: "center" }}
                  sx={{ mb: 1 }}
                >
                  <Typography variant="h6">Opportunities & gaps</Typography>
                  <Typography variant="caption" color="text.secondary">
                    Top interventions that still need stronger assets
                  </Typography>
                </Stack>
                <Stack spacing={1.25}>
                  {accountInterventions.map((gap) => {
                    const coverage = gap.coverage_score ?? 0;
                    const coveragePercent = Math.round(coverage * 100);
                    const coverageColor =
                      gap.coverage_state === "strong"
                        ? "success"
                        : gap.coverage_state === "steady"
                        ? "warning"
                        : "error";
                    return (
                      <Paper
                        key={gap.id}
                        variant="outlined"
                        sx={{ p: 1.25, borderColor: gap.needs_net_new ? "error.light" : "divider" }}
                      >
                        <Stack spacing={0.5}>
                          <Stack
                            direction={{ xs: "column", sm: "row" }}
                            spacing={0.5}
                            justifyContent="space-between"
                            alignItems={{ xs: "flex-start", sm: "center" }}
                          >
                            <Typography variant="subtitle2">
                              {gap.persona_label || gap.persona_descriptor || "Persona"} ·{" "}
                              {gap.stage_label || "Stage"}
                            </Typography>
                            <Chip size="small" color={coverageColor} label={fmtPercent(coverage)} />
                          </Stack>
                          <Typography variant="body2" color="text.secondary">
                            Need to prove:{" "}
                            {gap.concern_theme || "Belief transition still being inferred"} (
                            {fmtBasisPoints(gap.belief_lift_bp)} lift)
                          </Typography>
                          <LinearProgress
                            variant="determinate"
                            value={coveragePercent}
                            color={coverageColor}
                            sx={{ height: 6, borderRadius: 3 }}
                          />
                          <Typography variant="caption" color="text.secondary">
                            {gap.current_modality?.format_label
                              ? `${gap.current_modality.format_label} via ${
                                  gap.current_modality.channel_label || "channel"
                                }`
                              : "No mapped asset yet"}
                          </Typography>
                          <Stack direction="row" spacing={1} sx={{ mt: 0.5 }}>
                            <Button
                              size="small"
                              variant="contained"
                              component={RouterLink}
                              to="/marketing-planner"
                            >
                              Map in planner
                            </Button>
                            {gap.needs_net_new ? (
                              <Button size="small" variant="outlined" color="error">
                                Design asset
                              </Button>
                            ) : null}
                          </Stack>
                        </Stack>
                      </Paper>
                    );
                  })}
                </Stack>
              </CardContent>
            </Card>
          ) : null}
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
                keystonePersonas={activeAccount.keystone_personas ?? []}
                wolvesUpdatedAt={activeAccount.wolves_metrics_updated_at}
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
                    keystoneResolver={cadenceKeystoneResolver}
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
const isExpectedNextPersona = (
  entry: ExpectedNextPersona | { persona?: string; persona_label?: string; prob?: number } | null
): entry is ExpectedNextPersona => {
  return !!entry && typeof entry === "object" && "persona_id" in entry;
};
