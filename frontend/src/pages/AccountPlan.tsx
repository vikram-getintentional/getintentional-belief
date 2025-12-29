import React, { useCallback, useEffect, useMemo, useState } from "react";
import {
  Alert,
  Box,
  Breadcrumbs,
  Button,
  Card,
  CardActionArea,
  CardActions,
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
import AccountEnrichmentDialog, {
  type EnrichmentAccount,
} from "../components/AccountEnrichmentDialog";
import PlanQualityPanel from "../components/PlanQualityPanel";
import type {
  AccountPlanContract,
  PlanningMode,
  WinOutlook,
  WinOutlookDriver,
  PlanQuality,
} from "../types/apiContracts";

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

type PersonaDefinition = {
  persona_id: string;
  persona_label: string;
  title?: string | null;
  department?: string | null;
  seniority?: string | null;
  belief_level?: number | null;
  belief_band?: string | null;
  journey_phase?: string | null;
  phase_probs?: Record<string, number> | null;
  committee_probability?: number | null;
  on_primary_path?: boolean;
  display_group_key?: string;
  stages_covered?: string[];
};

type CadenceTier = "primary" | "probe" | "projection";

type PersonaStrategyEntry = {
  persona_id: string;
  persona_label?: string | null;
  belief_level?: number | null;
  belief_band?: string | null;
  journey_phase?: string | null;
  committee_probability?: number | null;
  path_probability?: number | null;
  recommended_action?: string | null;
  cadence_tier?: CadenceTier | null;
  role?: "primary_path" | "committee" | "projection";
  on_primary_path?: boolean;
  projection_probability?: number | null;
  projection_note?: string | null;
  engagement_state?: string | null;
  display_group_key?: string | null;
};

type BeliefRow = PersonaStrategyEntry & {
  display_group_key: string;
  stagesCovered?: string[];
};

type PersonaStrategyBudgetSplit = {
  primary_pct: number;
  secondary_pct: number;
  rationale?: string | null;
  escalation_triggers?: string[];
};

type PersonaStrategy = {
  primary_path_personas?: PersonaStrategyEntry[];
  committee_personas?: PersonaStrategyEntry[];
  projected_committee_members?: PersonaStrategyEntry[];
  budget_split?: PersonaStrategyBudgetSplit;
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

type JobPhase =
  | "account_context"
  | "persona_matching"
  | "journey_replay"
  | "belief_state"
  | "blockers_cascades"
  | "intervention_plan"
  | "done";

type JobStatus = {
  job_id: string;
  type: string;
  status: "queued" | "running" | "completed" | "failed";
  phase: JobPhase;
  message: string;
  progress: number;
  updated_at?: string | null;
  result_location?: string | null;
  error?: string | null;
};

const JOB_PHASE_LABELS: Record<JobPhase, string> = {
  account_context: "Account context",
  persona_matching: "Persona matching",
  journey_replay: "Journey replay",
  belief_state: "Belief state",
  blockers_cascades: "Blockers & cascades",
  intervention_plan: "Intervention plan",
  done: "Done",
};

const friendlyPhaseLabel = (phase: string) =>
  JOB_PHASE_LABELS[phase as JobPhase] || phase;

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
    persona_strategy?: PersonaStrategy;
    personas_by_id?: Record<string, PersonaDefinition>;
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
  planning_mode?: PlanningMode;
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

const personaDisplayLabel = (
  personaId?: string | null,
  fallback?: string | null,
  definitions?: Record<string, PersonaDefinition>
) => {
  if (personaId && definitions) {
    const def = definitions[personaId];
    if (def?.persona_label) return def.persona_label;
  }
  if (fallback) return fallback;
  if (personaId) return personaId;
  return "Persona";
};

const isSentinelPersona = (personaId?: string | null) =>
  Boolean(personaId && personaId.includes("__STOP__"));

const formatPlanningModeLabel = (mode?: PlanningMode) => {
  if (!mode) return undefined;
  switch (mode) {
    case "graph_hypothesis":
      return "Graph Hypothesis mode";
    case "observed_signal":
      return "Observed signal mode";
    case "learned_strategy":
      return "Learned strategy mode";
    default:
      return undefined;
  }
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
  return `${safeItems.slice(0, -1).join(", ")}, ${conjunction} ${
    safeItems[safeItems.length - 1]
  }`;
};

const formatWinRange = (
  ci?: { low?: number | null; high?: number | null }
): string | null => {
  if (!ci) return null;
  const { low, high } = ci;
  if (low == null || high == null) return null;
  return `${fmtPercent(low, true)}–${fmtPercent(high, true)}`;
};

const formatStatusLabel = (status?: string) => {
  if (!status) return "Unknown";
  return status
    .split(/[_\s]+/)
    .filter(Boolean)
    .map(
      (part) => part.charAt(0).toUpperCase() + part.slice(1).toLowerCase()
    )
    .join(" ");
};

const renderWinDrivers = (drivers?: WinOutlookDriver[]) => {
  if (!drivers?.length) {
    return (
      <Typography variant="body2" color="text.secondary">
        No meta drivers yet.
      </Typography>
    );
  }
  return (
    <Stack spacing={0.25}>
      {drivers.map((driver) => (
        <Typography
          key={`${driver.feature}-${driver.direction}`}
          variant="body2"
          color="text.secondary"
        >
          {driver.feature} ·{" "}
          {driver.direction === "+" ? "↑" : "↓"}{" "}
          {fmtPercent(driver.weight, true)}
        </Typography>
      ))}
    </Stack>
  );
};

const DEFAULT_PLAN_QUALITY: PlanQuality = {
  readiness: {
    score: 0,
    dimensions: [
      {
        id: "arsenal",
        label: "Assets & Channels",
        score: 0,
        status: "missing",
        reason: "No assets or channels configured yet.",
        recommended_actions: [{ label: "Add Arsenal", route: "/arsenal" }],
      },
      {
        id: "enrichment",
        label: "Account Enrichment",
        score: 0,
        status: "missing",
        reason: "No personas matched yet.",
        recommended_actions: [{ label: "Enrich Account", route: "/account-enrichment" }],
      },
      {
        id: "engagements",
        label: "Engagement Signal",
        score: 0,
        status: "missing",
        reason: "No observed engagement signals imported yet.",
        recommended_actions: [{ label: "Import Engagements", route: "/engagements-setup" }],
      },
      {
        id: "learning",
        label: "Historical Learning",
        score: 0,
        status: "missing",
        reason: "No won/lost journeys ingested yet.",
        recommended_actions: [{ label: "Import CRM History", route: "/crm-setup" }],
      },
    ],
  },
  predictiveConfidence: {
    stars: 1,
    score: 0.0,
    components: [
      {
        id: "graph_coverage",
        score: 0.0,
        note: "Graph coverage is still being established.",
      },
      {
        id: "engagement_alignment",
        score: 0.0,
        note: "No engagement signals captured yet.",
      },
      {
        id: "historical_similarity",
        score: 0.0,
        note: "No comparable closed outcomes yet.",
      },
    ],
    explanation:
      "GI is waiting on assets, engagement signals, and historical outcomes to calibrate confidence.",
  },
};

const DEFAULT_WIN_OUTLOOK: WinOutlook = {
  diagnostics: {
    graph_walk_reachability: 1.0,
    note: "Diagnostic only; not shown in UI",
  },
  right_to_win: {
    p: null,
    ci: { low: null, high: null },
    coverage: { similar_deals: 0, wins: 0, losses: 0 },
    top_drivers: [],
    status: "no_historical_data",
  },
  baseline_win: {
    p: 0.01,
    ci: { low: null, high: null },
    evidence: {
      engagement_count: 0,
      summary: "No engagement evidence yet.",
    },
    status: "no_engagements",
  },
  predicted_win: {
    p: 0.015,
    ci: { low: null, high: null },
    range: { low: null, high: null },
    assumptions: [],
    lift_over_current: 0.0,
    status: "uses_priors",
    evidence: {
      engagement_count: 0,
      summary: "Forecast follows baseline until recommended plays run.",
    },
  },
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

interface PersonaStageWeight {
  personaId: string;
  personaLabel: string;
  stageWeights: Array<{ stage: string; weight: number }>;
}

const renderPersonaPath = (
  path?: PersonaPathEntry,
  stageDistribution?: PersonaStageWeight[]
) => {
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
        {(stageDistribution && stageDistribution.length ? stageDistribution : []).map(
          (entry) => (
            <Paper
              key={`stage-summary-${entry.personaId}`}
              variant="outlined"
              sx={{ p: 1.25 }}
            >
              <Stack spacing={0.5}>
                <Typography variant="subtitle2">
                  {entry.personaLabel}
                </Typography>
                <Stack direction="row" spacing={0.5} sx={{ flexWrap: "wrap" }}>
                  {entry.stageWeights.map((stage) => (
                    <Chip
                      key={`${entry.personaId}-${stage.stage}`}
                      size="small"
                      variant="outlined"
                      label={`${stage.stage} ${fmtPercent(stage.weight, true)}`}
                    />
                  ))}
                </Stack>
              </Stack>
            </Paper>
          )
        )}
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

const isExpectedNextPersona = (
  entry: ExpectedNextPersona | { persona?: string; persona_label?: string; prob?: number } | null
) => {
  return !!entry && typeof entry === "object" && "persona_id" in entry;
};

// ---------------------------------------------------------------------------
// Belief pulse synthesis
// ---------------------------------------------------------------------------

const buildBeliefPulse = (
  account: AccountPlan | null,
  enrichment: AccountEnrichmentPayload | undefined,
  topPlay: Play | null,
  personaLabelLookup: Record<string, string>,
  keystonePersonas: KeystonePersona[] = [],
  planningMode?: PlanningMode
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
    personaEngagements.find((entry) => (entry as any).persona_id === personaId);

  const keystoneList =
    (keystonePersonas && keystonePersonas.length
      ? keystonePersonas
      : account.keystone_personas) ?? [];
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

  let expectedNext =
    account.thesis?.expected_next_personas?.[0] ||
    account.prediction?.expected_next_personas?.[0] ||
    account.prediction?.expected_next?.[0] ||
    null;
  const expectedNextPersonaId =
    (expectedNext as any)?.persona_id ||
    (expectedNext as any)?.persona ||
    (expectedNext as any)?.id;
  if (isSentinelPersona(expectedNextPersonaId)) {
    expectedNext = null;
  }

  const expectedNextProbability =
    typeof (expectedNext as any)?.probability === "number"
      ? (expectedNext as any).probability
      : typeof (expectedNext as any)?.prob === "number"
      ? (expectedNext as any).prob
      : null;

  const expectedNextLabel = describePersonaEntry(expectedNext);
  const expectedNextKeystone = getKeystoneFor(expectedNext);
  const expectedNextNarrative = describeKeystoneImpact(expectedNextKeystone);
  const expectedNextReason = isExpectedNextPersona(expectedNext)
    ? (expectedNext as ExpectedNextPersona).reason
    : undefined;
  const expectedNextStage = isExpectedNextPersona(expectedNext)
    ? (expectedNext as ExpectedNextPersona).journey_stage
    : undefined;
  const topRecommendedPlay = topPlay;

  const progressBuckets = new Set([
    "on_path",
    "near_path",
    "jump_ahead",
    "skip_hit",
  ]);
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
    const lastTouchDays = (riskEngagement as any)?.engagement_snapshot?.last_touch_days ?? null;
    const touchDensity = (riskEngagement as any)?.engagement_snapshot?.touch_density_per_week ?? null;
    const riskWhyBits = [
      lastTouchDays !== null && lastTouchDays !== undefined
        ? `Last touch ${fmtDays(lastTouchDays)} ago`
        : null,
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
        value:
          "High — this persona anchors the current hypothesis. Without a mapped contact + first-touch owner, predicted progress stalls.",
      },
      {
        label: "Why",
        value: riskWhyBits.length
          ? riskWhyBits.join(" · ")
          : "No engagement telemetry yet for this persona.",
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
          value: `${bucketLabel}${
            typeof accelerationStep.t === "number" ? ` at t=${accelerationStep.t}` : ""
          }`,
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
        value: `${fmtBasisPoints(
          topRecommendedPlay.expected_delta_bp
        )} in ${fmtDays(topRecommendedPlay.avg_duration_days)}`,
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

  const hasAssets = Boolean(
    account?.execution?.asset_cadence &&
      account.execution.asset_cadence.length > 0
  );
  const graphHypAction =
    "Map a real Sales Manager contact to validate this belief path.";
  const executionAction =
    "Increase mid-funnel touches to reinforce belief progression.";
  const recommendedActionValue = topRecommendedPlay
    ? `Queue ${topRecommendedPlay.name || topRecommendedPlay.asset_label || "top play"} via ${
        topRecommendedPlay.channel_label || "primary channel"
      }.`
    : !hasAssets
    ? "No asset recommendations yet. Add assets and channels before increasing touches."
    : planningMode === "graph_hypothesis"
    ? graphHypAction
    : executionAction;
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
      value: recommendedActionValue,
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
        Plays populate once the arsenal is configured. Engagement history
        will refine channel and asset attribution.
      </Typography>
    );
  }

  return (
    <Box sx={{ overflowX: "auto" }}>
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
                play.channel?.name ||
                play.channel_label ||
                play.channel?.channel_type_label;
              const name =
                play.name || assetName || channelName || "Recommended play";
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
                  <TableCell align="right">
                    {fmtPercent(play.avg_confidence, true)}
                  </TableCell>
                  <TableCell align="right">
                    {typeof play.avg_duration_days === "number"
                      ? fmtDays(play.avg_duration_days)
                      : "—"}
                  </TableCell>
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      </TableContainer>
    </Box>
  );
};

const renderConversionSequence = (entries?: ConversionSequenceEntry[]) => {
  if (!entries || entries.length === 0) {
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
          chips.push({
            label: `Effort ${fmtPercent(entry.effort_pct, true)}`,
            outlined: true,
          });
        }

        if (typeof entry.time_to_effect_days === "number") {
          chips.push({
            label: `Time ${fmtDays(entry.time_to_effect_days)}`,
            outlined: true,
          });
        }

        const description =
          entry.belief_transition_meta?.narrative ||
          entry.belief_transition_meta?.pain?.label ||
          entry.belief_transition_meta?.problem?.label ||
          entry.belief_transition_meta?.resolution?.label ||
          "Belief progression";

        const stageLabel = entry.belief_transition_meta?.stage_label
          ? `${entry.belief_transition_meta.stage_label}: `
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
                <Typography
                  key={`sequence-${entry.timeline_index}-detail-${idx}`}
                  variant="body2"
                >
                  {line}
                </Typography>
              ))}
            </Box>
          ) : null;

        const timelineLabel =
          entry.timeline_label ??
          (typeof entry.timeline_days === "number"
            ? `T+${entry.timeline_days}`
            : `Step ${entry.timeline_index + 1}`);

        return (
          <Paper key={`sequence-${entry.timeline_index}`} variant="outlined" sx={{ p: 2 }}>
            <Stack
              direction={{ xs: "column", md: "row" }}
              spacing={1}
              alignItems={{ xs: "flex-start", md: "center" }}
              justifyContent="space-between"
            >
              <Stack spacing={0.5} sx={{ flexGrow: 1 }}>
                <Typography variant="subtitle2">{timelineLabel}</Typography>
                <Typography variant="body2" color="text.secondary">
                  {description}
                </Typography>
              </Stack>

              <Stack
                direction="row"
                spacing={0.5}
                sx={{ flexWrap: "wrap", mt: { xs: 1, md: 0 } }}
              >
                {chips.map((chip) => (
                  <Chip
                    key={`${entry.timeline_index}-${chip.label}`}
                    size="small"
                    label={chip.label}
                    variant={chip.outlined ? "outlined" : "filled"}
                  />
                ))}
              </Stack>

              {tooltip && (
                <Tooltip title={tooltip} arrow>
                  <IconButton
                    size="small"
                    sx={{ ml: { xs: 0, md: 1 }, mt: { xs: 1, md: 0 } }}
                  >
                    <InfoOutlinedIcon fontSize="inherit" />
                  </IconButton>
                </Tooltip>
              )}
            </Stack>
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
  const personaStageDistribution = useMemo<PersonaStageWeight[]>(() => {
    if (!primaryPath?.personas?.length) return [];
    const map = new Map<
      string,
      {
        personaLabel: string;
        stageCounts: Map<string, number>;
        total: number;
        order: number;
      }
    >();
    primaryPath.personas.forEach((persona, index) => {
      const personaId =
        persona.id ||
        persona.persona_id ||
        persona.persona ||
        `persona-${index}`;
      if (!personaId) return;
      const stage =
        persona.journey_phase ||
        persona.stage_label ||
        persona.stage ||
        persona.persona_stage ||
        persona.stage_label ||
        `Stage ${index + 1}`;
      const entry =
        map.get(personaId) ||
        {
          personaLabel: persona.label || persona.persona_label || personaId,
          stageCounts: new Map<string, number>(),
          total: 0,
          order: index,
        };
      entry.stageCounts.set(stage, (entry.stageCounts.get(stage) ?? 0) + 1);
      entry.total += 1;
      map.set(personaId, entry);
    });
    return Array.from(map.entries())
      .sort((a, b) => a[1].order - b[1].order)
      .map(([personaId, entry]) => ({
        personaId,
        personaLabel: entry.personaLabel,
        stageWeights: Array.from(entry.stageCounts.entries()).map(
          ([stage, count]) => ({
            stage,
            weight: count / (entry.total || 1),
          })
        ),
      }));
  }, [primaryPath]);
  const plays = account.execution?.plays || [];
  const thesis = account.thesis;
  const personaLikelihoods = (
    thesis?.persona_likelihoods ?? []
  ).filter((entry) => !isSentinelPersona(entry.persona_id));
  const personLikelihoods = (
    thesis?.person_likelihoods ?? []
  ).filter((entry) => !isSentinelPersona(entry.persona_id));
  const rawPersonaCadence = (
    account.execution?.persona_engagements ?? []
  ).filter((persona) => !isSentinelPersona((persona as any).persona_id));
  const personaDefinitions = (
    account.thesis?.personas_by_id ?? {}
  ) as Record<string, PersonaDefinition>;
  const entryPoints = useMemo(() => {
    return (account.entry_points ?? thesis?.entry_points ?? []).filter(
      (entry) => !isSentinelPersona(entry.persona_id)
    );
  }, [account.entry_points, thesis?.entry_points]);
  const groupedEntryPoints = useMemo(() => {
    const groups = new Map<
      string,
      (typeof entryPoints[number] & { displayGroupKey: string })
    >();
    entryPoints.forEach((entry) => {
      const groupKey =
        personaDefinitions[entry.persona_id || ""]?.display_group_key ||
        entry.persona_label ||
        entry.persona_id ||
        "persona";
      const existing = groups.get(groupKey);
      const label =
        personaDefinitions[entry.persona_id || ""]?.persona_label ||
        entry.persona_label ||
        entry.persona_id;
      const candidate = {
        ...entry,
        persona_label: label,
        displayGroupKey: groupKey,
      };
      if (!existing || (entry.entry_score || 0) > (existing.entry_score || 0)) {
        groups.set(groupKey, candidate);
      }
    });
    return Array.from(groups.values());
  }, [entryPoints, personaDefinitions]);
  const groupedEntryPointsDisplay = useMemo(
    () => groupedEntryPoints.slice(0, 5),
    [groupedEntryPoints]
  );
  const personaStrategy = account.thesis?.persona_strategy;
  const primaryPathPersonas = personaStrategy?.primary_path_personas ?? [];
  const committeePersonas = personaStrategy?.committee_personas ?? [];
  const projectedCommitteeMembers = (
    personaStrategy?.projected_committee_members ?? []
  ).filter((entry) => !isSentinelPersona(entry.persona_id)) as PersonaStrategyEntry[];
  const personaEntries = useMemo(
    () => [...primaryPathPersonas, ...committeePersonas, ...projectedCommitteeMembers],
    [primaryPathPersonas, committeePersonas, projectedCommitteeMembers]
  );
  const personaCadenceEntries = useMemo(() => {
    const map = new Map<string, PersonaEngagementPlan>();
    rawPersonaCadence.forEach((entry) => {
      if (entry.persona_id && !isSentinelPersona(entry.persona_id)) {
        map.set(entry.persona_id, entry);
      }
    });
    personaEntries.forEach((entry) => {
      const personaId = entry.persona_id;
      if (
        !personaId ||
        map.has(personaId) ||
        isSentinelPersona(personaId)
      ) {
        return;
      }
      map.set(personaId, {
        persona_id: personaId,
        persona_label: entry.persona_label,
        journey_phase: entry.journey_phase,
        belief_level: entry.belief_level,
        belief_band: entry.belief_band,
        committee_probability: entry.committee_probability,
        cadence: [
          {
            phase: entry.cadence_tier || "probe",
            frequency_per_week:
              entry.cadence_tier === "primary" ? 1.0 : 0.5,
            policy_note:
              entry.recommended_action || "Graph hypothesis cadence",
          },
        ],
      });
    });
    return Array.from(map.values());
  }, [personaEntries, rawPersonaCadence]);
  const groupedCadence = useMemo(() => {
    const cadenceMap = new Map<string, PersonaEngagementPlan>();
    personaCadenceEntries.forEach((entry) => {
      const def = personaDefinitions[entry.persona_id || ""];
      const groupKey =
        def?.display_group_key ||
        def?.persona_label ||
        entry.persona_label ||
        entry.persona_id ||
        "persona";
      const existing = cadenceMap.get(groupKey);
      const targetCadence = existing ? [...existing.cadence] : [];
      entry.cadence.forEach((row) => {
        if (!targetCadence.some((existingRow) => existingRow.phase === row.phase)) {
          targetCadence.push(row);
        }
      });
      cadenceMap.set(groupKey, {
        ...existing,
        ...entry,
        persona_label: def?.persona_label || entry.persona_label,
        persona_id: existing?.persona_id || entry.persona_id,
        cadence: targetCadence,
      });
    });
    return Array.from(cadenceMap.values());
  }, [personaCadenceEntries, personaDefinitions]);

  const personaCadenceMap = useMemo(() => {
    const map = new Map<string, PersonaEngagementPlan>();
    groupedCadence.forEach((entry) => {
      const pid =
        entry.persona_id ||
        entry.persona_label ||
        (entry as any).persona?.id ||
        (entry as any).persona?.persona_id;
      if (pid && !isSentinelPersona(pid)) {
        map.set(pid, entry);
      }
    });
    return map;
  }, [groupedCadence]);
  const personaMap = useMemo(() => {
    const map = new Map<string, PersonaStrategyEntry>();
    personaEntries.forEach((entry) => {
      if (!entry.persona_id) return;
      const def = personaDefinitions[entry.persona_id];
      const merged: PersonaStrategyEntry = {
        ...entry,
        persona_label: entry.persona_label || def?.persona_label,
        belief_level: entry.belief_level ?? def?.belief_level,
        belief_band: entry.belief_band ?? def?.belief_band,
        journey_phase: entry.journey_phase ?? def?.journey_phase,
        committee_probability:
          entry.committee_probability ?? def?.committee_probability,
        display_group_key:
          entry.display_group_key || def?.display_group_key,
      };
      map.set(entry.persona_id, merged);
    });
    Object.entries(personaDefinitions).forEach(([pid, def]) => {
      if (!pid || map.has(pid)) return;
      map.set(pid, {
        persona_id: pid,
        persona_label: def.persona_label,
        belief_level: def.belief_level,
        belief_band: def.belief_band,
        journey_phase: def.journey_phase,
        committee_probability: def.committee_probability,
        display_group_key: def.display_group_key,
        cadence_tier: def.on_primary_path ? "primary" : "probe",
        recommended_action: def.on_primary_path
          ? "Invest heavily in this path"
          : "Keep committee probes warm",
        role: def.on_primary_path ? "primary_path" : "committee",
        engagement_state: undefined,
      });
    });
    return map;
  }, [personaEntries, personaDefinitions]);
  const allPersonas = useMemo(() => Array.from(personaMap.values()), [personaMap]);
  const beliefRows = useMemo(() => {
    const aggregated = new Map<string, BeliefRow>();
    const stageCoverage = new Map<string, Set<string>>();
    allPersonas.forEach((entry) => {
      const key =
        entry.display_group_key ||
        entry.persona_label ||
        entry.persona_id ||
        "persona";
      if (!aggregated.has(key)) {
        aggregated.set(key, {
          ...entry,
          display_group_key: key,
          stagesCovered: [],
        });
      }
      const stage = entry.journey_phase || entry.dominant_phase;
      if (stage) {
        const set = stageCoverage.get(key) || new Set<string>();
        set.add(stage);
        stageCoverage.set(key, set);
      }
    });
    return Array.from(aggregated.entries()).map(([key, row]) => ({
      ...row,
      stagesCovered: Array.from(stageCoverage.get(key) || []).filter(Boolean),
    }));
  }, [allPersonas]);
  const budgetSplit = personaStrategy?.budget_split;

  const formatEntryPercent = (value?: number | null) => {
    if (value === null || value === undefined) return "—";
    const pct = value * 100;
    const rounded = Math.round(pct * 10) / 10;
    return `${rounded.toFixed(1).replace(/\.0$/, "")}%`;
  };
  const formatGraphMetricCopy = (value?: number | null, label = "Graph") => {
    if (value && value > 0) {
      return `${label}: ${formatEntryPercent(value)}`;
    }
    return `${label}: Graph-only estimate (insufficient evidence)`;
  };

  const heading = (label: string, helper: string) => (
    <Stack direction="row" spacing={0.75} alignItems="center">
      <Typography variant="h6">{label}</Typography>
      <Tooltip title={helper}>
        <InfoOutlinedIcon fontSize="small" color="action" />
      </Tooltip>
    </Stack>
  );

  const getProjectedProbability = (
    entry: PersonaStrategyEntry | ExpectedNextPersona
  ) =>
    (entry as PersonaStrategyEntry).projection_probability ??
    entry.probability ??
    (entry as PersonaStrategyEntry).path_probability ??
    0;

  const getProjectedStage = (
    entry: PersonaStrategyEntry | ExpectedNextPersona
  ) => entry.journey_stage || (entry as PersonaStrategyEntry).journey_phase || "Journey";

  const getProjectedRecommendedAction = (
    entry: PersonaStrategyEntry | ExpectedNextPersona
  ) =>
    (entry as PersonaStrategyEntry).recommended_action ??
    (entry as ExpectedNextPersona).reason ??
    "Belief graph projection";

  const onProjectPrimaryPath = (
    entry: PersonaStrategyEntry | ExpectedNextPersona
  ) =>
    (entry as PersonaStrategyEntry).on_primary_path ??
    (entry as PersonaStrategyEntry).role === "primary_path";

  const renderStrategyList = (
    title: string,
    personas: PersonaStrategyEntry[]
  ) => {
    if (!personas.length) return null;
    return (
      <Box sx={{ mt: 2 }}>
        <Typography variant="subtitle2" sx={{ mb: 0.5 }}>
          {title}
        </Typography>
        <Stack spacing={0.75}>
          {personas.map((entry) => (
            <Stack
              key={`strategy-${entry.persona_id}-${entry.journey_phase || entry.role}`}
              direction="row"
              spacing={1}
              alignItems="center"
              flexWrap="wrap"
            >
              <Chip
                size="small"
                label={entry.cadence_tier?.toUpperCase() || "TRACK"}
                color={entry.role === "primary_path" ? "primary" : "default"}
              />
              <Typography variant="body2" sx={{ fontWeight: 600 }}>
                  {personaDisplayLabel(
                    entry.persona_id,
                    entry.persona_label,
                    personaDefinitions
                  )}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                {entry.recommended_action || "Maintain belief probe."}
              </Typography>
              {entry.journey_phase ? (
                <Typography variant="caption" color="text.secondary">
                  Stage: {entry.journey_phase}
                </Typography>
              ) : null}
            </Stack>
          ))}
        </Stack>
      </Box>
    );
  };

  const wolvesRefreshed = wolvesUpdatedAt ? formatRelativeTime(wolvesUpdatedAt) : null;
  const hasKeystone = keystonePersonas.length > 0;
  const hasAccountAssets =
    Boolean(account.execution?.asset_cadence) &&
    account.execution.asset_cadence.length > 0;

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

  const keystoneColor = (
    score?: number | null
  ): "default" | "primary" | "success" => {
    if (typeof score !== "number") return "default";
    if (score >= 0.7) return "success";
    if (score >= 0.55) return "primary";
    return "default";
  };

  const formatChipLabel = (
    band?: string | null,
    label?: string | null,
    prob?: number | null
  ) => {
    const parts: string[] = [];
    if (band) parts.push(titleize(band));
    parts.push(label || "Persona");
    if (typeof prob === "number") {
      parts.push(fmtPercent(prob, true));
    }
    return parts.join(" · ");
  };

  const personaLabelLookup = useMemo(() => {
    const lookup: Record<string, string> = {};
    account.prediction?.persona_paths.forEach((path) => {
      (path.personas || []).forEach((persona) => {
        if (persona.id && persona.label) {
          lookup[persona.id] = persona.label;
        }
      });
    });
    return lookup;
  }, [account.prediction?.persona_paths]);

  const personaLikelihoodRows = useMemo(() => {
    if (!personaLikelihoods.length) return [];
    const rows = personaLikelihoods
      .map((entry, idx) => {
        const keystone = getKeystoneFor(entry.persona_id, entry.persona_label);
        const displayLabel =
          entry.persona_label ||
          (entry.persona_id ? personaLabelLookup[entry.persona_id] : undefined) ||
          entry.persona_id ||
          `Persona ${idx + 1}`;
        return {
          key: entry.persona_id || entry.persona_label || `persona-${idx}`,
          persona: entry,
          displayLabel,
          keystone,
          roleLabel: classifyKeystoneRole(keystone?.journey_phase),
          wolvesNarrative: describeKeystoneImpact(keystone),
        };
      })
      .sort((a, b) => {
        const scoreA =
          typeof a.keystone?.wolves_score === "number"
            ? (a.keystone.wolves_score as number)
            : -1;
        const scoreB =
          typeof b.keystone?.wolves_score === "number"
            ? (b.keystone.wolves_score as number)
            : -1;
        if (scoreA !== scoreB) return scoreB - scoreA;
        return (
          (b.persona.committee_probability || b.persona.probability || 0) -
          (a.persona.committee_probability || a.persona.probability || 0)
        );
      });
    return rows;
  }, [getKeystoneFor, personaLikelihoods, personaLabelLookup]);
  const groupedPersonaLikelihoodRows = useMemo(() => {
    const groups = new Map<
      string,
      typeof personaLikelihoodRows[number] & { displayGroupKey: string }
    >();
    personaLikelihoodRows.forEach((row) => {
      const def = personaDefinitions[row.persona.persona_id || ""];
      const groupKey =
        def?.display_group_key ||
        def?.persona_label ||
        row.displayLabel ||
        row.persona.persona_id ||
        "persona";
      if (!groups.has(groupKey)) {
        groups.set(groupKey, { ...row, displayGroupKey: groupKey });
      }
    });
    return Array.from(groups.values());
  }, [personaLikelihoodRows, personaDefinitions]);

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
      (account.meta?.industry && titleize(account.meta?.industry as string)) ||
      (account.meta?.segment && titleize(account.meta?.segment as string)) ||
      "peer accounts";
    return `Typical pattern GI expects (unvalidated) (${segmentLabel}): ${entries.join(" → ")}`;
  }, [account.meta, getKeystoneFor, primaryPath]);

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
          <Box sx={{ overflowX: "auto" }}>
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
                {groupedEntryPointsDisplay.map((entry, idx) => {
                  const topPlay = entry.top_plays?.[0];
                  const stageLabel = topPlay?.stage || "Belief progression";
                  const concernLabel =
                    topPlay?.concern || "Graph-inferred concern (no CRM signal yet)";
                  const assetLabel = entry.top_plays?.[0]?.asset;
                  const assetChannel = topPlay?.channel;
                  return (
                    <TableRow
                      key={`entry-point-${entry.persona_id}-${stageLabel}-${idx}`}
                    >
                      <TableCell>
                        <Stack spacing={0.25}>
                          <Stack direction="row" spacing={0.5} alignItems="center">
                            <Typography variant="body2" sx={{ fontWeight: 600 }}>
                            {personaDisplayLabel(
                              entry.persona_id,
                              entry.persona_label,
                              personaDefinitions
                            )}
                            </Typography>
                            <Chip
                              size="small"
                              variant="outlined"
                              label="🧠 Graph-inferred"
                            />
                          </Stack>
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
                        <Stack spacing={0.5}>
                          <Stack spacing={0.25}>
                            <Typography variant="caption" color="text.secondary">
                              {formatGraphMetricCopy(entry.graph_perceptibility, "Perceptibility")}
                            </Typography>
                            <Typography variant="caption" color="text.secondary">
                              {formatGraphMetricCopy(entry.graph_proximity, "Proximity")}
                            </Typography>
                            <Typography variant="caption" color="text.secondary">
                              {formatGraphMetricCopy(entry.intervention_reach, "Intervention reach")}
                            </Typography>
                            <Typography variant="caption" color="text.secondary">
                              {graphConfidenceLabel}
                            </Typography>
                          </Stack>
                          <Stack direction="row" spacing={0.5} sx={{ flexWrap: "wrap" }}>
                            {(entry.top_people || []).slice(0, 2).map((person) => (
                              <Chip
                                key={`entry-point-${entry.persona_id}-person-${
                                  person.person_id || person.display_name
                                }`}
                                size="small"
                                variant="outlined"
                                label={`${person.display_name || "Person"} · ${formatEntryPercent(
                                  person.committee_probability
                                )}`}
                              />
                            ))}
                            {(entry.top_plays || []).slice(0, 2).map((play, playIdx) => (
                              <Chip
                                key={`entry-point-${entry.persona_id}-play-${
                                  play.play_id || playIdx
                                }`}
                                size="small"
                                color="primary"
                                variant="outlined"
                                label={describeEntryPlay(play) || "Recommended play"}
                              />
                            ))}
                          </Stack>
                        </Stack>
                      </TableCell>
                      <TableCell>
                        <Stack spacing={0.25}>
                          <Typography variant="body2" color="text.secondary">
                            Stage: {stageLabel}
                          </Typography>
                          <Typography variant="body2" color="text.secondary">
                            Concern: {concernLabel}
                          </Typography>
                          <Typography variant="body2" color="text.secondary">
                            Assets:{" "}
                            {assetLabel
                              ? `${assetLabel}${assetChannel ? ` via ${assetChannel}` : ""}`
                              : "No asset recommendations yet — keep touches lightweight."}
                          </Typography>
                        </Stack>
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </Box>
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
                      <Chip
                        size="small"
                        color="warning"
                        variant="outlined"
                        label="New persona"
                      />
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
                <Stack
                  direction="row"
                  spacing={0.75}
                  sx={{ mt: 0.75, flexWrap: "wrap" }}
                >
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
                  <Stack
                    direction="row"
                    spacing={0.5}
                    sx={{ flexWrap: "wrap", mt: 0.75 }}
                  >
                    {persona.top_people.slice(0, 3).map((person) => (
                      <Chip
                        key={`keystone-${persona.persona_id}-${
                          person.person_id || person.display_name
                        }`}
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
        {groupedPersonaLikelihoodRows.length ? (
          <>
            <Box sx={{ overflowX: "auto" }}>
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
                  {groupedPersonaLikelihoodRows.slice(0, 6).map((row) => {
                    const committeeProb =
                      row.persona.committee_probability ??
                      row.persona.probability ??
                      null;
                    return (
                      <TableRow key={`persona-likelihood-${row.key}`}>
                        <TableCell>
                          <Stack spacing={0.25}>
                            <Stack direction="row" spacing={0.5} alignItems="center">
                              <Typography variant="body2">
                                {row.displayLabel}
                              </Typography>
                              {row.keystone ? (
                                <Chip size="small" color="success" label="Keystone" />
                              ) : null}
                            </Stack>
                            {row.persona.band ? (
                              <Typography
                                variant="caption"
                                color="text.secondary"
                              >
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
                            color={
                              row.wolvesNarrative
                                ? "success.main"
                                : "text.secondary"
                            }
                          >
                            {row.wolvesNarrative ||
                              "Model expects this persona to unlock the next gate."}
                          </Typography>
                        </TableCell>
                      </TableRow>
                    );
                  })}
                </TableBody>
              </Table>
            </Box>
            {coalitionStory ? (
              <>
                <Typography
                  variant="caption"
                  color="text.secondary"
                  sx={{ mb: 0.5 }}
                >
                  {coalitionStory}
                </Typography>
                <Typography variant="caption" color="text.secondary" sx={{ mb: 1 }}>
                  This pattern is inferred from the belief graph. It will change as real engagement and outcomes are observed.
                </Typography>
              </>
            ) : null}
          </>
        ) : (
          <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
            Persona committee model will populate once this account accumulates more
            telemetry.
          </Typography>
        )}
        <Typography variant="body2" color="text.secondary">
          People mapped to these personas (highest probability first):
        </Typography>
        {personLikelihoods.length ? (
          <Stack
            direction="row"
            spacing={0.5}
            sx={{ flexWrap: "wrap", mt: 1 }}
          >
            {personLikelihoods.slice(0, 10).map((person) => (
              <Chip
                key={`person-likelihood-${
                  person.person_id || person.display_name
                }`}
                variant="outlined"
                size="small"
                label={formatChipLabel(
                  person.role_band,
                  person.display_name,
                  person.probability
                )}
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
                <Stack
                  key={`${pid}-people`}
                  direction="row"
                  spacing={0.5}
                  alignItems="center"
                >
                  <Chip
                    label={persona.label || pid || "Persona"}
                    size="small"
                  />
                  <Stack
                    direction="row"
                    spacing={0.5}
                    sx={{ flexWrap: "wrap" }}
                  >
                    {people.map((name) => (
                      <Chip
                        key={`${pid}-${name}`}
                        label={name}
                        size="small"
                        variant="outlined"
                      />
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
              <Typography
                variant="body2"
                color="text.secondary"
                sx={{ mb: 0.5 }}
              >
                Conversion path context:
              </Typography>
              {renderPersonaPath(primaryPath, personaStageDistribution)}
            </>
          ) : (
            renderTypicalPathSteps(
              canonicalTypicalPath,
              "Using canonical path until this account accumulates journey telemetry."
            )
          )}
        </Box>
        {renderStrategyList("Primary path personas (A)", primaryPathPersonas)}
        {renderStrategyList("Committee personas (B)", committeePersonas)}
        {budgetSplit ? (
          <Box sx={{ mt: 2 }}>
            <Typography variant="body2" color="text.secondary">
              Invest {budgetSplit.primary_pct}% in primary tracks and{" "}
              {budgetSplit.secondary_pct}% in committee probes.
            </Typography>
            {budgetSplit.rationale ? (
              <Typography variant="caption" color="text.secondary">
                {budgetSplit.rationale}
              </Typography>
            ) : null}
            {budgetSplit.escalation_triggers?.length ? (
              <Box sx={{ mt: 1 }}>
                <Typography variant="caption" color="text.secondary">
                  Escalation triggers:
                </Typography>
                <Stack spacing={0.5}>
                  {budgetSplit.escalation_triggers.map((trigger, idx) => (
                    <Typography
                      key={`trigger-${idx}`}
                      variant="caption"
                      color="text.secondary"
                    >
                      • {trigger}
                    </Typography>
                  ))}
                </Stack>
              </Box>
            ) : null}
          </Box>
        ) : null}
        {projectedCommitteeMembers.length ? (
          <Box sx={{ mt: 2 }}>
            <Typography variant="subtitle2" sx={{ mb: 0.5 }}>
              Likely committee members to appear next (projection)
            </Typography>
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Persona</TableCell>
                  <TableCell align="right">Probability</TableCell>
                  <TableCell>Stage</TableCell>
                  <TableCell>On primary path?</TableCell>
                  <TableCell>Recommended action</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {projectedCommitteeMembers.slice(0, 6).map((entry, idx) => (
                  <TableRow
                    key={`expected-next-${entry.persona_id}-${idx}`}
                    hover
                  >
                    <TableCell>
                      {personaDisplayLabel(
                        entry.persona_id,
                        entry.persona_label,
                        personaDefinitions
                      )}
                    </TableCell>
                    <TableCell align="right">
                      {fmtPercent(getProjectedProbability(entry), true)}
                    </TableCell>
                    <TableCell>{getProjectedStage(entry)}</TableCell>
                    <TableCell>
                      {onProjectPrimaryPath(entry) ? "Yes" : "No"}
                    </TableCell>
                    <TableCell>
                      {getProjectedRecommendedAction(entry)}
                    </TableCell>
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
        {beliefRows.length ? (
          <Table size="small" sx={{ mt: 1 }}>
            <TableHead>
              <TableRow>
                <TableCell>Persona</TableCell>
                <TableCell align="right">Belief</TableCell>
                <TableCell>Phase / Stages</TableCell>
                <TableCell align="right">Committee</TableCell>
                <TableCell align="right">Cadence</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {beliefRows.map((row) => {
                const cadenceEntry = personaCadenceMap.get(row.persona_id);
                return (
                  <TableRow key={`belief-${row.persona_id}`}>
                    <TableCell>
                      {personaDisplayLabel(
                        row.persona_id,
                        row.persona_label,
                        personaDefinitions
                      )}
                    </TableCell>
                    <TableCell align="right">
                      {row.belief_level != null ? (
                        <Chip
                          size="small"
                          variant="outlined"
                          label={
                            row.belief_band
                              ? `${fmtPercent(row.belief_level, true)} · ${titleize(
                                  row.belief_band
                                )}`
                              : fmtPercent(row.belief_level, true)
                          }
                        />
                      ) : (
                        "—"
                      )}
                    </TableCell>
                    <TableCell>
                      {(row.dominant_phase || row.journey_phase || row.stagesCovered?.length) ? (
                        <Tooltip
                          title={
                            row.phase_probs
                              ? Object.entries(row.phase_probs)
                                  .map(
                                    ([phase, prob]) =>
                                      `${phase}: ${fmtPercent(prob, true)}`
                                  )
                                  .join(" · ")
                              : ""
                          }
                        >
                          <Chip
                            size="small"
                            label={
                              row.stagesCovered && row.stagesCovered.length
                                ? row.stagesCovered.join(" · ")
                                : row.dominant_phase || row.journey_phase
                            }
                          />
                        </Tooltip>
                      ) : (
                        "—"
                      )}
                    </TableCell>
                    <TableCell align="right">
                      {row.committee_probability != null
                        ? fmtPercent(row.committee_probability, true)
                        : "—"}
                    </TableCell>
                    <TableCell align="right">
                      {cadenceEntry && cadenceEntry.engagement_state ? (
                        `${titleize(cadenceEntry.engagement_state)} · ${formatFrequency(
                          cadenceEntry.base_frequency_per_week
                        )}`
                      ) : (
                        "—"
                      )}
                    </TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        ) : (
          <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
            Belief table will appear once persona data is available.
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
          {hasAccountAssets
            ? "U-shaped cadence tuned to each persona's belief state and fatigue level."
            : "No asset recommendations yet. Cadence shown is exploratory and intentionally lightweight."}
        </Typography>
        <Box sx={{ mt: 1.5 }}>
          <PersonaEngagementCadence
            engagements={groupedCadence}
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

  const [products, setProducts] =
    useState<Array<{ id: string; name: string }>>([]);
  const [selectedProductId, setSelectedProductId] = useState<string>("");
  const [plan, setPlan] = useState<MarketingPlan | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [enrichmentLoading, setEnrichmentLoading] =
    useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [enrichmentMap, setEnrichmentMap] = useState<
    Record<string, AccountEnrichmentPayload>
  >({});
  const [accountPlanContract, setAccountPlanContract] =
    useState<AccountPlanContract | null>(null);
  const [contractError, setContractError] = useState<string | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [jobStatus, setJobStatus] = useState<JobStatus | null>(null);
  const [lastFetchedJobId, setLastFetchedJobId] = useState<string | null>(null);
  const [enrichmentDialogAccount, setEnrichmentDialogAccount] =
    useState<EnrichmentAccount | null>(null);
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
          setError(
            "No products found. Run Value Prop to create the first product."
          );
          return;
        }
        setProducts(list);
        setSelectedProductId((prev) => prev || list[0].id);
      } catch (err: unknown) {
        console.error(err);
        const message =
          err instanceof Error
            ? err.message
            : "Failed to load products.";
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
          console.warn(
            "Failed to load target accounts for account plan",
            err
          );
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
        const message =
          err instanceof Error
            ? err.message
            : "Failed to load account plan.";
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
      setContractError(null);
      setJobId(null);
      setJobStatus(null);
      setLastFetchedJobId(null);
      return;
    }
    setAccountPlanContract(null);
    setContractError(null);
    setJobStatus(null);
    setLastFetchedJobId(null);
    setJobId(null);
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch("http://localhost:8000/jobs/account-plan", {
          method: "POST",
          headers: {
            Authorization: `Bearer ${token}`,
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            account_id: accountId,
            product_id: selectedProductId,
          }),
        });
        if (!res.ok) throw new Error(await res.text());
        const data = await res.json();
        if (cancelled) return;
        setJobId(data.job_id);
      } catch (err: unknown) {
        if (!cancelled) {
          console.error("Failed to start account plan job", err);
          const message =
            err instanceof Error
              ? err.message
              : "Unable to start account plan job.";
          setContractError(message);
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [accountId, selectedProductId, token]);

  useEffect(() => {
    if (!jobId || !token) {
      return;
    }
    let cancelled = false;
    let intervalHandle: ReturnType<typeof setInterval> | null = null;
    const pollStatus = async () => {
      try {
        const res = await fetch(`http://localhost:8000/jobs/${jobId}`, {
          headers: { Authorization: `Bearer ${token}` },
        });
        if (!res.ok) throw new Error(await res.text());
        const status: JobStatus = await res.json();
        if (cancelled) return;
        setJobStatus(status);
        if (status.status === "completed" || status.status === "failed") {
          if (intervalHandle) {
            clearInterval(intervalHandle);
            intervalHandle = null;
          }
        }
      } catch (err) {
        console.warn("Failed to poll job status", jobId, err);
      }
    };
    intervalHandle = setInterval(pollStatus, 1500);
    pollStatus();
    return () => {
      cancelled = true;
      if (intervalHandle) {
        clearInterval(intervalHandle);
      }
    };
  }, [jobId, token]);

  useEffect(() => {
    if (!jobStatus || jobStatus.status !== "completed" || !token) {
      if (jobStatus?.status === "failed") {
        setContractError(
          jobStatus.error || "Account plan job failed. Click 'Show error' for details."
        );
        setAccountPlanContract(null);
      }
      return;
    }
    if (jobStatus.job_id === lastFetchedJobId) {
      return;
    }
    let cancelled = false;
    const fetchPath =
      jobStatus.result_location ||
      `/accounts/${accountId}/plan?product_id=${encodeURIComponent(selectedProductId)}`;
    (async () => {
      try {
        const res = await fetch(`http://localhost:8000${fetchPath}`, {
          headers: { Authorization: `Bearer ${token}` },
        });
        if (!res.ok) throw new Error(await res.text());
        const data: AccountPlanContract = await res.json();
        if (cancelled) return;
        setAccountPlanContract(data);
        setContractError(null);
        setLastFetchedJobId(jobStatus.job_id);
      } catch (err) {
        if (cancelled) return;
        console.warn("Failed to load account plan contract", err);
        setContractError(
          "Account plan quality metrics unavailable — backend contract fetch failed."
        );
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [accountId, jobStatus, lastFetchedJobId, selectedProductId, token]);

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
          console.warn(
            "Failed to load enrichment for account",
            account.account_id,
            err
          );
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
  const activePersonaDefinitions =
    activeAccount?.thesis?.personas_by_id ?? {};

  const topPlay =
    activeAccount?.execution?.plays &&
    activeAccount.execution.plays.length
      ? dedupePlays(activeAccount.execution.plays).sort(
          (a, b) => (b.expected_delta_bp || 0) - (a.expected_delta_bp || 0)
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
    const pushName = (
      pid: string | undefined | null,
      name?: string | null
    ) => {
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
        (row.persona_label &&
          byLabel.get(normalizePersonaLabel(row.persona_label))) ||
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
        activeAccount?.keystone_personas ?? [],
        accountPlanContract?.meta?.planningMode
      ),
    [
      activeAccount,
      activeEnrichment,
      topPlay,
      personaLabelLookup,
      activeAccount?.keystone_personas,
      accountPlanContract?.meta?.planningMode,
    ]
  );

  const accountInterventions = useMemo(
    () => (activeAccount?.interventions ?? []).slice(0, 3),
    [activeAccount?.interventions]
  );

  const accountUpdatedAt =
    activeEnrichment?.blueprint_generated_at ||
    plan?.summary.generated_at ||
    plan?.generated_at;
  const planningMode = accountPlanContract?.meta?.planningMode;
  const planningModeLabel = formatPlanningModeLabel(planningMode);
  const isGraphHypothesisMode = planningMode === "graph_hypothesis";
  const accountSummary = accountPlanContract?.account;
  const accountStatusCopy = accountSummary
    ? isGraphHypothesisMode
      ? `No conflicting signals yet (graph-only estimate) · Stall in ${accountSummary.predictedStallInDays} days`
      : `Win probability ${fmtPercent(accountSummary.predictedWinPct, true)} · Stall in ${accountSummary.predictedStallInDays} days`
    : null;

  const planQuality = accountPlanContract?.quality ?? DEFAULT_PLAN_QUALITY;
  const winOutlook = accountPlanContract?.winOutlook ?? DEFAULT_WIN_OUTLOOK;
  const rightToWin = winOutlook.right_to_win;
  const baselineWin = winOutlook.baseline_win;
  const predictedWin = winOutlook.predicted_win;
  const rightToWinRange = formatWinRange(rightToWin?.ci);
  const baselineWinRange = formatWinRange(baselineWin?.ci);
  const predictedWinRange = formatWinRange(predictedWin?.range);
  const diagnosticReachability = winOutlook.diagnostics?.graph_walk_reachability;
  const diagnosticNote = winOutlook.diagnostics?.note;
  const shouldShowContractMetrics = Boolean(
    accountId && (accountPlanContract || contractError)
  );
  const engagementSignals = accountPlanContract?.engagementSignals;
  const dataAvailability = accountPlanContract?.dataAvailability;
  const coverageDeals =
    (rightToWin?.coverage?.wins ?? 0) + (rightToWin?.coverage?.losses ?? 0);
  const observedSignals =
    dataAvailability?.observedEngagements ?? engagementSignals?.observed ?? 0;
  const plannedSignals =
    engagementSignals?.projected ??
    predictedWin?.evidence?.planned_signals ??
    0;
  const priorOnlyEstimate = observedSignals === 0 && coverageDeals === 0;
  const graphConfidenceLabel = priorOnlyEstimate
    ? "Confidence: Low (graph-only estimate, no CRM or historical deals yet)"
    : `Confidence: Moderate (${observedSignals} observed signal${observedSignals === 1 ? "" : "s"})`;

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
        <Box>
          <Typography variant="h5">Account Plan</Typography>
          {planningModeLabel ? (
            <Chip
              size="small"
              label={planningModeLabel}
              variant="outlined"
              sx={{ mt: 0.5 }}
            />
          ) : null}
          {accountStatusCopy ? (
            <Typography variant="caption" color="text.secondary">
              {accountStatusCopy}
            </Typography>
          ) : null}
          {engagementSignals ? (
            <Typography variant="caption" color="text.secondary">
              Observed signals: {observedSignals} · Planned signals: {plannedSignals}
            </Typography>
          ) : null}
          {dataAvailability ? (
            <Typography variant="caption" color="text.secondary">
              Graph: {dataAvailability.graph ? "ready" : "missing"} · Historic
              deals: {dataAvailability.historicDeals} · Arsenal assets:{" "}
              {dataAvailability.arsenalAssets} · People mapped:{" "}
              {dataAvailability.peopleMapped}
            </Typography>
          ) : null}
          {priorOnlyEstimate ? (
            <Typography variant="caption" color="text.secondary">
              Evidence grade: Prior-only estimate (no CRM or historical deals yet).
            </Typography>
          ) : null}
          {engagementSignals?.lastObservedAt ? (
            <Typography variant="caption" color="text.secondary">
              Last observed {formatRelativeTime(engagementSignals.lastObservedAt)}
            </Typography>
          ) : null}
        </Box>
        <Box sx={{ flexGrow: 1 }} />
        {products.length > 0 && (
          <FormControl size="small" sx={{ minWidth: 220 }}>
            <InputLabel id="account-plan-product-label">
              Product
            </InputLabel>
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

      {jobStatus && jobStatus.status !== "completed" ? (
        <Box
          sx={{
            mb: 3,
            p: 2,
            display: "flex",
            alignItems: "center",
            gap: 2,
            bgcolor: "background.paper",
            borderRadius: 1,
            boxShadow: 1,
          }}
        >
          <CircularProgress size={20} />
          <Box sx={{ flex: 1 }}>
            <Typography variant="body2" fontWeight={600}>
              {jobStatus.message}
            </Typography>
            <LinearProgress
              variant="determinate"
              value={Math.min(100, Math.max(0, jobStatus.progress ?? 0))}
              sx={{ mt: 0.5 }}
            />
            <Typography variant="caption" color="text.secondary">
              Step: {friendlyPhaseLabel(jobStatus.phase)}
            </Typography>
          </Box>
        </Box>
      ) : null}

      {jobStatus?.status === "failed" && jobStatus.error ? (
        <Alert severity="error" variant="outlined" sx={{ mb: 3 }}>
          {jobStatus.error}
        </Alert>
      ) : null}

      {isGraphHypothesisMode ? (
        <Alert severity="info" variant="outlined" sx={{ mb: 3 }}>
          <Stack spacing={1}>
            <Typography variant="subtitle2">Graph-only setup priority</Typography>
            <Typography variant="body2">
              Priority order: 1. Enrich account (map people). 2. Add assets & channels. 3. Import CRM history.
              Execution tuning can wait until these inputs are ready.
            </Typography>
            <Stack direction="row" spacing={1} sx={{ flexWrap: "wrap" }}>
              <Button
                size="small"
                variant="outlined"
                component={RouterLink}
                to="/account-enrichment"
              >
                Enrich account
              </Button>
              <Button
                size="small"
                variant="outlined"
                component={RouterLink}
                to="/arsenal"
              >
                Add assets & channels
              </Button>
              <Button
                size="small"
                variant="outlined"
                component={RouterLink}
                to="/crm-setup"
              >
                Import CRM history
              </Button>
            </Stack>
          </Stack>
        </Alert>
      ) : null}

      {shouldShowContractMetrics ? (
        <>
          {contractError ? (
            <Alert severity="warning" variant="outlined" sx={{ mb: 3 }}>
              {contractError}
            </Alert>
          ) : null}
          <PlanQualityPanel quality={planQuality} sx={{ mb: 3 }} />
          <Box sx={{ mb: 3 }}>
            <Typography variant="subtitle1" sx={{ mb: 1 }}>
              Win outlook
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={12} md={4}>
                <Paper variant="outlined" sx={{ p: 2, height: "100%" }}>
                  <Stack spacing={0.75}>
                    <Typography variant="subtitle2">Right to win (meta)</Typography>
                    <Typography variant="h5">
                      {fmtPercent(rightToWin?.p, true)}
                    </Typography>
                    {rightToWinRange ? (
                      <Typography variant="caption" color="text.secondary">
                        Range {rightToWinRange}
                      </Typography>
                    ) : null}
                    <Typography variant="body2" color="text.secondary">
                      Coverage: {rightToWin?.coverage?.similar_deals ?? 0} deals ·{" "}
                      {rightToWin?.coverage?.wins ?? 0} wins /{" "}
                      {rightToWin?.coverage?.losses ?? 0} losses
                    </Typography>
                    {renderWinDrivers(rightToWin?.top_drivers)}
                    <Chip
                      size="small"
                      variant="outlined"
                      label={formatStatusLabel(rightToWin?.status)}
                    />
                  </Stack>
                </Paper>
              </Grid>
              <Grid item xs={12} md={4}>
                <Paper variant="outlined" sx={{ p: 2, height: "100%" }}>
                  <Stack spacing={0.75}>
                    <Typography variant="subtitle2">
                      {priorOnlyEstimate
                        ? "Prior-only estimate (soft prior)"
                        : "Baseline win (observed reality)"}
                    </Typography>
                    <Typography variant="h5">
                      {fmtPercent(baselineWin?.p, true)}
                    </Typography>
                    {!priorOnlyEstimate && baselineWinRange ? (
                      <Typography variant="caption" color="text.secondary">
                        Range {baselineWinRange}
                      </Typography>
                    ) : null}
                    <Typography variant="body2" color="text.secondary">
                      {baselineWin?.evidence?.summary}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      Observed signals:{" "}
                      {baselineWin?.evidence?.observed_signals ??
                        baselineWin?.evidence?.engagement_count ??
                        0}
                    </Typography>
                    <Chip
                      size="small"
                      variant="outlined"
                      label={formatStatusLabel(baselineWin?.status)}
                    />
                  </Stack>
                </Paper>
              </Grid>
              <Grid item xs={12} md={4}>
                <Paper variant="outlined" sx={{ p: 2, height: "100%" }}>
                  <Stack spacing={0.75}>
                    <Typography variant="subtitle2">
                      Predicted win (if plan executes) · Evidence-adjusted
                    </Typography>
                    <Typography variant="h5">
                      {fmtPercent(predictedWin?.p, true)}
                    </Typography>
                    {predictedWinRange ? (
                      <Typography variant="caption" color="text.secondary">
                        Range {predictedWinRange}
                      </Typography>
                    ) : null}
                    <Typography variant="body2" color="text.secondary">
                      Forecasted lift {fmtPercent(predictedWin?.lift_over_current, true)}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      {predictedWin?.evidence?.summary}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      Observed:{" "}
                      {predictedWin?.evidence?.observed_signals ?? observedSignals}
                      {" · "}Planned:{" "}
                      {predictedWin?.evidence?.planned_signals ?? plannedSignals}
                    </Typography>
                    {(predictedWin?.assumptions || []).slice(0, 2).map((assumption, index) => (
                      <Typography
                        key={`forecast-${assumption.action_id}-${index}`}
                        variant="body2"
                        color="text.secondary"
                      >
                        {assumption.label} · Engagement {fmtPercent(assumption.p_engage, true)} · Δlog{" "}
                        {assumption.expected_delta_log_odds.toFixed(3)}
                        {assumption.history_note ? ` · ${assumption.history_note}` : ""}
                      </Typography>
                    ))}
                    <Chip
                      size="small"
                      variant="outlined"
                      label={formatStatusLabel(predictedWin?.status)}
                    />
                  </Stack>
                </Paper>
              </Grid>
            </Grid>
            {(diagnosticReachability != null || diagnosticNote) && (
              <Typography variant="caption" color="text.secondary" sx={{ mt: 1 }}>
                Graph reachability {fmtPercent(diagnosticReachability, true)} ·{" "}
                {diagnosticNote || "Diagnostic check only"}
              </Typography>
            )}
          </Box>
        </>
      ) : null}

      {accountId && activeAccount && (
        <Breadcrumbs sx={{ mb: 2 }}>
          <MUILink component={RouterLink} color="inherit" to="/account-plan">
            Account Plan
          </MUILink>
          <Typography color="text.primary">
            {activeAccountDisplayName}
          </Typography>
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
            (acct.engagementSignals?.observed ?? 0) > 0;
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
            <Grid
              key={acct.account_id}
              item
              xs={12}
              md={6}
              lg={4}
            >
                <Card
                  variant="outlined"
                  sx={{
                    borderColor: isSelected ? "primary.main" : undefined,
                    boxShadow: isSelected ? 2 : undefined,
                  }}
                >
                  <CardActionArea
                    onClick={() =>
                      navigate(`/account-plan/${acct.account_id}`)
                    }
                    sx={{ alignItems: "stretch" }}
                  >
                    <CardContent>
                      <Typography
                        variant="subtitle1"
                        sx={{ fontWeight: 600 }}
                      >
                        {accountName}
                      </Typography>
                      <Stack
                        direction="row"
                        spacing={0.75}
                        sx={{ flexWrap: "wrap", mt: 1 }}
                      >
                        {metaChips.slice(0, 5).map((chip) => (
                          <Chip
                            key={`${acct.account_id}-${chip}`}
                            label={chip}
                            size="small"
                          />
                        ))}
                      </Stack>
                      <Stack
                        direction="row"
                        spacing={1.5}
                        sx={{ mt: 2 }}
                      >
                        <Stack spacing={0.25}>
                          <Typography
                            variant="caption"
                            color="text.secondary"
                          >
                            Enrichment
                          </Typography>
                          <Typography variant="body2">
                            {enrichmentLoading && !enrichment
                              ? "Loading…"
                              : fmtPercent(coverage)}
                          </Typography>
                          {required !== undefined ? (
                            <Typography
                              variant="caption"
                              color="text.secondary"
                            >
                              {matchedPeople}/{required} personas matched
                            </Typography>
                          ) : null}
                        </Stack>
                        <Stack spacing={0.25}>
                          <Typography
                            variant="caption"
                            color="text.secondary"
                          >
                            Engagements
                          </Typography>
                          <Typography variant="body2">
                            {hasEngagements ? "Live" : "No engagements"}
                          </Typography>
                        </Stack>
                      </Stack>
                    </CardContent>
                  </CardActionArea>
                  <CardActions sx={{ p: 2 }}>
                    <Button
                      size="small"
                      onClick={(e) => {
                        e.stopPropagation();
                        setEnrichmentDialogAccount({
                          id: acct.account_id,
                          account_name: accountName,
                        });
                      }}
                      disabled={!selectedProductId || !token}
                    >
                      View persona details
                    </Button>
                  </CardActions>
                </Card>
              </Grid>
            );
          })}

        {!accounts.length && (
          <Grid item xs={12}>
            <Alert severity="info">
              No target accounts yet. Add accounts from the Target Accounts page to
              generate Account Plans.
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
                  <Typography variant="h6">
                    Opportunities & gaps
                  </Typography>
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
                        sx={{
                          p: 1.25,
                          borderColor: gap.needs_net_new
                            ? "error.light"
                            : "divider",
                        }}
                      >
                        <Stack spacing={0.5}>
                          <Stack
                            direction={{ xs: "column", sm: "row" }}
                            spacing={0.5}
                            justifyContent="space-between"
                            alignItems={{
                              xs: "flex-start",
                              sm: "center",
                            }}
                          >
                    <Typography variant="subtitle2">
                      {personaDisplayLabel(
                        gap.persona_id,
                        gap.persona_label || gap.persona_descriptor,
                        activePersonaDefinitions
                      )}
                      {" · "}
                      {gap.stage_label || "Stage"}
                    </Typography>
                            <Chip
                              size="small"
                              color={coverageColor}
                              label={fmtPercent(coverage, true)}
                            />
                          </Stack>
                          <Typography
                            variant="body2"
                            color="text.secondary"
                          >
                            Need to prove:{" "}
                            {gap.concern_theme ||
                              gap.stage_label ||
                              "Belief transition still being inferred"}{" "}
                            ({fmtBasisPoints(gap.belief_lift_bp)} lift)
                          </Typography>
                          <LinearProgress
                            variant="determinate"
                            value={coveragePercent}
                            color={coverageColor}
                            sx={{ height: 6, borderRadius: 3 }}
                          />
                          <Typography
                            variant="caption"
                            color="text.secondary"
                          >
                            {gap.current_modality?.format_label
                              ? `${gap.current_modality.format_label} via ${
                                  gap.current_modality.channel_label ||
                                  "channel"
                                }`
                              : "No mapped asset yet"}
                          </Typography>
                          <Stack
                            direction="row"
                            spacing={1}
                            sx={{ mt: 0.5 }}
                          >
                            <Button
                              size="small"
                              variant="contained"
                              component={RouterLink}
                              to="/marketing-planner"
                            >
                              Map in planner
                            </Button>
                            {gap.needs_net_new ? (
                              <Button
                                size="small"
                                variant="outlined"
                                color="error"
                              >
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
                {renderConversionSequence(
                  activeAccount.execution?.conversion_sequence
                )}
              </Box>
              <Divider sx={{ my: 3 }} />
              <Box>
                <Typography variant="subtitle2" sx={{ mb: 1 }}>
                  Asset Cadence Table
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Filterable mix of assets & channels aligned to funnel
                  stages and campaign themes.
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

      <AccountEnrichmentDialog
        open={Boolean(enrichmentDialogAccount)}
        onClose={() => setEnrichmentDialogAccount(null)}
        account={enrichmentDialogAccount}
        productId={selectedProductId}
        token={token}
      />
    </Box>
  );
};

export default AccountPlanPage;
