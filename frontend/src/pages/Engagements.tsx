// EngagementsSetup.tsx — FULL REPLACEMENT (with Global Insights + per-account learnings/feed)
import React, { useCallback, useEffect, useMemo, useState } from "react";
import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  CardHeader,
  Chip,
  CircularProgress,
  Collapse,
  Divider,
  FormControl,
  FormControlLabel,
  IconButton,
  InputLabel,
  MenuItem,
  Paper,
  Select,
  Snackbar,
  Stack,
  Switch,
  Tab,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Tabs,
  TextField,
  Tooltip,
  Typography,
} from "@mui/material";
import InfoOutlinedIcon from "@mui/icons-material/InfoOutlined";
import StorylineNarrative, {
  StorylinePayload,
} from "../components/StorylineNarrative";

const personaLabelOverrides = new Map<string, string>();

type PersonaIdentifier =
  | string
  | null
  | undefined
  | {
      persona?: string | null;
      persona_id?: string | null;
      id?: string | null;
      label?: string | null;
    };

function normalizePersonaId(id: PersonaIdentifier): string | null {
  if (id === null || id === undefined) return null;
  if (typeof id === "string") return id;
  if (typeof id === "object") {
    return (
      id.persona ||
      id.persona_id ||
      id.id ||
      (typeof id.label === "string" ? id.label : null)
    );
  }
  return null;
}

function registerPersonaLabel(id?: PersonaIdentifier, label?: string | null) {
  const normalized = normalizePersonaId(id);
  if (!normalized || typeof normalized !== "string") return;
  if (!label || typeof label !== "string") return;
  personaLabelOverrides.set(normalized, label);
}

/**
 * NOTE:
 * - This version consumes the backend "belief thesis" response:
 *   {
 *     account_id, product_id, persons, observed_persona_ids,
 *     initial_rcs, journey:{ steps:[{t, predicted_topK, observed_next, bucket, ...}] },
 *     walk_paths:[{personas:[id], probability, score, ...}],
 *     fit:{...},
 *     learning_summary?: {
 *       persona_graph_inferences?: PersonaPathThesis[],
 *       full_graph_inferences?: PersonaPathThesis[],
 *       top_edge_updates?: EdgeUpdate[],
 *       node_updates_ranked?: NodeUpdate[],
 *       ...
 *     },
 *     learning_neighborhoods?: {
 *       upstream?: {...},
 *       handoff?: {...},
 *       downstream?: {...}
 *     }
 *   }
 *
 * - For backward compatibility, if `learning_summary` is missing but
 *   `human_readable_learning` is present, we synthesize a minimal
 *   `learning_summary` from those strings.
 */

// -----------------------------------------
// Types
// -----------------------------------------
export type PersonaMatchesResponse = {
  product_id: string;
  account_id: string;
  global: {
    persona_recommendations: Array<{
      persona_id: string;
      persona_label?: string | null;
    }>;
  };
  incremental: any;   // your existing type from incremental_learnings_from_thesis
  activity: any[];    // journey steps
  latent_activity?: LatentActivityItem[];
  activity_story?: ActivityStoryItem[];
  thesis: any;        // full thesis if you want it
};

export type TargetAccount = {
  id: string;
  account_name: string;
  deal_status?: string;
};

export type EngagementActor = {
  name?: string;
  role?: string; // legacy
  title?: string; // canonical
  department: string;
  seniority?: string;
  confidence?: number;
};

export type Engagement = {
  account_id: string;
  timestamp: string; // ISO Z
  actor: EngagementActor;
  channel?: string;
  source: "marketing" | "sales" | "cs" | "product" | "other";
  raw_activity: string;
  asset_id?: string;
  inferred: boolean;
  __persisted__?: boolean;
};

type PersonaJobSuggestion = {
  job_text: string;
  canonical_job_id?: string | null;
  relevance?: number;
  likelihood?: number;
  score?: number;
  evidence?: any;
};

type PersonaMatchRow = {
  // entered/meta
  name?: string;
  title?: string;
  department?: string;
  seniority?: string;

  // graph/canonical match
  graph_persona_node_id?: string | null; // persona id ("title|department|seniority")
  graph_persona_score?: number | null; // 0..1
  graph_persona_label?: string | null; // pretty label if present

  canonical_persona_best?: string | null;
  canonical_persona_score?: number | null;
  canonical_persona_label?: string | null;
  canonical_meta?: { title?: string; department?: string; seniority?: string };

  alternates?: string[];
  jobs?: PersonaJobSuggestion[];
};

type CandidatePath = {
  personas: string[];
  belief_states: string[];
  score: number;
  probability: number;
  rationale: string;
};

type OverallFit = {
  best_path_probability: number;
  accuracy: number;
  surprisal_index: number;
  variance: number;
};

// -----------------------------------------
// Learning: edge & node updates
// -----------------------------------------

export type LearningBand = "upstream" | "handoff" | "downstream";

export interface EdgeUpdate {
  u: string; // node id
  v: string; // node id
  rel: string; // relationship type
  delta: number; // suggested weight change (+/-)
  band?: LearningBand; // upstream / handoff / downstream
  confidence?: number; // 0..1
  rationale?: string;

  u_label?: string;
  v_label?: string;
}

export interface NodeUpdate {
  node_id: string;
  node_label: string;
  field: string; // e.g. "perceptibility"
  delta: number; // +/-
  rationale?: string;
}

// -----------------------------------------
// Learning: inferences & summary
// -----------------------------------------

export interface PersonaPathThesis {
  anchor_persona_id?: string;
  anchor_label?: string;
  diagnosis: string;
  rationale?: string;
  band?: LearningBand;
}

export interface LearningSummary {
  // raw blobs (you may or may not use these)
  graphstore_summary?: any;
  graphstore_summary_v3?: any;
  diff_summary_v3?: {
    persona_path_diff?: {
      expected_top3?: string[];
      expected_top3_labels?: string[];
      observed_top3?: string[];
      observed_top3_labels?: string[];
    };
    learned?: {
      A_persona_path_theses?: PersonaPathThesis[];
      B_latent_node_theses?: PersonaPathThesis[];
      C_persona_product_theses?: PersonaPathThesis[];
      meta?: any;
    };
    recommendations?: {
      edge_updates_ranked?: EdgeUpdate[];
      node_updates_ranked?: NodeUpdate[];
    };
    version?: number;
  };

  // FE-ready slices
  persona_graph_inferences?: PersonaPathThesis[];
  full_graph_inferences?: PersonaPathThesis[];
  top_edge_updates?: EdgeUpdate[];
  node_updates_ranked?: NodeUpdate[];

  // nested recs, kept for completeness
  recommendations?: {
    edge_updates_ranked?: EdgeUpdate[];
    node_updates_ranked?: NodeUpdate[];
  };
}

// -----------------------------------------
// Learning: neighborhoods
// -----------------------------------------

export interface LearningNeighborhoodEntry {
  anchor: string; // node id or persona pair
  anchor_label: string;
  band: LearningBand;
  delta_sum: number;
  rationale?: string;
  edges: [string, string, string][]; // [u, v, rel]
  edge_labels: [string, string, string][]; // [u_label, v_label, rel]
}

export interface LearningNeighborhoods {
  upstream?: Record<string, LearningNeighborhoodEntry>;
  handoff?: Record<string, LearningNeighborhoodEntry>;
  downstream?: Record<string, LearningNeighborhoodEntry>;
}

// -----------------------------------------
// Belief Theses per account
// -----------------------------------------

export interface BeliefThesis {
  account_id: string;
  product_id: string;

  learning_summary?: LearningSummary;
  learning_neighborhoods?: LearningNeighborhoods;
}

type ObservedPersona = { id: string; weight: number };

type EngagementEdgeImpact = {
  from: string;
  to: string;
  fromLabel?: string;
  toLabel?: string;
  supports: boolean;
  deltaLogProb?: number | null;
  pathProbability?: number | null;
};

type EngagementEdgeIndication = {
  edge: string;
  reason?: string;
  band: LearningBand;
  fromId?: string | null;
  toId?: string | null;
  fromLabel?: string | null;
  toLabel?: string | null;
  rel?: string | null;
  delta?: number | null;
  confidence?: number | null;
};

type EngagementImpact = {
  edges: EngagementEdgeImpact[];
  hiddenStatePrior?: Record<string, number>;
  hiddenStatePosterior?: Record<string, number>;
};

type EngagementBeliefAnnotation = {
  expectedBefore: string | null;
  actualPersona: string | null;
  classification:
    | "expected"
    | "jump_ahead"
    | "off_path"
    | "no_persona"
    | "no_path";
  note: string;
  nextExpectedAfter: string | null;
  quality?: number | null; // 0..1 based on predicted_topK rank when known
  surpriseScore?: number | null; // 0..1
  indicates?: {
    upstream?: EngagementEdgeIndication[];
    handoff?: EngagementEdgeIndication[];
    downstream?: EngagementEdgeIndication[];
  };
  impact?: EngagementImpact;
};

type PredictedTopEntry =
  | string
  | {
      persona?: string | null;
      id?: string | null;
      persona_id?: string | null;
      persona_label?: string | null;
      prob?: number | null;
    };

type JourneyStep = {
  t: number;
  state_personas: string[];
  predicted_topK: PredictedTopEntry[];
  observed_next: string | null; // telemetry
  bucket:
    | "on_path"
    | "near_path"
    | "off_path_known"
    | "out_of_graph"
    | "no_path"
    | "perfect_match"
    | "skip_hit"
    | "no_observation";
  hit_at_1: boolean;
  hit_at_3: boolean;

  // replay beam / expectations
  paths?: CandidatePath[];
  expected_next?: string[];
  win_likelihood?: number; // snapshot proxy
};

type LatentEdgeEvidence = {
  from?: string;
  to?: string;
  supports?: boolean;
  delta_log_prob?: number;
  path_probability?: number;
  rel?: string;
};

export type LatentActivityItem = {
  id: string;
  kind: "latent";
  insert_before_step_index: number;
  step_t?: number | null;
  persona_id?: string | null;
  persona_label?: string | null;
  target_persona_id?: string | null;
  target_persona_label?: string | null;
  confidence?: number | null;
  path_probability?: number | null;
  narrative?: string | null;
  reason?: string | null;
  edge_evidence?: LatentEdgeEvidence | null;
  path_excerpt?: string[];
};

export type ActivityStoryObservedItem = {
  id: string;
  kind: "observed";
  step_index: number;
  step_t?: number | null;
  timestamp?: string | null;
  persona_id?: string | null;
  persona_label?: string | null;
  bucket?: string | null;
  engagement_meta?: any;
};

export type ActivityStoryItem = LatentActivityItem | ActivityStoryObservedItem;
type TimelineRow =
  | { key: string; type: "latent"; latent: LatentActivityItem }
  | {
      key: string;
      type: "observed";
      observed: { annotated?: AnnotatedEngagement; engagement: Engagement };
    };

type AccountBeliefThesis = {
  persons: PersonaMatchRow[];
  paths: CandidatePath[]; // existing (walk_paths)
  journey_steps?: JourneyStep[];

  // New structured learning
  learning_summary?: LearningSummary;
  learning_neighborhoods?: LearningNeighborhoods;

  baseline_paths?: CandidatePath[];
  baseline_expected_next?: string[];
  current_paths?: CandidatePath[];
  current_expected_next?: string[];
  overall_fit?: OverallFit | null;
  latent_activity?: LatentActivityItem[];
  activity_story?: ActivityStoryItem[];
  storyline?: StorylinePayload | null;
};

type StoryTabState = Record<string, "activity" | "story">;
type StoryEditState = Record<string, Record<string, string>>;

// -----------------------------------------
// Helpers
// -----------------------------------------
function toIsoZ(local: string) {
  try {
    const d = new Date(local);
    return new Date(d.getTime() - d.getTimezoneOffset() * 60000)
      .toISOString()
      .replace(/\.\d{3}Z$/, "Z");
  } catch {
    return new Date().toISOString();
  }
}
function nowLocalForInput() {
  const d = new Date();
  const tz = new Date(d.getTime() - d.getTimezoneOffset() * 60000);
  return tz.toISOString().slice(0, 16);
}
function pad(n: number) {
  return n.toString().padStart(2, "0");
}
function toLocalInputFromDate(d: Date) {
  const yr = d.getFullYear();
  const mo = pad(d.getMonth() + 1);
  const da = pad(d.getDate());
  const hr = pad(d.getHours());
  const mi = pad(d.getMinutes());
  return `${yr}-${mo}-${da}T${hr}:${mi}`;
}
function parseUserDatetimeText(text: string): string | null {
  if (!text) return null;
  const t = text.trim();
  const today = new Date();

  if (/^now$/i.test(t)) return toLocalInputFromDate(new Date());

  if (/^today\b/i.test(t)) {
    const m = t.match(/today\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?/i);
    if (m) {
      let h = parseInt(m[1], 10);
      const min = m[2] ? parseInt(m[2], 10) : 0;
      const ap = (m[3] || "").toLowerCase();
      if (ap === "pm" && h < 12) h += 12;
      if (ap === "am" && h === 12) h = 0;
      const d = new Date(
        today.getFullYear(),
        today.getMonth(),
        today.getDate(),
        h,
        min
      );
      return toLocalInputFromDate(d);
    }
    return toLocalInputFromDate(
      new Date(
        today.getFullYear(),
        today.getMonth(),
        today.getDate(),
        9,
        0
      )
    );
  }

  if (/^yesterday\b/i.test(t)) {
    const y = new Date(today.getTime() - 24 * 60 * 60 * 1000);
    const m = t.match(/yesterday\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?/i);
    if (m) {
      let h = parseInt(m[1], 10);
      const min = m[2] ? parseInt(m[2], 10) : 0;
      const ap = (m[3] || "").toLowerCase();
      if (ap === "pm" && h < 12) h += 12;
      if (ap === "am" && h === 12) h = 0;
      const d = new Date(
        y.getFullYear(),
        y.getMonth(),
        y.getDate(),
        h,
        min
      );
      return toLocalInputFromDate(d);
    }
    return toLocalInputFromDate(
      new Date(y.getFullYear(), y.getMonth(), y.getDate(), 9, 0)
    );
  }

  const isoLike = t.replace(/\//g, "-").replace(" ", "T");
  const dt1 = new Date(isoLike);
  if (!isNaN(dt1.getTime())) return toLocalInputFromDate(dt1);

  const hm = t.match(/^(\d{1,2})(?::(\d{2}))?\s*(am|pm)?$/i);
  if (hm) {
    let h = parseInt(hm[1], 10);
    const min = hm[2] ? parseInt(hm[2], 10) : 0;
    const ap = (hm[3] || "").toLowerCase();
    if (ap === "pm" && h < 12) h += 12;
    if (ap === "am" && h === 12) h = 0;
    const d = new Date(
      today.getFullYear(),
      today.getMonth(),
      today.getDate(),
      h,
      min
    );
    return toLocalInputFromDate(d);
  }

  const dt2 = new Date(t);
  if (!isNaN(dt2.getTime())) return toLocalInputFromDate(dt2);
  return null;
}

function labelForAccount(a: TargetAccount) {
  const status = a.deal_status ? ` [${a.deal_status}]` : "";
  return `${a.account_name}${status}`;
}

function engagementKey(e: Engagement) {
  return `${e.account_id}||${e.timestamp}||${e.raw_activity}`;
}

const sortAscByTimestamp = (a: Engagement, b: Engagement) =>
  new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime();

function fmtWhen(zIso: string) {
  const d = new Date(zIso);
  return d.toLocaleString();
}

function norm(s?: string | null) {
  return (s || "").trim().toLowerCase();
}

function titleCase(s?: string | null) {
  return (s || "")
    .split(/[\s_/|]+/)
    .filter(Boolean)
    .map((w) =>
      w[0] ? w[0].toUpperCase() + w.slice(1).toLowerCase() : ""
    )
    .join(" ");
}

// id looks like "title|department|seniority"
function personaLabelFromId(rawId?: PersonaIdentifier) {
  const id = normalizePersonaId(rawId);
  if (!id) return "";
  const override = personaLabelOverrides.get(id);
  if (override) return override;
  if (id.includes("|")) {
    const [t = "", d = "", s = ""] = id.split("|");
    const label = [titleCase(t), titleCase(d), titleCase(s)]
      .filter(Boolean)
      .join(" | ");
    if (label) {
      return label;
    }
  }
  if (id.includes(":")) {
    const tail = id.split(":").pop() || id;
    if (/^[a-f0-9-]+$/i.test(tail)) {
      return tail;
    }
    return titleCase(tail.replace(/[_/|]+/g, " "));
  }
  return titleCase(id.replace(/[_/|]+/g, " "));
}

function normalizeStorylinePersona(raw?: string | null) {
  if (!raw) return raw || null;
  const label = personaLabelFromId(raw);
  return label || raw;
}

function decorateStoryline(
  storyline?: StorylinePayload | null
): StorylinePayload | null {
  if (!storyline) return storyline ?? null;
  const convertList = (list?: string[]) =>
    (list || []).map((entry) => normalizeStorylinePersona(entry) || entry);
  return {
    ...storyline,
    stakeholders: {
      early: convertList(storyline.stakeholders?.early),
      mid: convertList(storyline.stakeholders?.mid),
      late: convertList(storyline.stakeholders?.late),
    },
    nodes: (storyline.nodes || []).map((node) => ({
      ...node,
      persona: node.persona
        ? normalizeStorylinePersona(node.persona) || node.persona
        : node.persona,
    })),
  };
}

function fmtScore(x?: number | null) {
  return typeof x === "number" ? x.toFixed(3) : "";
}

function fmtPercent(
  value?: number | null,
  {
    sign = false,
    decimals = 0,
    inputIsFraction = false,
  }: { sign?: boolean; decimals?: number; inputIsFraction?: boolean } = {}
) {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  const numeric = inputIsFraction ? value * 100 : value;
  if (!Number.isFinite(numeric)) return "—";
  const rounded = Number(numeric.toFixed(decimals));
  const prefix = sign && rounded > 0 ? "+" : "";
  const text = rounded
    .toFixed(decimals)
    .replace(/-0(\.0+)?$/, "0")
    .replace(/^-0$/, "0");
  return `${prefix}${text}%`;
}

function fmtCount(value?: number | null) {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return Math.round(Number(value)).toLocaleString();
}

type LevelMeta = {
  label: string;
  helper: string;
};

function describePathConfidence(score?: number | null): LevelMeta {
  if (score === null || score === undefined) {
    return {
      label: "Unknown",
      helper: "Model did not emit a probability for this engagement.",
    };
  }
  if (score >= 0.7) {
    return {
      label: "High",
      helper: "Model was confident this persona would appear next on the best-fit path.",
    };
  }
  if (score >= 0.4) {
    return {
      label: "Medium",
      helper: "Persona was plausible on the predicted path, but not guaranteed.",
    };
  }
  return {
    label: "Low",
    helper: "Model saw this persona as unlikely in this slot.",
  };
}

function describeSurpriseLevel(score?: number | null): LevelMeta {
  if (score === null || score === undefined) {
    return {
      label: "Unknown",
      helper: "No surprise score recorded for this event.",
    };
  }
  if (score >= 0.7) {
    return {
      label: "High",
      helper: `Highly surprising (${fmtPercent(score, {
        inputIsFraction: true,
        decimals: 0,
      })}). Worth investigating.`,
    };
  }
  if (score >= 0.4) {
    return {
      label: "Medium",
      helper: `Moderately surprising (${fmtPercent(score, {
        inputIsFraction: true,
        decimals: 0,
      })}).`,
    };
  }
  return {
    label: "Low",
    helper: `Close to expected (${fmtPercent(score, {
      inputIsFraction: true,
      decimals: 0,
    })}).`,
  };
}

function verdictFromClassification(
  classification: EngagementBeliefAnnotation["classification"],
  note?: string
): string {
  switch (classification) {
    case "expected":
      return "As expected and in the right order.";
    case "jump_ahead":
      return "Expected persona, but earlier or later than we thought.";
    case "off_path":
      return "Unexpected persona — this is where the deal veered off our predicted script.";
    case "no_persona":
      return "We couldn't map this engagement to a known persona yet.";
    case "no_path":
    default:
      return note?.trim()
        ? note
        : "We don't have a confident belief path for this engagement yet.";
  }
}

type ImpactSummary = {
  badge: {
    label: string;
    color: "default" | "primary" | "secondary" | "warning";
  } | null;
  totalDelta: number;
  bullets: string[];
  tooltipLines: string[];
};

function summarizeImpact(
  belief: EngagementBeliefAnnotation,
  engagement: Engagement,
  match?: PersonaMatchRow
): ImpactSummary {
  const edges = belief.impact?.edges || [];
  const totalDelta = edges.reduce(
    (acc, edge) => acc + Math.abs(edge.deltaLogProb ?? 0),
    0
  );

  let badge: ImpactSummary["badge"] = null;
  if (totalDelta >= 0.25) {
    badge = { label: `High belief shift (+${totalDelta.toFixed(2)})`, color: "secondary" };
  } else if (totalDelta >= 0.08) {
    badge = { label: `Moderate belief shift (+${totalDelta.toFixed(2)})`, color: "primary" };
  } else if (totalDelta > 0) {
    badge = { label: `Minimal belief shift (+${totalDelta.toFixed(2)})`, color: "default" };
  }

  const bullets: string[] = [];

  edges.slice(0, 3).forEach((edge) => {
    const target =
      edge.toLabel || personaLabelFromId(edge.to) || "downstream persona";
    const actor =
      engagement.actor?.name ||
      match?.graph_persona_label ||
      personaLabelFromId(edge.from) ||
      "This engagement";
    const verb = edge.supports ? "strengthened" : "raised doubts about";
    bullets.push(`${verb[0].toUpperCase() + verb.slice(1)} ${target}'s belief via ${actor}.`);
  });

  if (bullets.length === 0 && belief.classification === "off_path") {
    bullets.push(
      "Unexpected persona involvement — validate why this role leaned in now."
    );
  } else if (bullets.length === 0 && belief.classification === "no_persona") {
    bullets.push(
      "Weak signal: model could not confidently tie this to a known persona."
    );
  }

  const tooltipLines =
    edges.length > 0
      ? edges.map((edge) => {
          const delta =
            typeof edge.deltaLogProb === "number"
              ? `Δ log p ${edge.deltaLogProb >= 0 ? "+" : ""}${edge.deltaLogProb.toFixed(3)}`
              : "";
          return `${edge.fromLabel || edge.from} → ${edge.toLabel || edge.to} ${delta}`;
        })
      : [];

  return { badge, totalDelta, bullets: bullets.slice(0, 3), tooltipLines };
}

function personaChipLabel(
  engagement: Engagement,
  bestLabel?: string | null
): string | null {
  const pieces: string[] = [];
  if (engagement.actor?.name) {
    pieces.push(engagement.actor.name);
  }
  if (engagement.actor?.title) {
    pieces.push(engagement.actor.title);
  }
  if (bestLabel) {
    pieces.push(`(${bestLabel})`);
  }
  if (pieces.length === 0) return bestLabel || null;
  return pieces.join(" — ");
}

function latentHintsFromIndicates(
  indicates?: {
    upstream?: EngagementEdgeIndication[];
    handoff?: EngagementEdgeIndication[];
    downstream?: EngagementEdgeIndication[];
  }
): string[] {
  if (!indicates) return [];
  const bands: Array<{ key: keyof typeof indicates; label: string }> = [
    { key: "upstream", label: "Guessed previous step" },
    { key: "handoff", label: "Likely handoff" },
    { key: "downstream", label: "Downstream impact" },
  ];
  const lines: string[] = [];
  bands.forEach(({ key, label }) => {
    const entries = indicates[key];
    if (!entries || entries.length === 0) return;
    const first = entries[0];
    const from =
      first.fromLabel || (first.fromId ? personaLabelFromId(first.fromId) : null);
    const to =
      first.toLabel || (first.toId ? personaLabelFromId(first.toId) : first.edge);
    const reason = first.reason || "";
    const textParts = [
      `${label}:`,
      [from, to].filter(Boolean).join(" → ") || first.edge,
      reason,
    ]
      .filter(Boolean)
      .join(" ");
    lines.push(textParts);
  });
  return lines;
}

function deriveLatentItemsFromBelief(
  belief: EngagementBeliefAnnotation | undefined,
  insertIdx: number
): LatentActivityItem[] {
  if (!belief || !belief.indicates) return [];
  const entries: LatentActivityItem[] = [];
  (["upstream", "handoff"] as const).forEach((band, order) => {
    const list = belief.indicates?.[band];
    if (!list || list.length === 0) return;
    const entry = list[0];
    const personaId = entry.fromId || entry.edge?.split("→")[0]?.trim() || null;
    const targetId = entry.toId || entry.edge?.split("→")[1]?.trim() || null;
    const personaLabel =
      entry.fromLabel || (personaId ? personaLabelFromId(personaId) : null);
    const targetLabel =
      entry.toLabel || (targetId ? personaLabelFromId(targetId) : null);
    entries.push({
      id: `derived-latent-${band}-${insertIdx}-${order}`,
      kind: "latent",
      insert_before_step_index: insertIdx,
      step_t: null,
      persona_id: personaId,
      persona_label: personaLabel,
      target_persona_id: targetId,
      target_persona_label: targetLabel,
      confidence: entry.confidence ?? null,
      narrative:
        entry.reason ||
        (personaLabel
          ? `${personaLabel} likely prepared ${targetLabel || "the next persona"}`
          : "Hidden persona likely advanced the context."),
      reason: entry.reason || `Inferred ${band} step`,
      edge_evidence: null,
      path_excerpt: [],
    });
  });
  return entries;
}

// Attach a match row to an engagement's actor
function findMatchForEngagement(
  e: Engagement,
  matches?: PersonaMatchRow[]
): PersonaMatchRow | undefined {
  if (!matches || matches.length === 0) return undefined;
  const nm = norm(e.actor?.name);
  const roleOrTitle = norm(e.actor?.title || e.actor?.role);
  const dept = norm(e.actor?.department);
  const snr = norm(e.actor?.seniority);

  return (
    matches.find(
      (m) => norm(m.name) === nm && (dept ? norm(m.department) === dept : true)
    ) ||
    matches.find((m) => norm(m.name) === nm) ||
    matches.find(
      (m) =>
        norm(m.title) === roleOrTitle &&
        (dept ? norm(m.department) === dept : true) &&
        (snr ? norm(m.seniority) === snr : true)
    ) ||
    undefined
  );
}

function channelWeight(channel?: string | null): number {
  const c = (channel || "").toLowerCase();
  if (!c) return 0.4;
  if (c.includes("display") || c.includes("social")) return 0.2;
  if (c.includes("email") || c.includes("blog")) return 0.3;
  if (c.includes("download") || c.includes("webinar")) return 0.5;
  if (c.includes("call") || c.includes("demo")) return 0.8;
  if (c.includes("poc") || c.includes("security") || c.includes("legal"))
    return 0.9;
  if (c.includes("e-sign") || c.includes("signature") || c.includes("contract"))
    return 1.0;
  return 0.4;
}

// -----------------------------------------
// Path scoring (heuristic fallback)
// -----------------------------------------
type ScoredPath = CandidatePath & { probability: number; score: number };

function scorePathsForObserved(
  candidatePaths: CandidatePath[],
  observed: ObservedPersona[],
  alphaHit = 1.0,
  betaOff = 0.7,
  gammaMissed = 0.5
): ScoredPath[] {
  if (!candidatePaths.length) return [];
  const scores: number[] = [];
  const out: ScoredPath[] = [];

  for (const path of candidatePaths) {
    const expected = path.personas || [];
    let idx = 0;
    let hits = 0;
    let off = 0;

    for (const { id, weight } of observed) {
      if (idx < expected.length && expected.slice(idx).includes(id)) {
        const j = expected.indexOf(id, idx);
        hits += weight;
        idx = j + 1;
      } else {
        off += weight;
      }
    }

    const missed = Math.max(0, expected.length - Math.round(hits));
    const prior = path.score || 0;
    const post = prior + alphaHit * hits - betaOff * off - gammaMissed * missed;

    scores.push(post);
    out.push({ ...path, score: post, probability: 0 });
  }

  const max = Math.max(...scores);
  const exps = scores.map((s) => Math.exp(s - max));
  const Z = exps.reduce((a, b) => a + b, 0) || 1;

  return out.map((p, i) => ({ ...p, probability: exps[i] / Z }));
}

function expectedNextPersonaFromBest(
  best: ScoredPath | null,
  observed: ObservedPersona[]
): string | null {
  if (!best) return null;
  const expected = best.personas || [];
  let idx = 0;

  for (const { id } of observed) {
    if (idx < expected.length && expected.slice(idx).includes(id)) {
      const j = expected.indexOf(id, idx);
      idx = j + 1;
    }
  }

  return idx < expected.length ? expected[idx] : null;
}

// -----------------------------------------
// Backend-aware annotation helpers
// -----------------------------------------
function qualityFromPredictedList(
  predicted_topK: PredictedTopEntry[] | undefined,
  observed: string | null
): number | null {
  if (!observed || !Array.isArray(predicted_topK) || predicted_topK.length === 0)
    return null;
  const flattened = predicted_topK.map((entry) =>
    typeof entry === "string"
      ? entry
      : entry?.persona || entry?.id || entry?.persona_id || ""
  );
  const idx = flattened.indexOf(observed);
  if (idx === -1) return 0.15; // unseen
  if (idx === 0) return 1.0; // perfect
  if (idx < 3) return 0.7; // top-3
  if (idx < 5) return 0.45; // top-5
  return 0.25; // low, but present
}

function mapBucketToClassification(
  bucket?: string
): EngagementBeliefAnnotation["classification"] {
  switch (bucket) {
    // v2/v3 style buckets from replay_learn_persona_paths
    case "on_path":
      return "expected";
    case "near_path":
      return "jump_ahead";
    case "off_path_known":
    case "out_of_graph":
      return "off_path";
    case "no_path":
      return "no_path";

    // older names (kept for safety)
    case "perfect_match":
      return "expected";
    case "skip_hit":
      return "jump_ahead";

    default:
      return "no_path";
  }
}

function _indexAndNextFromBest(
  best: ScoredPath | null,
  observed: ObservedPersona[]
) {
  if (!best) return { idx: 0, next: null as string | null };
  const expected = best.personas || [];
  let idx = 0;
  for (const { id } of observed) {
    if (idx < expected.length && expected.slice(idx).includes(id)) {
      const j = expected.indexOf(id, idx);
      idx = j + 1;
    }
  }
  return { idx, next: idx < expected.length ? expected[idx] : null };
}

// -----------------------------------------
// String → EdgeUpdate (fallback helper)
// -----------------------------------------
function parseEdgeUpdateFromString(s: string): EdgeUpdate {
  const raw = (s || "").trim();
  const defaultEdge: EdgeUpdate = {
    u: raw,
    v: raw,
    rel: "unknown",
    delta: 0,
  };
  if (!raw) return defaultEdge;

  // crude parse: "A —[rel]→ B | Δ=... | band=... | conf=..."
  const [lhs, ...rest] = raw.split("|");
  const main = lhs.trim();
  const relMatch = main.match(/—\[(.+?)\]→/);
  const [uLabel, vLabel] = (() => {
    const parts = main.split("—");
    if (parts.length < 2) return [main, main];
    const left = parts[0].trim();
    const rightRaw = parts[1];
    const rightParts = rightRaw.split("→");
    const right = rightParts.length > 1 ? rightParts[1].trim() : rightRaw.trim();
    return [left, right];
  })();

  let delta = 0;
  let band: LearningBand | undefined;
  let conf: number | undefined;

  const restStr = rest.join("|");
  const mDelta = restStr.match(/Δ\s*=\s*([+-]?\d*\.?\d+)/i);
  if (mDelta) {
    const val = parseFloat(mDelta[1]);
    if (!Number.isNaN(val)) delta = val;
  }
  const mBand = restStr.match(/band\s*=\s*(upstream|handoff|downstream)/i);
  if (mBand) band = mBand[1].toLowerCase() as LearningBand;
  const mConf = restStr.match(/conf\s*=\s*([0-9]*\.?[0-9]+)/i);
  if (mConf) {
    const val = parseFloat(mConf[1]);
    if (!Number.isNaN(val)) conf = val;
  }

  return {
    u: uLabel,
    v: vLabel,
    rel: relMatch?.[1] || "unknown",
    u_label: uLabel,
    v_label: vLabel,
    delta,
    band,
    confidence: conf,
    rationale: raw,
  };
}

// ---- insights helpers -------------------------------------------------------
function surpriseFrom(
  classification: EngagementBeliefAnnotation["classification"],
  quality: number | null
) {
  // simple mapping: unexpected events are more surprising; quality modulates
  const base =
    classification === "expected"
      ? 0.1
      : classification === "jump_ahead"
      ? 0.45
      : classification === "off_path"
      ? 0.8
      : 0.3;
  const q = typeof quality === "number" ? quality : 0.4;
  return Math.max(0, Math.min(1, base * (1 + (1 - q) * 0.5)));
}

function bandedIndicationsFromTopEdges(edges?: EdgeUpdate[]) {
  const out: {
    upstream: EngagementEdgeIndication[];
    handoff: EngagementEdgeIndication[];
    downstream: EngagementEdgeIndication[];
  } = {
    upstream: [],
    handoff: [],
    downstream: [],
  };
  if (!Array.isArray(edges)) return out;
  for (const e of edges.slice(0, 6)) {
    const band = (e.band || "upstream") as LearningBand;
    const edgeLabel = `${e.u_label || e.u} → ${e.v_label || e.v} (${e.rel})`;
    out[band].push({
      edge: edgeLabel,
      reason: e.rationale,
      band,
      fromId: e.u ? String(e.u) : null,
      toId: e.v ? String(e.v) : null,
      fromLabel: e.u_label || null,
      toLabel: e.v_label || null,
      rel: e.rel || null,
      delta: typeof e.delta === "number" ? e.delta : null,
      confidence: typeof e.confidence === "number" ? e.confidence : null,
    });
  }
  return out;
}

function indicationEdgesToImpacts(
  indicates?:
    | {
        upstream?: EngagementEdgeIndication[];
        handoff?: EngagementEdgeIndication[];
        downstream?: EngagementEdgeIndication[];
      }
    | undefined
): EngagementEdgeImpact[] {
  if (!indicates) return [];
  const parsed: EngagementEdgeImpact[] = [];
  const seen = new Set<string>();

  const parseLabelParts = (label: string) => {
    const [leftRaw = "", rightRaw = ""] = label.split("→");
    const left = leftRaw.trim();
    const rightClean = rightRaw.replace(/\(.*?\)\s*$/, "").trim();
    return { left, right: rightClean };
  };

  (["upstream", "handoff", "downstream"] as const).forEach((band) => {
    const entries = indicates[band] || [];
    entries.forEach((indication) => {
      const { left, right } = parseLabelParts(indication.edge || "");
      const fromId =
        (indication.fromId && String(indication.fromId)) || normalizePersonaId(left);
      const toId =
        (indication.toId && String(indication.toId)) || normalizePersonaId(right);
      if (!fromId || !toId) return;
      const key = `${fromId}||${toId}`;
      if (seen.has(key)) return;
      seen.add(key);
      parsed.push({
        from: fromId,
        to: toId,
        fromLabel: indication.fromLabel || left || undefined,
        toLabel: indication.toLabel || right || undefined,
        supports: (indication.delta ?? 0) >= 0,
        deltaLogProb: indication.delta ?? null,
        pathProbability: undefined,
      });
    });
  });

  return parsed;
}

// -----------------------------------------
// Backend-aware annotation (BLENDED w/ virtual-advance on off-path)
// -----------------------------------------
type AnnotatedEngagement = {
  engagement: Engagement;
  match?: PersonaMatchRow;
  belief: EngagementBeliefAnnotation;
};

function annotateEngagementsForAccount(
  engagements: Engagement[],
  persons: PersonaMatchRow[],
  paths: CandidatePath[],
  journeySteps?: JourneyStep[],
  topEdges?: EdgeUpdate[]
): AnnotatedEngagement[] {
  if (!engagements.length) return [];

  const annotated: {
    engagement: Engagement;
    match?: PersonaMatchRow;
    belief: EngagementBeliefAnnotation;
  }[] = [];
  const observed: ObservedPersona[] = [];

  const steps = Array.isArray(journeySteps)
    ? journeySteps.slice().sort((a, b) => a.t - b.t)
    : [];
  let replayIdx = 0;

  // track to avoid stacking multiple virtual nudges for the same “next”
  let lastVirtualExpected: string | null = null;
  let lastVirtualIdx = -1;

  for (const e of engagements) {
    const match = findMatchForEngagement(e, persons);
    const actualPersona =
      match?.graph_persona_node_id || match?.canonical_persona_best || null;
    const backendStep = steps[replayIdx] ?? null;

    // Heuristic state BEFORE
    const scoredBefore = paths.length ? scorePathsForObserved(paths, observed) : [];
    const bestBefore =
      scoredBefore.length
        ? scoredBefore.reduce((a, b) => (a.probability >= b.probability ? a : b))
        : null;
    const { idx: heuristicIdx, next: heuristicExpectedBefore } =
      _indexAndNextFromBest(bestBefore, observed);

    const backendExpectedBefore =
      (Array.isArray(backendStep?.expected_next) && backendStep?.expected_next?.length
        ? normalizePersonaId(backendStep.expected_next[0])
        : null) ||
      (Array.isArray(backendStep?.predicted_topK) && backendStep?.predicted_topK?.length
        ? normalizePersonaId(backendStep.predicted_topK[0])
        : null);

    // Prefer heuristic to drive UI; add backend as hint in note if different
    const expectedBefore = heuristicExpectedBefore || backendExpectedBefore;

    let classification: EngagementBeliefAnnotation["classification"] = "no_path";
    let note = steps.length ? "" : "No belief paths yet for this product.";
    let quality: number | null = null;

    if (!paths.length && !steps.length) {
      classification = actualPersona ? "expected" : "no_path";
    } else if (!actualPersona) {
      classification = "no_persona";
      note = "No matched persona for this engagement (neutral).";
    } else {
      // Backend exact match → trust backend bucket & advance pointer
      if (backendExpectedBefore && actualPersona === backendExpectedBefore) {
        classification = mapBucketToClassification(backendStep?.bucket);
        quality = qualityFromPredictedList(
          backendStep?.predicted_topK,
          backendStep?.observed_next || null
        );
        replayIdx = Math.min(replayIdx + 1, Math.max(steps.length - 1, 0));
        // reset virtual tracker on real progress
        lastVirtualExpected = null;
        lastVirtualIdx = -1;
      } else {
        // Heuristic comparison
        if (!bestBefore || !bestBefore.personas?.length) {
          classification = "expected";
        } else if (expectedBefore && actualPersona === expectedBefore) {
          classification = "expected";
          quality = 1.0;
        } else if (bestBefore.personas.includes(actualPersona)) {
          classification = "jump_ahead";
          quality = 0.6;
          note = "On path but earlier/later than predicted.";
        } else {
          classification = "off_path";
          quality = 0.25;
          if (backendExpectedBefore && backendExpectedBefore !== expectedBefore) {
            note = `Persona did not match backend’s next. (backend next: ${personaLabelFromId(
              backendExpectedBefore
            )})`;
          } else if (backendExpectedBefore) {
            note = "Persona did not match the next expected node in backend replay.";
          }

          // Single virtual nudge per unique expectedBefore/position
          if (
            heuristicExpectedBefore &&
            (lastVirtualExpected !== heuristicExpectedBefore ||
              lastVirtualIdx !== heuristicIdx)
          ) {
            observed.push({ id: heuristicExpectedBefore, weight: 0.03 }); // tiny, once
            lastVirtualExpected = heuristicExpectedBefore;
            lastVirtualIdx = heuristicIdx;
          }
        }
      }
    }

    // Expected AFTER: recompute heuristically after adding evidence
    const newObserved = [...observed];
    if (actualPersona)
      newObserved.push({
        id: actualPersona,
        weight: channelWeight(e.channel),
      });
    const scoredAfter = scorePathsForObserved(paths, newObserved);
    const bestAfter =
      scoredAfter.length
        ? scoredAfter.reduce((a, b) => (a.probability >= b.probability ? a : b))
        : null;
    const { next: nextExpectedAfter } = _indexAndNextFromBest(
      bestAfter,
      newObserved
    );

    const indicates =
      classification === "off_path" || classification === "jump_ahead"
        ? bandedIndicationsFromTopEdges(topEdges)
        : undefined;

    const stepMetrics =
      backendStep && typeof backendStep.metrics === "object"
        ? backendStep.metrics
        : undefined;
    const edgeEvidence = Array.isArray(stepMetrics?.edge_evidence)
      ? stepMetrics.edge_evidence
      : [];
    const impactEdgesFromEvidence = edgeEvidence
      .map((edge: any) => {
        if (!edge) return null;
        const from = edge.from || edge.u || edge.source;
        const to = edge.to || edge.v || edge.target;
        if (!from || !to) return null;
        const deltaLog =
          typeof edge.delta_log_prob === "number"
            ? edge.delta_log_prob
            : typeof edge.deltaLogProb === "number"
            ? edge.deltaLogProb
            : null;
        const pathProb =
          typeof edge.path_probability === "number"
            ? edge.path_probability
            : typeof edge.pathProbability === "number"
            ? edge.pathProbability
            : null;
        return {
          from: String(from),
          to: String(to),
          fromLabel: personaLabelFromId(String(from)),
          toLabel: personaLabelFromId(String(to)),
          supports: !!edge.supports,
          deltaLogProb: deltaLog,
          pathProbability: pathProb,
        };
      })
      .filter(Boolean) as EngagementEdgeImpact[];

    const indicationEdges = indicationEdgesToImpacts(indicates);
    const impactEdges: EngagementEdgeImpact[] = (() => {
      if (impactEdgesFromEvidence.length === 0 && indicationEdges.length > 0) {
        return indicationEdges;
      }
      if (impactEdgesFromEvidence.length > 0 && indicationEdges.length > 0) {
        const seen = new Set(
          impactEdgesFromEvidence.map((edge) => `${edge.from}||${edge.to}`)
        );
        indicationEdges.forEach((edge) => {
          const key = `${edge.from}||${edge.to}`;
          if (!seen.has(key)) {
            seen.add(key);
            impactEdgesFromEvidence.push(edge);
          }
        });
      }
      return impactEdgesFromEvidence;
    })();
    const hiddenStatePrior =
      stepMetrics && typeof stepMetrics.hidden_state_prior === "object"
        ? stepMetrics.hidden_state_prior
        : undefined;
    const hiddenStatePosterior =
      stepMetrics && typeof stepMetrics.hidden_state_posterior === "object"
        ? stepMetrics.hidden_state_posterior
        : undefined;

    annotated.push({
      engagement: e,
      match,
      belief: {
        expectedBefore,
        actualPersona,
        classification,
        note: note || "",
        nextExpectedAfter,
        quality,
        surpriseScore: surpriseFrom(classification, quality ?? null),
        indicates,
        impact:
          (impactEdges.length > 0 ||
            (hiddenStatePrior && Object.keys(hiddenStatePrior).length > 0) ||
            (hiddenStatePosterior &&
              Object.keys(hiddenStatePosterior).length > 0))
            ? {
                edges: impactEdges,
                hiddenStatePrior,
                hiddenStatePosterior,
              }
            : undefined,
      },
    });

    // Append the real observation last
    if (actualPersona) {
      observed.push({
        id: actualPersona,
        weight: channelWeight(e.channel),
      });
    }
  }

  return annotated;
}

// -----------------------------------------
// Section model
// -----------------------------------------
type SectionForm = {
  account_id: string;
  timestampLocal: string;
  timestampText: string;
  timestampErr: string;
  actorName: string;
  actorRole: string;
  actorDept: string;
  actorSeniority: string;
  source: "marketing" | "sales" | "cs" | "product" | "other";
  rawActivity: string;
  inferred: boolean;
  channel: string;
  assetId: string;
  added: Engagement[];

  tab: number; // 0 = Manual, 1 = Sync HS
  hsText: string;
  hsTextErr: string;
};

function newSection(account_id = ""): SectionForm {
  const initLocal = nowLocalForInput();
  return {
    account_id,
    timestampLocal: initLocal,
    timestampText: initLocal.replace("T", " "),
    timestampErr: "",
    actorName: "",
    actorRole: "",
    actorDept: "",
    actorSeniority: "",
    source: "marketing",
    rawActivity: "",
    inferred: false,
    channel: "",
    assetId: "",
    added: [],
    tab: 0,
    hsText: "",
    hsTextErr: "",
  };
}

// Incremental Learnings Card (per account)
// -----------------------------------------
function IncrementalLearningsCard({
  thesis,
  annotatedFeed,
}: {
  thesis?: AccountBeliefThesis;
  annotatedFeed: AnnotatedEngagement[];
}) {
  const [showAdvanced, setShowAdvanced] = useState(false);
  const ls = thesis?.learning_summary;

  const personaInferences: PersonaPathThesis[] =
    ls?.persona_graph_inferences || [];
  const fullInferences: PersonaPathThesis[] =
    ls?.full_graph_inferences || [];

  const hasAny =
    (personaInferences && personaInferences.length > 0) ||
    (fullInferences && fullInferences.length > 0) ||
    annotatedFeed.length > 0;

  const totalEvents = annotatedFeed.length;
  const classificationCounts = annotatedFeed.reduce((acc, item) => {
    const key = item.belief.classification;
    acc[key] = (acc[key] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);

  const expectedCount = classificationCounts.expected ?? 0;
  const partialCount = classificationCounts.jump_ahead ?? 0;
  const offPathCount = classificationCounts.off_path ?? 0;
  const noPersonaCount =
    (classificationCounts.no_persona ?? 0) + (classificationCounts.no_path ?? 0);

  const onPathRate = totalEvents ? expectedCount / totalEvents : null;
  const partialRate = totalEvents ? partialCount / totalEvents : null;
  const offPathRate = totalEvents ? offPathCount / totalEvents : null;

  const surprisingEngagements = annotatedFeed
    .filter((item) => (item.belief.surpriseScore ?? 0) >= 0.6)
    .slice(0, 4);

  const personaQualityMap = new Map<
    string,
    { count: number; qualitySum: number }
  >();
  annotatedFeed.forEach((item) => {
    const personaId = item.belief.actualPersona;
    if (!personaId) return;
    const entry =
      personaQualityMap.get(personaId) || { count: 0, qualitySum: 0 };
    entry.count += 1;
    if (typeof item.belief.quality === "number") {
      entry.qualitySum += item.belief.quality;
    }
    personaQualityMap.set(personaId, entry);
  });

  const personaFitSummaries =
    thesis?.persons?.map((p) => {
      const personaId =
        p.graph_persona_node_id || p.canonical_persona_best || null;
      const label =
        p.graph_persona_label ||
        p.canonical_persona_label ||
        (personaId ? personaLabelFromId(personaId) : p.name || "Persona");
      const immediateFit =
        typeof p.graph_persona_score === "number"
          ? p.graph_persona_score
          : typeof p.canonical_persona_score === "number"
          ? p.canonical_persona_score
          : null;
      const engagementStats = personaId
        ? personaQualityMap.get(personaId)
        : undefined;
      const neighborhoodFit =
        engagementStats && engagementStats.count > 0
          ? engagementStats.qualitySum / engagementStats.count
          : null;
      return {
        personaId,
        label,
        immediateFit,
        neighborhoodFit,
        sampleCount: engagementStats?.count ?? 0,
      };
    }) || [];

  const poorPersonaFits = personaFitSummaries
    .filter(
      (p) =>
        (p.immediateFit ?? 1) < 0.35 ||
        ((p.neighborhoodFit ?? 1) < 0.45 && (p.neighborhoodFit ?? 0) > 0) ||
        (p.sampleCount > 0 && (p.neighborhoodFit ?? 0) < 0.5)
    )
    .slice(0, 4);

  const engagementImpactRows = annotatedFeed.map(({ engagement, belief }) => {
    const edges = belief.impact?.edges || [];
    const impactSummary =
      edges.length === 0
        ? "No significant edge impact recorded."
        : edges
            .slice(0, 3)
            .map((edge) => {
              const fromLabel =
                edge.fromLabel || personaLabelFromId(edge.from) || edge.from;
              const toLabel =
                edge.toLabel || personaLabelFromId(edge.to) || edge.to;
              const delta =
                typeof edge.deltaLogProb === "number"
                  ? `Δlog p ${edge.deltaLogProb >= 0 ? "+" : ""}${edge.deltaLogProb.toFixed(3)}`
                  : "";
              return `${fromLabel} → ${toLabel} ${edge.supports ? "(supports)" : "(contradicts)"} ${delta}`.trim();
            })
            .join("; ");
    const totalDelta = edges.reduce(
      (acc, edge) => acc + (edge.deltaLogProb ?? 0),
      0
    );
    const confidenceRaw =
      typeof belief.quality === "number"
        ? belief.quality
        : 1 - Math.min(1, belief.surpriseScore ?? 0.5);
    const confidence = Math.max(0, Math.min(1, confidenceRaw));
    return {
      engagementLabel: engagement.raw_activity || "Engagement",
      timestamp: engagement.timestamp,
      channel: engagement.channel || engagement.source || "—",
      expected: belief.expectedBefore
        ? personaLabelFromId(belief.expectedBefore)
        : "—",
      observed: belief.actualPersona
        ? personaLabelFromId(belief.actualPersona)
        : "No persona match",
      impactSummary,
      totalDelta,
      confidence,
    };
  });

  const pathFitnessSummary = totalEvents
    ? (() => {
        let descriptor = "with low confidence";
        if (onPathRate && onPathRate >= 0.55) descriptor = "reasonably well";
        else if (onPathRate && onPathRate >= 0.35) descriptor = "with mixed confidence";
        return `We predicted this account ${descriptor}: ${fmtCount(
          expectedCount
        )}/${fmtCount(totalEvents)} events on-path, ${fmtCount(
          partialCount
        )} partial jumps, ${fmtCount(offPathCount)} off-path, and ${fmtCount(
          noPersonaCount
        )} without persona matches.`;
      })()
    : "No engagements recorded yet for this account.";

  const topSurprises = surprisingEngagements.slice(0, 2);
  const weakSignals = poorPersonaFits.slice(0, 3);
  const topImpactMoments = engagementImpactRows
    .slice()
    .sort((a, b) => Math.abs(b.totalDelta) - Math.abs(a.totalDelta))
    .slice(0, 3);
  const netGraphBullets: string[] = [];
  (personaInferences || []).slice(0, 3).forEach((inf) => {
    netGraphBullets.push(
      `${inf.anchor_label || "Persona"}: ${inf.diagnosis}${
        inf.rationale ? ` (${inf.rationale})` : ""
      }`
    );
  });
  if (netGraphBullets.length === 0 && fullInferences.length > 0) {
    fullInferences.slice(0, 2).forEach((inf) =>
      netGraphBullets.push(
        `${inf.anchor_label || "Graph"}: ${inf.diagnosis}${
          inf.rationale ? ` (${inf.rationale})` : ""
        }`
      )
    );
  }

  if (!hasAny && annotatedFeed.length === 0) return null;

  return (
    <Card variant="outlined" sx={{ mt: 3, mb: 1 }}>
      <CardHeader
        title="Incremental Learnings"
        subheader="How this account's journey is updating the persona graph"
      />
      <CardContent>
        <Box sx={{ mb: 2 }}>
          <Typography variant="subtitle2" gutterBottom>
            Path prediction fitness
          </Typography>
          <Typography variant="body2">{pathFitnessSummary}</Typography>
        </Box>

        {weakSignals.length > 0 && (
          <Box sx={{ mb: 2 }}>
            <Typography variant="subtitle2" gutterBottom>
              Personas with weak signals
            </Typography>
            <Stack spacing={0.5}>
              {weakSignals.map((p) => (
                <Typography variant="body2" key={p.personaId || p.label}>
                  • {p.label}: seen {fmtCount(p.sampleCount)} time(s) with{" "}
                  {fmtPercent(p.neighborhoodFit, {
                    inputIsFraction: true,
                    decimals: 0,
                  })}{" "}
                  confidence. Confirm with the rep whether this persona truly mattered.
                </Typography>
              ))}
            </Stack>
          </Box>
        )}

        {topSurprises.length > 0 && (
          <Box sx={{ mb: 2 }}>
            <Typography variant="subtitle2" gutterBottom>
              Moments to investigate
            </Typography>
            <Stack spacing={0.5}>
              {topSurprises.map(({ engagement, belief }, idx) => (
                <Typography variant="body2" key={`surprise-${idx}`}>
                  • {engagement.raw_activity || "Engagement"} — we expected{" "}
                  {belief.expectedBefore
                    ? personaLabelFromId(belief.expectedBefore)
                    : "another persona"}
                  , but{" "}
                  {belief.actualPersona
                    ? personaLabelFromId(belief.actualPersona)
                    : "an unmapped role"}
                  {" "}took ownership ({fmtPercent(belief.surpriseScore, {
                    inputIsFraction: true,
                    decimals: 0,
                  })} surprise).
                </Typography>
              ))}
            </Stack>
          </Box>
        )}

        {topImpactMoments.length > 0 && (
          <Box sx={{ mb: 2 }}>
            <Typography variant="subtitle2" gutterBottom>
              Most belief-shifting engagements
            </Typography>
            <Stack spacing={0.5}>
              {topImpactMoments.map((row, idx) => (
                <Typography variant="body2" key={`impact-${idx}`}>
                  • {row.engagementLabel} ({fmtWhen(row.timestamp)}): {row.impactSummary}
                </Typography>
              ))}
            </Stack>
          </Box>
        )}

        {netGraphBullets.length > 0 && (
          <Box sx={{ mb: 2 }}>
            <Typography variant="subtitle2" gutterBottom>
              Net effect on the graph
            </Typography>
            <Stack spacing={0.5}>
              {netGraphBullets.map((line, idx) => (
                <Typography variant="body2" key={`net-${idx}`}>
                  • {line}
                </Typography>
              ))}
            </Stack>
          </Box>
        )}

        {personaInferences.length > 0 && (
          <Box sx={{ mb: 2 }}>
            <Typography variant="subtitle2" gutterBottom>
              How this account moved internally
            </Typography>
            <Stack spacing={0.5}>
              {personaInferences.slice(0, 5).map((inf, idx) => (
                <Typography variant="body2" key={`path-${idx}`}>
                  • {inf.diagnosis}
                </Typography>
              ))}
            </Stack>
            {personaInferences.length > 5 && (
              <Typography variant="caption" color="text.secondary">
                {personaInferences.length - 5} more persona insights available in Insights Inbox.
              </Typography>
            )}
          </Box>
        )}

        <Button
          size="small"
          onClick={() => setShowAdvanced((prev) => !prev)}
          sx={{ mt: 1 }}
        >
          {showAdvanced ? "Hide detailed tables" : "Show detailed tables"}
        </Button>

        <Collapse in={showAdvanced} unmountOnExit sx={{ mt: 2 }}>
          {totalEvents > 0 && (
            <TableContainer component={Paper} variant="outlined" sx={{ mb: 2 }}>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Outcome</TableCell>
                    <TableCell align="right">Count</TableCell>
                    <TableCell align="right">Share</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  <TableRow>
                    <TableCell>On-path</TableCell>
                    <TableCell align="right">{fmtCount(expectedCount)}</TableCell>
                    <TableCell align="right">
                      {fmtPercent(onPathRate, {
                        inputIsFraction: true,
                        decimals: 0,
                      })}
                    </TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell>Partial jumps</TableCell>
                    <TableCell align="right">{fmtCount(partialCount)}</TableCell>
                    <TableCell align="right">
                      {fmtPercent(partialRate, {
                        inputIsFraction: true,
                        decimals: 0,
                      })}
                    </TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell>Off-path</TableCell>
                    <TableCell align="right">{fmtCount(offPathCount)}</TableCell>
                    <TableCell align="right">
                      {fmtPercent(offPathRate, {
                        inputIsFraction: true,
                        decimals: 0,
                      })}
                    </TableCell>
                  </TableRow>
                  {noPersonaCount > 0 && (
                    <TableRow>
                      <TableCell>No persona match / no path</TableCell>
                      <TableCell align="right">{fmtCount(noPersonaCount)}</TableCell>
                      <TableCell align="right">
                        {fmtPercent(noPersonaCount / totalEvents, {
                          inputIsFraction: true,
                          decimals: 0,
                        })}
                      </TableCell>
                    </TableRow>
                  )}
                </TableBody>
              </Table>
            </TableContainer>
          )}

          {poorPersonaFits.length > 0 && (
            <TableContainer component={Paper} variant="outlined" sx={{ mb: 2 }}>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Persona</TableCell>
                    <TableCell align="right">Immediate fit</TableCell>
                    <TableCell align="right">Neighborhood fit</TableCell>
                    <TableCell align="right">Engagements</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {poorPersonaFits.map((p) => (
                    <TableRow key={p.personaId || p.label}>
                      <TableCell>{p.label}</TableCell>
                      <TableCell align="right">
                        {fmtPercent(p.immediateFit, {
                          inputIsFraction: true,
                          decimals: 0,
                        })}
                      </TableCell>
                      <TableCell align="right">
                        {fmtPercent(p.neighborhoodFit, {
                          inputIsFraction: true,
                          decimals: 0,
                        })}
                      </TableCell>
                      <TableCell align="right">
                        {fmtCount(p.sampleCount)}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          )}

          {surprisingEngagements.length > 0 && (
            <TableContainer component={Paper} variant="outlined" sx={{ mb: 2 }}>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Engagement</TableCell>
                    <TableCell>Expected persona</TableCell>
                    <TableCell>Observed persona</TableCell>
                    <TableCell align="right">Surprise</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {surprisingEngagements.map(({ engagement, belief }, idx) => (
                    <TableRow key={`${engagement.account_id}-${idx}`}>
                      <TableCell sx={{ maxWidth: 320 }}>
                        {engagement.raw_activity || "Engagement"}
                      </TableCell>
                      <TableCell>
                        {belief.expectedBefore
                          ? personaLabelFromId(belief.expectedBefore)
                          : "—"}
                      </TableCell>
                      <TableCell>
                        {belief.actualPersona
                          ? personaLabelFromId(belief.actualPersona)
                          : "No persona match"}
                      </TableCell>
                      <TableCell align="right">
                        {fmtPercent(belief.surpriseScore, {
                          inputIsFraction: true,
                          decimals: 0,
                        })}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          )}

          {engagementImpactRows.length > 0 && (
            <TableContainer component={Paper} variant="outlined">
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Engagement</TableCell>
                    <TableCell>Channel</TableCell>
                    <TableCell>Expected persona</TableCell>
                    <TableCell>Observed persona</TableCell>
                    <TableCell>Impact on graph</TableCell>
                    <TableCell align="right">Σ Δ log&nbsp;p</TableCell>
                    <TableCell align="right">Confidence</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {engagementImpactRows.map((row, idx) => (
                    <TableRow key={`${row.engagementLabel}-${idx}`}>
                      <TableCell sx={{ maxWidth: 280 }}>
                        <Typography variant="body2">{row.engagementLabel}</Typography>
                        <Typography
                          variant="caption"
                          color="text.secondary"
                          sx={{ display: "block" }}
                        >
                          {fmtWhen(row.timestamp)}
                        </Typography>
                      </TableCell>
                      <TableCell>{row.channel}</TableCell>
                      <TableCell>{row.expected}</TableCell>
                      <TableCell>{row.observed}</TableCell>
                      <TableCell sx={{ maxWidth: 360 }}>
                        <Typography
                          variant="caption"
                          color="text.secondary"
                          sx={{ display: "block" }}
                        >
                          {row.impactSummary}
                        </Typography>
                      </TableCell>
                      <TableCell align="right">
                        {row.totalDelta === 0 ? "—" : row.totalDelta.toFixed(3)}
                      </TableCell>
                      <TableCell align="right">
                        {fmtPercent(row.confidence, {
                          inputIsFraction: true,
                          decimals: 0,
                        })}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          )}
        </Collapse>
      </CardContent>
    </Card>
  );
}

// -----------------------------------------
// Component
// -----------------------------------------
export default function EngagementsSetup() {
  const token = localStorage.getItem("token");
  const [companyId, setCompanyId] = useState<string | null>(null);
  const [selectedProductId, setSelectedProductId] = useState<string | null>(
    null
  );

  const [targets, setTargets] = useState<TargetAccount[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [statusMsg, setStatusMsg] = useState<string>("");

  const [sections, setSections] = useState<SectionForm[]>([newSection()]);

  const [beliefByAccount, setBeliefByAccount] = useState<
    Record<string, AccountBeliefThesis>
  >({});
  const [loadingMatchesFor, setLoadingMatchesFor] = useState<string | null>(
    null
  );

  const [arsenalMetaGaps, setArsenalMetaGaps] = useState<{
    assets: string[];
    channels: string[];
  }>({ assets: [], channels: [] });
  const [showLatentStory, setShowLatentStory] = useState(true);
  const [storyTabByAccount, setStoryTabByAccount] = useState<StoryTabState>({});
  const [storyEdits, setStoryEdits] = useState<StoryEditState>({});
  const [, setGlobalInsights] = useState<PersonaMatchesResponse["global"] | null>(null);

  const [snack, setSnack] = useState<{
    open: boolean;
    msg: string;
    sev: "success" | "error" | "info" | "warning";
  }>({ open: false, msg: "", sev: "success" });

  // ---------- init ----------
  useEffect(() => {
    (async () => {
      try {
        const meRes = await fetch("http://localhost:8000/me", {
          headers: { Authorization: `Bearer ${token}` },
        });
        if (!meRes.ok) throw new Error("Auth failed");
        const me = await meRes.json();
        setCompanyId(me.company_id);

        const prodRes = await fetch(
          `http://localhost:8000/get-products/${me.company_id}`,
          {
            headers: { Authorization: `Bearer ${token}` },
          }
        );
        const prodData = await prodRes.json();
        if (!prodData.products || prodData.products.length === 0) {
          setStatusMsg("No products found. Please run Value Prop first.");
          setLoading(false);
          return;
        }
        const picked = (prodData.products[0]?.id as string) || null;
        setSelectedProductId(picked);

        const tRes = await fetch(
          `http://localhost:8000/get-target-accounts/${me.company_id}?product_id=${picked}`,
          { headers: { Authorization: `Bearer ${token}` } }
        );
        if (!tRes.ok) throw new Error("Failed to fetch target accounts");
        const tData = await tRes.json();
        const rows: TargetAccount[] = (tData.accounts || []).map((row: any) => ({
          id: row.id,
          account_name: row.account_name,
          deal_status: row.deal_status || "New",
        }));
        setTargets(rows);

        setSections((prev) => {
          const next = [...prev];
          next[0] = { ...next[0], account_id: rows[0]?.id || "" };
          return next;
        });
      } catch (e: any) {
        setStatusMsg(e?.message || "Initialization failed");
      } finally {
        setLoading(false);
      }
    })();
  }, [token]);

  const accountsById = useMemo(() => {
    const m = new Map<string, TargetAccount>();
    targets.forEach((t) => m.set(t.id, t));
    return m;
  }, [targets]);

  const freeAccounts = useMemo(() => {
    const used = new Set(sections.map((s) => s.account_id).filter(Boolean));
    return targets.filter((t) => !used.has(t.id));
  }, [targets, sections]);

  useEffect(() => {
    setArsenalMetaGaps({ assets: [], channels: [] });
  }, [selectedProductId]);

  // ---------- loaders ----------
  async function fetchEngagementsForAccount(
    productId: string,
    accountId: string,
    token: string
  ) {
    const res = await fetch(
      `http://localhost:8000/load-account-engagements/${productId}/${accountId}`,
      {
        headers: { Authorization: `Bearer ${token}` },
      }
    );
    if (!res.ok) throw new Error("Failed to load engagements");
    const data = await res.json();
    const loaded: Engagement[] = (data.engagements || [])
      .map((r: any) => ({
        account_id: r.account_id,
        timestamp: r.timestamp,
        actor: {
          name: r.actor?.name,
          role: r.actor?.role ?? r.actor?.title ?? "",
          title: r.actor?.title ?? r.actor?.role ?? "",
          department: r.actor?.department ?? "",
          seniority: r.actor?.seniority ?? "",
          confidence: r.actor?.confidence,
        },
        channel: r.channel || undefined,
        source: r.source,
        raw_activity: r.raw_activity,
        asset_id: r.asset_id || undefined,
        inferred: !!r.inferred,
        __persisted__: true,
      }))
      .sort(sortAscByTimestamp);
    return loaded;
  }

  async function fetchPersonaMatches(
    productId: string,
    accountId: string,
    token: string
  ) {
    setLoadingMatchesFor(accountId);
    try {
      const res = await fetch(
        `http://localhost:8000/get-persona-matches/${productId}/${accountId}`,
        {
          headers: { Authorization: `Bearer ${token}` },
        }
      );
      if (!res.ok) {
        throw new Error("Failed to load persona matches / belief thesis");
      }
      const data = await res.json();
      console.log("Fetched persona matches / belief thesis:", data);

      if (data && Object.prototype.hasOwnProperty.call(data, "global")) {
        const insights =
          (data.global as PersonaMatchesResponse["global"]) ?? null;
        if (insights) {
          insights.persona_recommendations.forEach((rec) => {
            registerPersonaLabel(rec.persona_id, rec.persona_label || null);
          });
          setGlobalInsights(insights);
        }
      } else {
        setGlobalInsights(null);
      }

      const payload = data?.thesis ?? data ?? {};
      const storylineRaw: StorylinePayload | null =
        (payload?.storyline as StorylinePayload | null) ??
        (data?.storyline as StorylinePayload | null) ??
        null;
      const storyline = decorateStoryline(storylineRaw);

      // Map "persons" → PersonaMatchRow[]
      const personsSource = Array.isArray(payload?.persons)
        ? payload.persons
        : Array.isArray(data?.persons)
        ? data.persons
        : [];
      const persons: PersonaMatchRow[] = personsSource.map((p: any) => {
        const bestId =
          p.best ?? p.graph_persona_node_id ?? p.canonical_persona_best ?? null;
        const graphLabel =
          p.best_label ?? p.graph_persona_label ?? undefined;
        const canonicalLabel = p.canonical_persona_label ?? undefined;
        if (bestId && graphLabel) {
          registerPersonaLabel(bestId, graphLabel);
        }
        if (p.canonical_persona_best && canonicalLabel) {
          registerPersonaLabel(p.canonical_persona_best, canonicalLabel);
        }
        return {
          name: p.name,
          title: p?.canonical_meta?.title || p.title,
          department: p?.canonical_meta?.department || p.department,
          seniority: p?.canonical_meta?.seniority || p.seniority,

          graph_persona_node_id: bestId,
          graph_persona_score: typeof p.score === "number" ? p.score : null,
          graph_persona_label:
            p.best_label ?? p.graph_persona_label ?? null,

          canonical_persona_best: p.canonical_persona_best ?? null,
          canonical_persona_score:
            typeof p.canonical_persona_score === "number"
              ? p.canonical_persona_score
              : null,
          canonical_persona_label: p.canonical_persona_label ?? null,
          canonical_meta: p.canonical_meta,
          alternates: p.alternates || [],
          jobs: (p.jobs || []).map((j: any) => ({
            job_text: j.job_text,
            canonical_job_id: j.canonical_job_id ?? null,
            relevance: j.relevance,
            likelihood: j.likelihood,
            score: j.score,
            evidence: j.evidence,
          })),
        };
      });

      // helper
      const mapCandidatePath = (w: any): CandidatePath => ({
        personas: Array.isArray(w?.personas)
          ? w.personas
          : Array.isArray(w?.path)
          ? w.path
          : [],
        belief_states: w?.belief_states || [],
        score: Number(w?.score ?? 0),
        probability: Number(w?.probability ?? 0),
        rationale: w?.rationale || "",
      });

      // Map "walk_paths" → CandidatePath[]
      const pathsSource = Array.isArray(payload?.walk_paths)
        ? payload.walk_paths
        : Array.isArray(data?.walk_paths)
        ? data.walk_paths
        : [];
      const paths: CandidatePath[] = pathsSource.map(mapCandidatePath);

      const stepsSource = Array.isArray(payload?.journey?.steps)
        ? payload.journey.steps
        : Array.isArray(data?.journey?.steps)
        ? data.journey.steps
        : [];
      const steps: JourneyStep[] = stepsSource.map((s: any) => ({
        t: s.t,
        state_personas: s.state_personas || [],
        predicted_topK: s.predicted_topK || [],
        observed_next: s.observed_next || null,
        bucket: s.bucket,
        hit_at_1: !!s.hit_at_1,
        hit_at_3: !!s.hit_at_3,
        paths: Array.isArray(s.paths) ? s.paths.map(mapCandidatePath) : [],
        expected_next: Array.isArray(s.expected_next) ? s.expected_next : [],
        win_likelihood:
          typeof s.win_likelihood === "number" ? s.win_likelihood : undefined,
      }));

      const sanitizeLatentEdge = (edge: any): LatentEdgeEvidence | null => {
        if (!edge || typeof edge !== "object") return null;
        const fromRaw = edge.from ?? edge.u ?? edge.source ?? null;
        const toRaw = edge.to ?? edge.v ?? edge.target ?? null;
        const supports =
          typeof edge.supports === "boolean"
            ? edge.supports
            : typeof edge.support === "boolean"
            ? edge.support
            : undefined;
        const deltaRaw =
          typeof edge.delta_log_prob === "number"
            ? edge.delta_log_prob
            : typeof edge.deltaLogProb === "number"
            ? edge.deltaLogProb
            : typeof edge.delta_log_prob === "string"
            ? Number(edge.delta_log_prob)
            : undefined;
        const pathProbRaw =
          typeof edge.path_probability === "number"
            ? edge.path_probability
            : typeof edge.pathProbability === "number"
            ? edge.pathProbability
            : typeof edge.path_probability === "string"
            ? Number(edge.path_probability)
            : undefined;

        return {
          from: fromRaw != null ? String(fromRaw) : undefined,
          to: toRaw != null ? String(toRaw) : undefined,
          supports,
          delta_log_prob:
            typeof deltaRaw === "number" && Number.isFinite(deltaRaw)
              ? deltaRaw
              : undefined,
          path_probability:
            typeof pathProbRaw === "number" && Number.isFinite(pathProbRaw)
              ? pathProbRaw
              : undefined,
          rel: typeof edge.rel === "string" ? edge.rel : undefined,
        };
      };

      const normalizeLatent = (row: any, idx: number): LatentActivityItem => {
        const insertIdxRaw =
          row?.insert_before_step_index ?? row?.step_index ?? row?.step ?? idx;
        const insertIdx =
          typeof insertIdxRaw === "number"
            ? insertIdxRaw
            : Number.isFinite(Number(insertIdxRaw))
            ? Number(insertIdxRaw)
            : idx;
        const stepTRaw = row?.step_t;
        const step_t =
          typeof stepTRaw === "number"
            ? stepTRaw
            : Number.isFinite(Number(stepTRaw))
            ? Number(stepTRaw)
            : null;
        const personaVal = row?.persona_id ?? row?.personaId ?? null;
        const targetVal = row?.target_persona_id ?? row?.targetPersonaId ?? null;
        const personaId = personaVal != null ? String(personaVal) : null;
        const targetId = targetVal != null ? String(targetVal) : null;
        const confidenceRaw = row?.confidence;
        const pathProbRaw = row?.path_probability ?? row?.pathProbability;
        const narrative =
          typeof row?.narrative === "string" ? row.narrative : null;
        const reason = typeof row?.reason === "string" ? row.reason : null;
        const pathExcerpt = Array.isArray(row?.path_excerpt)
          ? row.path_excerpt.map((x: any) => String(x))
          : [];

        return {
          id:
            typeof row?.id === "string" && row.id.trim()
              ? row.id
              : `latent-${idx}-${insertIdx}`,
          kind: "latent",
          insert_before_step_index: insertIdx,
          step_t,
          persona_id: personaId,
          persona_label:
            typeof row?.persona_label === "string"
              ? row.persona_label
              : personaId
              ? personaLabelFromId(personaId)
              : null,
          target_persona_id: targetId,
          target_persona_label:
            typeof row?.target_persona_label === "string"
              ? row.target_persona_label
              : targetId
              ? personaLabelFromId(targetId)
              : null,
          confidence:
            typeof confidenceRaw === "number"
              ? confidenceRaw
              : typeof confidenceRaw === "string"
              ? Number(confidenceRaw)
              : null,
          path_probability:
            typeof pathProbRaw === "number"
              ? pathProbRaw
              : typeof pathProbRaw === "string"
              ? Number(pathProbRaw)
              : null,
          narrative,
          reason,
          edge_evidence: sanitizeLatentEdge(row?.edge_evidence),
          path_excerpt: pathExcerpt,
        };
      };

      const normalizeStoryItem = (row: any, idx: number): ActivityStoryItem => {
        if (row && row.kind === "latent") {
          return normalizeLatent(row, idx);
        }
        const stepIdxRaw =
          row?.step_index ?? row?.stepIndex ?? row?.insert_before_step_index ?? idx;
        const stepIdx =
          typeof stepIdxRaw === "number"
            ? stepIdxRaw
            : Number.isFinite(Number(stepIdxRaw))
            ? Number(stepIdxRaw)
            : idx;
        const stepTRaw = row?.step_t;
        const step_t =
          typeof stepTRaw === "number"
            ? stepTRaw
            : Number.isFinite(Number(stepTRaw))
            ? Number(stepTRaw)
            : null;
        const personaVal = row?.persona_id ?? row?.personaId ?? null;
        const personaId = personaVal != null ? String(personaVal) : null;

        return {
          id:
            typeof row?.id === "string" && row.id.trim()
              ? row.id
              : `observed-${idx}-${stepIdx}`,
          kind: "observed",
          step_index: stepIdx,
          step_t,
          timestamp:
            typeof row?.timestamp === "string" ? row.timestamp : null,
          persona_id: personaId,
          persona_label:
            typeof row?.persona_label === "string"
              ? row.persona_label
              : personaId
              ? personaLabelFromId(personaId)
              : null,
          bucket: typeof row?.bucket === "string" ? row.bucket : null,
          engagement_meta: row?.engagement_meta ?? null,
        };
      };

      const latentActivitySource = Array.isArray(data?.latent_activity)
        ? data.latent_activity
        : Array.isArray(payload?.latent_activity)
        ? payload.latent_activity
        : [];
      const latentActivity: LatentActivityItem[] = latentActivitySource.map(
        (row: any, idx: number) => normalizeLatent(row, idx)
      );

      latentActivity.forEach((row) => {
        if (row.persona_id && row.persona_label) {
          registerPersonaLabel(row.persona_id, row.persona_label);
        }
        if (row.target_persona_id && row.target_persona_label) {
          registerPersonaLabel(row.target_persona_id, row.target_persona_label);
        }
      });

      const activityStorySource = Array.isArray(data?.activity_story)
        ? data.activity_story
        : Array.isArray(payload?.activity_story)
        ? payload.activity_story
        : [];
      const activityStory: ActivityStoryItem[] = activityStorySource.map(
        (row: any, idx: number) => normalizeStoryItem(row, idx)
      );

        // ---- learning_summary + learning_neighborhoods (with fallback) ----
      let learning_summary: LearningSummary | undefined;
      const lsRaw: any =
        payload?.learning_summary ??
        data?.learning_summary ??
        payload?.human_readable_learning;

      const normalizeTheses = (
        arr: any[] | undefined,
        isGraphLevel: boolean
      ): PersonaPathThesis[] => {
        if (!Array.isArray(arr)) return [];
        return arr.map((p: any): PersonaPathThesis => {
          const anchorPersonaId =
            p.anchor_persona_id || p.persona_id || p.anchor_id || null;

          let anchorLabel =
            p.anchor_label ||
            p.persona_label ||
            p.label ||
            null;

          // for graph-level items we often have pair_labels instead
          if (!anchorLabel && isGraphLevel && Array.isArray(p.pair_labels)) {
            anchorLabel = p.pair_labels.join(" → ");
          }

          const diagnosis =
            p.diagnosis ||
            p.thesis ||
            p.text ||
            (typeof p === "string" ? p : "");

          const rationale =
            p.rationale ||
            p.explanation ||
            p.reason ||
            undefined;

          const band = p.band as LearningBand | undefined;

          return {
            anchor_persona_id: anchorPersonaId ?? undefined,
            anchor_label: anchorLabel ?? undefined,
            diagnosis,
            rationale,
            band,
          };
        });
      };

      const isObject = (val: any) =>
        val !== null && typeof val === "object" && !Array.isArray(val);

      if (isObject(lsRaw)) {
        const effectiveLs = payload?.learning_summary ?? lsRaw;
        const topEdges: EdgeUpdate[] = [
          ...(effectiveLs?.top_edge_updates || []),
          ...(effectiveLs?.recommendations?.edge_updates_ranked || []),
        ];
        const nodeUpdates: NodeUpdate[] = [
          ...(effectiveLs?.node_updates_ranked || []),
          ...(effectiveLs?.recommendations?.node_updates_ranked || []),
        ];

        // allow both old & new field names
        const persona_graph_inferences = normalizeTheses(
          effectiveLs.persona_graph_inferences ||
            effectiveLs.persona_path_inferences,
          false
        );
        const full_graph_inferences = normalizeTheses(
          effectiveLs.full_graph_inferences || effectiveLs.graph_level_inferences,
          true
        );

        learning_summary = {
          ...effectiveLs,
          persona_graph_inferences,
          full_graph_inferences,
          top_edge_updates: topEdges,
          node_updates_ranked: nodeUpdates,
          recommendations: effectiveLs.recommendations,
        };
      } else if (data?.incremental) {
        // fallback if only incremental_learning exists
        const inc: any = data.incremental || data.incremental_learning;

        const persona_graph_inferences = normalizeTheses(
          inc.persona_path_inferences,
          false
        );
        const full_graph_inferences = normalizeTheses(
          inc.graph_level_inferences,
          true
        );

        learning_summary = {
          persona_graph_inferences,
          full_graph_inferences,
          top_edge_updates: [],
          node_updates_ranked: [],
        };
      } else if (isObject(payload?.human_readable_learning)) {
        // older human-readable shape (strings or objects)
        const hr: any = payload.human_readable_learning;

        const persona_graph_inferences = normalizeTheses(
          hr.persona_graph_inferences,
          false
        );
        const full_graph_inferences = normalizeTheses(
          hr.full_graph_inferences,
          true
        );

        const top_edge_updates: EdgeUpdate[] = (hr.top_edge_updates || []).map(
          (x: any) =>
            typeof x === "string" ? parseEdgeUpdateFromString(x) : (x as EdgeUpdate)
        );

        learning_summary = {
          persona_graph_inferences,
          full_graph_inferences,
          top_edge_updates,
        };
      } else if (Array.isArray(payload?.human_readable_learning)) {
        // Fallback when we only have human-readable strings
        const hrList: any[] = payload.human_readable_learning;
        const persona_graph_inferences = normalizeTheses(hrList, false);
        learning_summary = {
          persona_graph_inferences,
          full_graph_inferences: [],
          top_edge_updates: hrList
            .map((x: any) =>
              typeof x === "string" ? parseEdgeUpdateFromString(x) : null
            )
            .filter(Boolean) as EdgeUpdate[],
        };
      }

      const learning_neighborhoods: LearningNeighborhoods | undefined =
        payload?.learning_neighborhoods ||
        data?.learning_neighborhoods ||
        (data?.incremental?.neighborhoods
          ? {
              upstream: Object.fromEntries(
                (data.incremental.neighborhoods.upstream || []).map(
                  (entry: any, idx: number) => [entry?.anchor || `up-${idx}`, entry]
                )
              ),
              handoff: Object.fromEntries(
                (data.incremental.neighborhoods.handoff || []).map(
                  (entry: any, idx: number) => [entry?.anchor || `handoff-${idx}`, entry]
                )
              ),
              downstream: Object.fromEntries(
                (data.incremental.neighborhoods.downstream || []).map(
                  (entry: any, idx: number) => [entry?.anchor || `down-${idx}`, entry]
                )
              ),
            }
          : undefined);



      const thesis: AccountBeliefThesis = {
        persons,
        paths,
        learning_summary,
        learning_neighborhoods,
        baseline_paths: Array.isArray(payload?.baseline_paths)
          ? payload.baseline_paths.map(mapCandidatePath)
          : [],
        baseline_expected_next:
          payload?.baseline_expected_next ?? data?.baseline_expected_next ?? [],
        current_paths: Array.isArray(payload?.current_paths)
          ? payload.current_paths.map(mapCandidatePath)
          : [],
        current_expected_next:
          payload?.current_expected_next ?? data?.current_expected_next ?? [],
        overall_fit: payload?.fit
          ? {
              best_path_probability: Number(
                payload.fit.best_path_probability ?? 0
              ),
              accuracy: Number(payload.fit.hit_at_1 === true ? 1 : 0),
              surprisal_index: Number(payload.fit.surprisal_index ?? 0),
              variance: Number(payload.fit.variance ?? 0),
            }
          : null,
        journey_steps: steps,
        latent_activity: latentActivity,
        activity_story: activityStory,
        storyline,
      };

      setBeliefByAccount((prev) => ({ ...prev, [accountId]: thesis }));

      // Refresh global rollups so the recommendations card stays in sync with the latest run.
    } catch (err: any) {
      setSnack({
        open: true,
        msg: err?.message || "Failed to load persona matches / belief thesis",
        sev: "warning",
      });
      setBeliefByAccount((prev) => ({
        ...prev,
        [accountId]: {
          persons: [],
          paths: [],
          overall_fit: null,
          journey_steps: [],
          latent_activity: [],
          activity_story: [],
          storyline: null,
        } as AccountBeliefThesis,
      }));
    } finally {
      setLoadingMatchesFor(null);
    }
  }

  // preload first section
  useEffect(() => {
    (async () => {
      const first = sections[0];
      if (!first) return;
      if (!first.account_id || !selectedProductId || !token) return;

      if (first.added.length === 0) {
        try {
          const existing = await fetchEngagementsForAccount(
            selectedProductId,
            first.account_id,
            token
          );
          const hsText = formatAsHsJson(existing);
          setSections((prev) => {
            const next = [...prev];
            next[0] = { ...next[0], added: existing, hsText, hsTextErr: "" };
            return next;
          });
        } catch {
          // ignore
        }
      }
      await fetchPersonaMatches(selectedProductId, first.account_id, token);
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sections[0]?.account_id, selectedProductId, token]);

  // ---------- section helpers ----------
  function updateSection(idx: number, patch: Partial<SectionForm>) {
    setSections((prev) => prev.map((s, i) => (i === idx ? { ...s, ...patch } : s)));
  }

  function validateSection(idx: number): string | null {
    const s = sections[idx];
    if (!s.account_id) return "Please select a target account.";
    if (!s.actorRole.trim()) return "Please enter the actor's role/title.";
    if (!s.actorDept.trim()) return "Please enter the actor's department.";
    if (!s.rawActivity.trim()) return "Please enter a raw activity.";
    if (!s.timestampLocal) return "Please pick or type a timestamp.";
    return null;
  }

  function handleTimestampTextChange(idx: number, val: string) {
    const parsed = parseUserDatetimeText(val);
    if (parsed) {
      updateSection(idx, { timestampText: val, timestampLocal: parsed, timestampErr: "" });
    } else {
      updateSection(idx, {
        timestampText: val,
        timestampErr: val.trim() ? "Could not parse. Try YYYY-MM-DD HH:mm or 'today 5pm'." : "",
      });
    }
  }
  function handleTimestampTextBlur(idx: number) {
    const s = sections[idx];
    if (!s.timestampText) return;
    const parsed = parseUserDatetimeText(s.timestampText);
    if (parsed) {
      updateSection(idx, {
        timestampLocal: parsed,
        timestampText: parsed.replace("T", " "),
        timestampErr: "",
      });
    }
  }
  function handleTimestampPickerChange(idx: number, val: string) {
    updateSection(idx, {
      timestampLocal: val,
      timestampText: val.replace("T", " "),
      timestampErr: "",
    });
  }

  function handleAddEngagement(idx: number) {
    const err = validateSection(idx);
    if (err) {
      setSnack({ open: true, msg: err, sev: "error" });
      return;
    }
    const s = sections[idx];
    const engagement: Engagement = {
      account_id: s.account_id,
      timestamp: toIsoZ(s.timestampLocal),
      actor: {
        name: s.actorName.trim() || undefined,
        title: s.actorRole.trim(),
        role: s.actorRole.trim(),
        department: s.actorDept.trim(),
        seniority: s.actorSeniority.trim() || undefined,
        confidence: s.inferred ? 0.6 : 1.0,
      },
      channel: s.channel.trim() || undefined,
      source: s.source,
      raw_activity: s.rawActivity.trim(),
      asset_id: s.assetId.trim() || undefined,
      inferred: s.inferred,
    };
    const nextAdded = [...s.added, engagement].sort(sortAscByTimestamp);
    updateSection(idx, {
      added: nextAdded,
      hsText: formatAsHsJson(nextAdded),
      rawActivity: "",
      actorName: "",
      actorRole: "",
      actorDept: "",
      actorSeniority: "",
      assetId: "",
      channel: "",
      inferred: false,
      timestampLocal: nowLocalForInput(),
      timestampText: nowLocalForInput().replace("T", " "),
      timestampErr: "",
    });
    setSnack({ open: true, msg: "Engagement added", sev: "success" });
  }

  function handleAddNewAccountSection() {
    if (freeAccounts.length === 0) {
      setSnack({ open: true, msg: "No more target accounts available.", sev: "info" });
      return;
    }
    const nextId = freeAccounts[0].id;
    setSections((prev) => [...prev, newSection(nextId)]);
  }

  const handleStoryTabChange = useCallback(
    (accountKey: string, nextTab: "activity" | "story") => {
      setStoryTabByAccount((prev) => {
        if (prev[accountKey] === nextTab) return prev;
        return { ...prev, [accountKey]: nextTab };
      });
    },
    []
  );

  const handleStoryEditChange = useCallback(
    (accountKey: string, nodeId: string, value: string) => {
      setStoryEdits((prev) => ({
        ...prev,
        [accountKey]: {
          ...(prev[accountKey] || {}),
          [nodeId]: value,
        },
      }));
    },
    []
  );

  async function handleSaveAndAnalyze() {
    try {
      if (!companyId || !selectedProductId) throw new Error("Missing company/product context");
      const allEngagements: Engagement[] = sections.flatMap((s) => s.added);
      const onlyNew = allEngagements.filter((e) => !e.__persisted__);
      if (onlyNew.length === 0) {
        setSnack({ open: true, msg: "No new engagements to save.", sev: "info" });
        return;
      }
      const body = { engagements: onlyNew };
      const res = await fetch(
        `http://localhost:8000/save-account-engagements/${selectedProductId}`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${token}`,
          },
          body: JSON.stringify(body),
        }
      );
      const saveSummary = await res.json();
      if (!res.ok) {
        const detail =
          (saveSummary && typeof saveSummary.detail === "string"
            ? saveSummary.detail
            : null) || "Save failed";
        throw new Error(detail);
      }

      const assetGapList = Array.isArray(saveSummary?.arsenal_assets_missing_meta)
        ? saveSummary.arsenal_assets_missing_meta
        : [];
      const channelGapList = Array.isArray(saveSummary?.arsenal_channels_missing_meta)
        ? saveSummary.arsenal_channels_missing_meta
        : [];
      setArsenalMetaGaps({
        assets: assetGapList,
        channels: channelGapList,
      });

      const newlySavedKeys = new Set(onlyNew.map(engagementKey));
      setSections((prev) =>
        prev.map((sec) => {
          const updatedAdded = sec.added.map((e) =>
            newlySavedKeys.has(engagementKey(e)) ? { ...e, __persisted__: true } : e
          );
          return {
            ...sec,
            added: updatedAdded,
            hsText: formatAsHsJson(updatedAdded),
          };
        })
      );

      const accountsToRefresh = Array.from(
        new Set(sections.map((s) => s.account_id).filter(Boolean))
      );
      await Promise.all(
        accountsToRefresh.map((aid) =>
          fetchPersonaMatches(selectedProductId!, aid, token!)
        )
      );
      const gapNoticeParts: string[] = [];
      if (assetGapList.length) {
        gapNoticeParts.push(
          `${assetGapList.length} asset${assetGapList.length === 1 ? "" : "s"} need metadata`
        );
      }
      if (channelGapList.length) {
        gapNoticeParts.push(
          `${channelGapList.length} channel${channelGapList.length === 1 ? "" : "s"} need metadata`
        );
      }
      const gapSummary = gapNoticeParts.length ? ` — ${gapNoticeParts.join(" • ")}` : "";
      setSnack({
        open: true,
        msg: `Saved & analyzing engagements…${gapSummary}`,
        sev: "success",
      });
    } catch (e: any) {
      setSnack({
        open: true,
        msg: e?.message || "Failed to save engagements.",
        sev: "error",
      });
    }
  }

  // ---------- HS tab helpers ----------
  function formatAsHsJson(list: Engagement[]) {
    const payload = {
      engagements: list.map((e) => ({
        account_id: e.account_id,
        timestamp: e.timestamp,
        actor: {
          name: e.actor?.name || null,
          title: e.actor?.title ?? e.actor?.role ?? "",
          department: e.actor?.department ?? "",
          seniority: e.actor?.seniority || null,
          confidence: e.actor?.confidence ?? 1.0,
        },
        channel: e.channel || null,
        source: e.source,
        raw_activity: e.raw_activity,
        asset_id: e.asset_id || null,
        inferred: !!e.inferred,
      })),
    };
    return JSON.stringify(payload, null, 2);
  }

  function validateHsJson(text: string, currentAccountId: string): string | null {
    let obj: any;
    try {
      obj = JSON.parse(text);
    } catch {
      return "Invalid JSON.";
    }
    if (!obj || typeof obj !== "object") return "Payload must be a JSON object.";
    if (!Array.isArray(obj.engagements)) return "Missing 'engagements' array.";
    for (let i = 0; i < obj.engagements.length; i++) {
      const e = obj.engagements[i];
      if (!e) return `engagements[${i}] missing.`;
      if (!e.timestamp) return `engagements[${i}].timestamp is required.`;
      if (!e.actor || typeof e.actor !== "object") return `engagements[${i}].actor is required.`;
      if (!e.actor.title && !e.actor.role) return `engagements[${i}].actor.title (or role) is required.`;
      if (!e.actor.department) return `engagements[${i}].actor.department is required.`;
      if (!e.source) return `engagements[${i}].source is required.`;
      if (!e.raw_activity) return `engagements[${i}].raw_activity is required.`;
      if (!e.account_id) return `engagements[${i}].account_id is required.`;
      if (currentAccountId && e.account_id !== currentAccountId) {
        return `engagements[${i}].account_id must be '${currentAccountId}' for this tab.`;
      }
    }
    return null;
  }

  async function submitHsImport(idx: number) {
    const s = sections[idx];
    if (!selectedProductId || !s.account_id) {
      setSnack({
        open: true,
        msg: "Select product & account first.",
        sev: "warning",
      });
      return;
    }
    const err = validateHsJson(s.hsText, s.account_id);
    if (err) {
      updateSection(idx, { hsTextErr: err });
      setSnack({ open: true, msg: err, sev: "error" });
      return;
    }
    updateSection(idx, { hsTextErr: "" });
    try {
      const payload = JSON.parse(s.hsText);
      const res = await fetch(
        `http://localhost:8000/engagements/bulk/${selectedProductId}/${s.account_id}`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${token}`,
          },
          body: JSON.stringify(payload),
        }
      );
      if (!res.ok) throw new Error("Bulk import failed.");
      const reloaded = await fetchEngagementsForAccount(
        selectedProductId,
        s.account_id,
        token!
      );
      updateSection(idx, {
        added: reloaded,
        hsText: formatAsHsJson(reloaded),
      });
      await fetchPersonaMatches(selectedProductId, s.account_id, token!);
      setSnack({
        open: true,
        msg: "Imported from HubSpot payload (overwrite complete).",
        sev: "success",
      });
    } catch (e: any) {
      setSnack({
        open: true,
        msg: e?.message || "Bulk import failed.",
        sev: "error",
      });
    }
  }

  // ---------- Feed item ----------
  function TimelineRail({
    color,
    dashed = false,
  }: {
    color: string;
    dashed?: boolean;
  }) {
    return (
      <Box
        sx={{
          width: 16,
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
        }}
      >
        <Box
          sx={{
            width: 8,
            height: 8,
            borderRadius: "50%",
            bgcolor: dashed ? "background.paper" : color,
            border: dashed ? "2px solid" : "none",
            borderColor: color,
            mt: 0.75,
          }}
        />
        <Box
          sx={{
            flex: 1,
            width: dashed ? 0 : 2,
            bgcolor: dashed ? "transparent" : color,
            borderLeft: dashed ? `2px dashed ${color}` : "none",
            mt: 0.5,
          }}
        />
      </Box>
    );
  }

  function FeedItem({
    e,
    match,
    belief,
  }: {
    e: Engagement;
    match?: PersonaMatchRow;
    belief?: EngagementBeliefAnnotation;
  }) {
    const enteredTitle = e.actor?.title ?? e.actor?.role ?? "";
    const enteredDept = e.actor?.department ?? "";
    const enteredSnr = e.actor?.seniority ?? "";

    const nodeId =
      match?.graph_persona_node_id ?? match?.canonical_persona_best ?? null;
    const score =
      typeof match?.graph_persona_score === "number"
        ? match.graph_persona_score
        : typeof match?.canonical_persona_score === "number"
        ? match.canonical_persona_score!
        : null;

    const topJobs = (match?.jobs || [])
      .slice()
      .sort(
        (a, b) =>
          (b.score ?? 0) - (a.score ?? 0) ||
          (b.relevance ?? 0) - (a.relevance ?? 0)
      )
      .slice(0, 3);

    const bestLabel =
      match?.graph_persona_label ||
      match?.canonical_persona_label ||
      (nodeId ? personaLabelFromId(nodeId) : "");

    const personaChip = personaChipLabel(e, bestLabel);
    const pathMeta = describePathConfidence(belief?.quality);
    const surpriseMeta = describeSurpriseLevel(belief?.surpriseScore);
    const verdict = belief
      ? verdictFromClassification(belief.classification, belief.note)
      : null;
    const impactSummary = belief
      ? summarizeImpact(belief, e, match)
      : { badge: null, totalDelta: 0, bullets: [], tooltipLines: [] };
    const latentHints = belief ? latentHintsFromIndicates(belief.indicates) : [];
    const guessedPrev = belief?.expectedBefore
      ? personaLabelFromId(belief.expectedBefore)
      : "Model unsure";
    const guessedNext = belief?.nextExpectedAfter
      ? personaLabelFromId(belief.nextExpectedAfter)
      : "Model unsure";

    return (
      <Box sx={{ display: "flex", gap: 2 }}>
        <TimelineRail color="primary.main" />
        <Box sx={{ flex: 1, pb: 2 }}>
          {/* header row: activity + time */}
          <Box
            sx={{
              display: "flex",
              alignItems: "center",
              gap: 1,
              flexWrap: "wrap",
            }}
          >
            <Typography variant="subtitle1" fontWeight={600}>
              {e.raw_activity || "Activity"}
            </Typography>
            <Chip
              size="small"
              label={e.inferred ? "Inferred" : "Actual"}
              variant={e.inferred ? "filled" : "outlined"}
            />
            <Typography variant="caption" color="text.secondary">
              • {fmtWhen(e.timestamp)}
            </Typography>
          </Box>

          {/* Entered persona */}
          {(enteredTitle || enteredDept || enteredSnr || e.actor?.name) && (
            <Typography variant="body2" sx={{ mt: 0.75 }}>
              <strong>{e.actor?.name || "Unknown"}</strong>
              {enteredTitle ? ` — ${enteredTitle}` : ""}
              {enteredDept ? ` · ${enteredDept}` : ""}
              {enteredSnr ? ` · ${enteredSnr}` : ""}
            </Typography>
          )}

          {/* Best-match persona + jobs (includes persona id chip) */}
          {(bestLabel || topJobs.length > 0) && (
            <Box
              sx={{
                mt: 1,
                p: 1,
                borderRadius: 1,
                bgcolor: (theme) => theme.palette.action.hover,
                display: "flex",
                flexWrap: "wrap",
                gap: 1,
                alignItems: "center",
              }}
            >
              {bestLabel && (
                <>
                  <Chip
                    size="small"
                    color="default"
                    variant="outlined"
                    label={`Best: ${bestLabel}${
                      typeof score === "number" ? ` (${fmtScore(score)})` : ""
                    }`}
                  />
                  {nodeId && (
                    <Chip
                      size="small"
                      variant="outlined"
                      label={`id: ${nodeId}`}
                      sx={{
                        fontFamily:
                          "ui-monospace, SFMono-Regular, Menlo, monospace",
                      }}
                    />
                  )}
                </>
              )}
              {topJobs.length > 0 && (
                <>
                  <Typography
                    variant="caption"
                    color="text.secondary"
                    sx={{ ml: bestLabel ? 0.5 : 0 }}
                  >
                    Jobs:
                  </Typography>
                  {topJobs.map((j, i) => (
                    <Chip key={i} size="small" label={j.job_text} sx={{ maxWidth: 260 }} />
                  ))}
                </>
              )}
            </Box>
          )}

          {/* Engagement match (graph delta neighborhood) */}
          {belief?.indicates && (
            <Box sx={{ mt: 1 }}>
              <Typography
                variant="caption"
                color="text.secondary"
                sx={{ display: "block", mb: 0.25 }}
              >
                Engagement match — best-guess neighborhood of graph update:
              </Typography>
              {(["upstream", "handoff", "downstream"] as const).map((band) => {
                const arr = belief.indicates?.[band];
                if (!arr || !arr.length) return null;
                return (
                  <Typography key={band} variant="body2" sx={{ display: "block" }}>
                    <strong>{band}:</strong>{" "}
                    {arr.slice(0, 2).map((x) => x.edge).join("; ")}
                  </Typography>
                );
              })}
            </Box>
          )}

          {/* Meta chips */}
          <Box
            sx={{
              mt: 0.75,
              display: "flex",
              gap: 1,
              flexWrap: "wrap",
            }}
          >
            {e.source && <Chip size="small" label={e.source} />}
            {e.channel && <Chip size="small" label={e.channel} />}
            {e.asset_id && <Chip size="small" label={`asset: ${e.asset_id}`} />}
          </Box>

          {personaChip && (
            <Box sx={{ mt: 1 }}>
              <Chip size="small" label={personaChip} />
            </Box>
          )}

          {verdict && (
            <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
              {verdict}
            </Typography>
          )}

          {impactSummary.badge && (
            <Chip
              sx={{ mt: 1 }}
              color={impactSummary.badge.color}
              label={impactSummary.badge.label}
              size="small"
            />
          )}

          {impactSummary.bullets.length > 0 && (
            <Stack spacing={0.5} sx={{ mt: 1 }}>
              {impactSummary.bullets.map((line, idx) => (
                <Typography variant="body2" key={idx}>
                  • {line}
                </Typography>
              ))}
            </Stack>
          )}

          {latentHints.length > 0 && (
            <Stack spacing={0.5} sx={{ mt: 1 }}>
              {latentHints.map((hint, idx) => (
                <Typography variant="caption" color="text.secondary" key={`hint-${idx}`}>
                  {hint}
                </Typography>
              ))}
            </Stack>
          )}

          <Box
            sx={{
              mt: 1.5,
              display: "flex",
              flexWrap: "wrap",
              gap: 2,
            }}
          >
            <Box>
              <Typography variant="caption" color="text.secondary">
                Guessed previous step
              </Typography>
              <Typography variant="body2">{guessedPrev}</Typography>
            </Box>
            <Box>
              <Typography variant="caption" color="text.secondary">
                Who to engage next
              </Typography>
              <Typography variant="body2">{guessedNext}</Typography>
            </Box>
            <Box>
              <Typography variant="caption" color="text.secondary">
                Path confidence
              </Typography>
              <Tooltip title={pathMeta.helper}>
                <Typography variant="body2">{pathMeta.label}</Typography>
              </Tooltip>
            </Box>
            <Box>
              <Typography variant="caption" color="text.secondary">
                Surprise
              </Typography>
              <Tooltip title={surpriseMeta.helper}>
                <Typography variant="body2">{surpriseMeta.label}</Typography>
              </Tooltip>
            </Box>
          </Box>

          {impactSummary.tooltipLines.length > 0 && (
            <Tooltip
              title={
                <Box sx={{ p: 1 }}>
                  {impactSummary.tooltipLines.map((line, idx) => (
                    <Typography
                      key={`math-${idx}`}
                      variant="caption"
                      color="inherit"
                      sx={{ display: "block" }}
                    >
                      {line}
                    </Typography>
                  ))}
                </Box>
              }
            >
              <Button
                size="small"
                startIcon={<InfoOutlinedIcon fontSize="small" />}
                sx={{ mt: 1 }}
              >
                Show math
              </Button>
            </Tooltip>
          )}
        </Box>
      </Box>
    );
  }

  function LatentTimelineItem({ item }: { item: LatentActivityItem }) {
    const personaLabel =
      item.persona_label ||
      (item.persona_id ? personaLabelFromId(item.persona_id) : "Hidden persona");
    const targetLabel =
      item.target_persona_label ||
      (item.target_persona_id ? personaLabelFromId(item.target_persona_id) : null);
    const confidencePct =
      typeof item.confidence === "number"
        ? Math.round(item.confidence * 100)
        : null;
    const pathPct =
      typeof item.path_probability === "number"
        ? Math.round(item.path_probability * 100)
        : null;
    const pathLabels =
      Array.isArray(item.path_excerpt) && item.path_excerpt.length > 0
        ? item.path_excerpt.map((pid) => personaLabelFromId(pid))
        : [];

    return (
      <Box sx={{ display: "flex", gap: 2 }}>
        <TimelineRail color="grey.400" dashed />
        <Box
          sx={{
            flex: 1,
            pb: 2,
            px: 2,
            py: 1.5,
            borderRadius: 1,
            bgcolor: "action.hover",
            border: "1px dashed",
            borderColor: "divider",
          }}
        >
          <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 0.5, opacity: 0.9 }}>
            <Chip size="small" variant="outlined" label="Inferred" />
            {confidencePct !== null && (
              <Chip size="small" variant="outlined" label={`confidence ${confidencePct}%`} />
            )}
            {pathPct !== null && pathPct > 0 && (
              <Chip size="small" variant="outlined" label={`path ${pathPct}%`} />
            )}
          </Stack>
          <Typography variant="body2" color="text.secondary" sx={{ fontStyle: "italic" }}>
            {item.narrative ||
              `${personaLabel} likely progressed the belief state before the next recorded step.`}
          </Typography>
          <Typography variant="caption" color="text.secondary" sx={{ display: "block", mt: 0.75 }}>
            {personaLabel}
            {targetLabel ? ` → ${targetLabel}` : ""}
          </Typography>
          {item.reason && (
            <Typography variant="caption" color="text.disabled" sx={{ display: "block", mt: 0.5 }}>
              {item.reason}
            </Typography>
          )}
          {pathLabels.length > 0 && (
            <Typography variant="caption" color="text.disabled" sx={{ display: "block", mt: 0.5 }}>
              Path hint: {pathLabels.join(" → ")}
            </Typography>
          )}
        </Box>
      </Box>
    );
  }

  // ---------- render ----------
  if (loading) {
    return (
      <Box
        sx={{
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          p: 6,
        }}
      >
        <CircularProgress />
      </Box>
    );
  }
  if (targets.length === 0) {
    return (
      <Box sx={{ maxWidth: 900, mx: "auto", p: 3 }}>
        <Card variant="outlined">
          <CardHeader
            title={<Typography variant="h6">No Target Accounts Found</Typography>}
          />
          <CardContent>
            <Typography variant="body1" gutterBottom>
              You don't have any target accounts yet. Please upload your accounts in the{" "}
              <strong>Target Accounts</strong> tab first.
            </Typography>
            <Button href="/target-accounts" variant="contained">
              Go to Target Accounts
            </Button>
            {statusMsg && (
              <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
                {statusMsg}
              </Typography>
            )}
          </CardContent>
        </Card>
      </Box>
    );
  }

  

  return (
    <Box sx={{ maxWidth: 1100, mx: "auto", p: { xs: 2, md: 3 } }}>
      <Typography variant="h4" fontWeight={700} gutterBottom>
        Engagements Setup
      </Typography>

      {(arsenalMetaGaps.assets.length > 0 || arsenalMetaGaps.channels.length > 0) && (
        <Alert severity="warning" sx={{ mb: 2 }}>
          {[
            arsenalMetaGaps.assets.length
              ? `${arsenalMetaGaps.assets.length} asset${arsenalMetaGaps.assets.length === 1 ? "" : "s"} need metadata`
              : null,
            arsenalMetaGaps.channels.length
              ? `${arsenalMetaGaps.channels.length} channel${arsenalMetaGaps.channels.length === 1 ? "" : "s"} need metadata`
              : null,
          ]
            .filter(Boolean)
            .join(" • ")}
        </Alert>
      )}

      {/* Global insights moved to Insights Inbox */}
      {/* 2. Add accounts / engagements sections */}
      {sections.map((s, idx) => {
        const account = s.account_id ? accountsById.get(s.account_id) : undefined;
        const thesis: AccountBeliefThesis | undefined =
          s.account_id && beliefByAccount[s.account_id]
            ? beliefByAccount[s.account_id]
            : undefined;
        const matches = thesis?.persons || [];
        const candidatePaths = thesis?.paths || [];
        const isMatchesLoading = loadingMatchesFor === s.account_id;
        const journeySteps = thesis?.journey_steps || [];

        const engagementsChrono = s.added.slice().sort(sortAscByTimestamp);
        const annotatedFeed =
          engagementsChrono.length
            ? annotateEngagementsForAccount(
                engagementsChrono,
                matches,
                candidatePaths,
                journeySteps,
                thesis?.learning_summary?.top_edge_updates
              )
            : [];
        const activityStory = thesis?.activity_story ?? [];
        const latentActivity = thesis?.latent_activity ?? [];

        const enrichLatentRow = (row: LatentActivityItem, fallbackId: string): LatentActivityItem => ({
          ...row,
          id:
            typeof row.id === "string" && row.id.trim()
              ? row.id
              : fallbackId,
          persona_label:
            row.persona_label ||
            (row.persona_id ? personaLabelFromId(row.persona_id) : row.persona_label),
          target_persona_label:
            row.target_persona_label ||
            (row.target_persona_id
              ? personaLabelFromId(row.target_persona_id)
              : row.target_persona_label),
        });

        const buildLatentsByStep = (rowsToIndex: LatentActivityItem[]) => {
          const latentsByStep = new Map<number, LatentActivityItem[]>();
          rowsToIndex.forEach((row, latentIdx) => {
            const insertIdxRaw =
              row.insert_before_step_index !== undefined &&
              row.insert_before_step_index !== null
                ? row.insert_before_step_index
                : (row as any)?.step_index;
            const insertIdx = Number.isFinite(Number(insertIdxRaw))
              ? Math.max(0, Number(insertIdxRaw))
              : 0;
            const enriched = enrichLatentRow(row, `latent-${insertIdx}-${latentIdx}`);
            const bucket = latentsByStep.get(insertIdx) || [];
            bucket.push(enriched);
            latentsByStep.set(insertIdx, bucket);
          });
          return latentsByStep;
        };

        const pushLatentBucketRows = (
          targetRows: TimelineRow[],
          bucket: LatentActivityItem[],
          fallbackIdx: number
        ) => {
          bucket
            .slice()
            .sort((a, b) => {
              const aT = a.step_t ?? fallbackIdx;
              const bT = b.step_t ?? fallbackIdx;
              if (aT !== bT) return aT - bT;
              return (a.confidence ?? 0) > (b.confidence ?? 0) ? -1 : 1;
            })
            .forEach((latent) => {
              const key =
                latent.id && latent.id.trim()
                  ? latent.id
                  : `latent-${fallbackIdx}-${targetRows.length}`;
              targetRows.push({
                key,
                type: "latent",
                latent,
              });
            });
        };

        const storyHasLatentSteps = activityStory.some((item) => item.kind === "latent");
        const fallbackLatentsByStep: Map<number, LatentActivityItem[]> =
          showLatentStory && !storyHasLatentSteps
            ? buildLatentsByStep(latentActivity)
            : new Map<number, LatentActivityItem[]>();

        const timelineRows: TimelineRow[] = (() => {
          if (activityStory.length > 0) {
            const rows: TimelineRow[] = [];
            activityStory.forEach((item, storyIdx) => {
              if (item.kind === "latent") {
                if (!showLatentStory) return;
                const latentRow = enrichLatentRow(item, `latent-${storyIdx}`);
                rows.push({
                  key: latentRow.id || `latent-${storyIdx}`,
                  type: "latent",
                  latent: latentRow,
                });
                return;
              }
              const stepIdxRaw =
                typeof item.step_index === "number"
                  ? item.step_index
                  : Number.isFinite(Number((item as any)?.step_index))
                  ? Number((item as any)?.step_index)
                  : storyIdx;
              const stepIdx = Math.max(0, stepIdxRaw);
              if (fallbackLatentsByStep.size > 0) {
                const bucket = fallbackLatentsByStep.get(stepIdx);
                if (bucket && bucket.length > 0) {
                  pushLatentBucketRows(rows, bucket, stepIdx);
                  fallbackLatentsByStep.delete(stepIdx);
                }
              }
              const annotated = annotatedFeed[stepIdx];
              const engagement =
                annotated?.engagement ?? engagementsChrono[stepIdx];
              if (!engagement) {
                return;
              }
              rows.push({
                key: item.id || `observed-${storyIdx}-${stepIdx}`,
                type: "observed",
                observed: {
                  annotated,
                  engagement,
                },
              });
            });

            if (fallbackLatentsByStep.size > 0) {
              Array.from(fallbackLatentsByStep.entries())
                .sort((a, b) => a[0] - b[0])
                .forEach(([insertIdx, bucket]) => {
                  pushLatentBucketRows(rows, bucket, insertIdx);
                });
            }

            return rows;
          }

          const latentsToInsert =
            showLatentStory && latentActivity.length > 0 ? latentActivity : [];
          if (latentsToInsert.length === 0) {
            return annotatedFeed.map((row, idx) => ({
              key: `observed-${idx}`,
              type: "observed" as const,
              observed: {
                annotated: row,
                engagement: row.engagement,
              },
            }));
          }

          const latentsByStep = buildLatentsByStep(latentsToInsert);
          const rows: TimelineRow[] = [];

          annotatedFeed.forEach((row, idx) => {
            const bucket = latentsByStep.get(idx);
            if (bucket && bucket.length > 0) {
              pushLatentBucketRows(rows, bucket, idx);
              latentsByStep.delete(idx);
            }
            if (showLatentStory) {
              const derived = deriveLatentItemsFromBelief(
                row.belief,
                idx
              );
              derived.forEach((latent) => {
                rows.push({
                  key: latent.id,
                  type: "latent",
                  latent,
                });
              });
            }
            rows.push({
              key: `observed-${idx}`,
              type: "observed",
              observed: {
                annotated: row,
                engagement: row.engagement,
              },
            });
          });

          if (latentsByStep.size > 0) {
            Array.from(latentsByStep.entries())
              .sort((a, b) => a[0] - b[0])
              .forEach(([insertIdx, bucket]) => {
                pushLatentBucketRows(rows, bucket, insertIdx);
              });
          }

          return rows;
        })();
        const accountStoryline = thesis?.storyline ?? null;
        const storyKey = s.account_id || `section-${idx}`;
        const storyTab = storyTabByAccount[storyKey] ?? "activity";
        const storyEditsForAccount = storyEdits[storyKey] || {};
        const { observedCount, latentCount } = timelineRows.reduce(
          (acc, row) => {
            if (row.type === "observed") acc.observedCount += 1;
            else acc.latentCount += 1;
            return acc;
          },
          { observedCount: 0, latentCount: 0 }
        );

        return (
          <Card key={idx} variant="outlined" sx={{ mb: 3 }}>
            <CardHeader
              title={
                <Typography variant="h6">
                  Add Engagements {account ? `for ${labelForAccount(account)}` : ""}
                </Typography>
              }
            />
            <CardContent>
              {/* Account picker */}
              <Box sx={{ mb: 2 }}>
                <FormControl fullWidth>
                  <InputLabel id={`account-${idx}`}>Target Account</InputLabel>
                  <Select
                    labelId={`account-${idx}`}
                    label="Target Account"
                    value={s.account_id}
                    onChange={async (e) => {
                      const accountId = e.target.value as string;
                      updateSection(idx, {
                        account_id: accountId,
                        added: [],
                        hsText: "",
                        hsTextErr: "",
                      });
                      try {
                        if (selectedProductId && token && accountId) {
                          const existing = await fetchEngagementsForAccount(
                            selectedProductId,
                            accountId,
                            token
                          );
                          updateSection(idx, {
                            added: existing,
                            hsText: formatAsHsJson(existing),
                          });
                          await fetchPersonaMatches(
                            selectedProductId,
                            accountId,
                            token
                          );
                        }
                      } catch {
                        setSnack({
                          open: true,
                          msg: "Failed to load existing data",
                          sev: "warning",
                        });
                      }
                    }}
                  >
                    {targets.map((a) => (
                      <MenuItem key={a.id} value={a.id}>
                        {labelForAccount(a)}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Box>

              {/* Tabs */}
              <Box sx={{ mt: 1 }}>
                <Tabs
                  value={s.tab}
                  onChange={(_, val) => updateSection(idx, { tab: val })}
                  aria-label="engagement input mode"
                >
                  <Tab label="Manual" />
                  <Tab label="Sync HS Activity" />
                </Tabs>
              </Box>

              {/* -------- Tab: Manual -------- */}
              {s.tab === 0 && (
                <Box sx={{ mt: 2 }}>
                  {/* Form grid */}
                  <Box
                    sx={{
                      display: "grid",
                      gridTemplateColumns: { xs: "1fr", md: "1fr 1fr" },
                      gap: 2,
                    }}
                  >
                    <TextField
                      fullWidth
                      label="Timestamp (text or natural language)"
                      placeholder="e.g., 2024-01-02 10:15, today 2:30pm, yesterday 09:00"
                      value={s.timestampText}
                      onChange={(e) => handleTimestampTextChange(idx, e.target.value)}
                      onBlur={() => handleTimestampTextBlur(idx)}
                      error={!!s.timestampErr}
                      helperText={s.timestampErr || "We'll normalize to ISO Z on save"}
                    />
                    <TextField
                      fullWidth
                      type="datetime-local"
                      label="Timestamp (picker)"
                      value={s.timestampLocal}
                      onChange={(e) => handleTimestampPickerChange(idx, e.target.value)}
                      helperText="Stored as ISO (UTC Z) on save"
                      InputLabelProps={{ shrink: true }}
                    />

                    <TextField
                      fullWidth
                      label="Actor Name (optional)"
                      value={s.actorName}
                      onChange={(e) => updateSection(idx, { actorName: e.target.value })}
                    />
                    <TextField
                      fullWidth
                      label="Role / Title"
                      placeholder="e.g., CFO, Controller, PM"
                      value={s.actorRole}
                      onChange={(e) => updateSection(idx, { actorRole: e.target.value })}
                    />

                    <TextField
                      fullWidth
                      label="Department"
                      placeholder="e.g., Finance, RevOps, Product"
                      value={s.actorDept}
                      onChange={(e) => updateSection(idx, { actorDept: e.target.value })}
                    />
                    <TextField
                      fullWidth
                      label="Seniority (optional)"
                      placeholder="Executive, Director, Manager, Senior, Operator"
                      value={s.actorSeniority}
                      onChange={(e) => updateSection(idx, { actorSeniority: e.target.value })}
                      helperText="If empty, we'll infer from title."
                    />

                    <FormControl fullWidth>
                      <InputLabel id={`source-${idx}`}>Source</InputLabel>
                      <Select
                        labelId={`source-${idx}`}
                        label="Source"
                        value={s.source}
                        onChange={(e) =>
                          updateSection(idx, {
                            source: e.target.value as SectionForm["source"],
                          })
                        }
                      >
                        <MenuItem value="marketing">Marketing</MenuItem>
                        <MenuItem value="sales">Sales</MenuItem>
                        <MenuItem value="cs">Customer Success</MenuItem>
                        <MenuItem value="product">Product</MenuItem>
                        <MenuItem value="other">Other</MenuItem>
                      </Select>
                    </FormControl>

                    <TextField
                      fullWidth
                      label="Channel (optional)"
                      placeholder="e.g., webinar, email, sales_call"
                      value={s.channel}
                      onChange={(e) => updateSection(idx, { channel: e.target.value })}
                    />
                    <TextField
                      fullWidth
                      label="Asset ID (optional)"
                      placeholder="e.g., webinar_123"
                      value={s.assetId}
                      onChange={(e) => updateSection(idx, { assetId: e.target.value })}
                    />
                  </Box>

                  <TextField
                    sx={{ mt: 2 }}
                    fullWidth
                    label="Raw Activity"
                    placeholder="e.g., Webinar: Billing Pitfalls in Scale-ups"
                    value={s.rawActivity}
                    onChange={(e) => updateSection(idx, { rawActivity: e.target.value })}
                    multiline
                    minRows={3}
                  />

                  <FormControlLabel
                    sx={{ mt: 1 }}
                    control={
                      <Switch
                        checked={s.inferred}
                        onChange={(e) => updateSection(idx, { inferred: e.target.checked })}
                      />
                    }
                    label="Inferred (vs. Actual)"
                  />

                  <Box mt={2} display="flex" gap={1}>
                    <Button variant="contained" onClick={() => handleAddEngagement(idx)}>
                      Add Engagement
                    </Button>
                    <Button
                      variant="outlined"
                      onClick={() =>
                        updateSection(idx, {
                          timestampLocal: nowLocalForInput(),
                          timestampText: nowLocalForInput().replace("T", " "),
                          timestampErr: "",
                          actorName: "",
                          actorRole: "",
                          actorDept: "",
                          actorSeniority: "",
                          rawActivity: "",
                          channel: "",
                          assetId: "",
                          inferred: false,
                        })
                      }
                    >
                      Reset Form
                    </Button>
                  </Box>
                </Box>
              )}

              {/* -------- Tab: Sync HS Activity -------- */}
              {s.tab === 1 && (
                <Box sx={{ mt: 2 }}>
                  <Alert severity="warning" sx={{ mb: 2 }}>
                    Importing here will <strong>overwrite</strong> engagements for this account on the
                    server.
                  </Alert>
                  <TextField
                    fullWidth
                    multiline
                    minRows={12}
                    label="Paste HubSpot-like JSON ( { engagements: [ ... ] } )"
                    value={s.hsText}
                    onChange={(e) => updateSection(idx, { hsText: e.target.value })}
                    error={!!s.hsTextErr}
                    helperText={
                      s.hsTextErr ||
                      "Tip: pre-populated with current engagements; edit and submit to overwrite."
                    }
                    placeholder={`{
  "engagements": [
    {
      "account_id": "${s.account_id || "<account_id>"}",
      "timestamp": "2025-01-01T10:00:00Z",
      "actor": { "name": "Jane Doe", "title": "CFO", "department": "Finance", "seniority": "Executive", "confidence": 1.0 },
      "channel": "CALL",
      "source": "sales",
      "raw_activity": "Initial consultation call",
      "asset_id": null,
      "inferred": false
    }
  ]
}`}
                  />
                  <Box mt={2} display="flex" gap={1}>
                    <Button
                      variant="contained"
                      onClick={() => submitHsImport(idx)}
                      disabled={!s.account_id || !selectedProductId}
                    >
                      Import & Overwrite
                    </Button>
                    <Button
                      variant="outlined"
                      onClick={() =>
                        updateSection(idx, { hsText: formatAsHsJson(s.added), hsTextErr: "" })
                      }
                    >
                      Reset to Current
                    </Button>
                  </Box>
                </Box>
              )}

              {/* 3. Incremental Learnings (account-level) */}
              <IncrementalLearningsCard thesis={thesis} annotatedFeed={annotatedFeed} />

              {/* 4. Unified Activity Feed + Belief context */}
              {s.account_id && (
                <Box mt={3}>
                  <Divider sx={{ mb: 2 }} />
                  <Stack
                    direction={{ xs: "column", md: "row" }}
                    alignItems={{ xs: "flex-start", md: "center" }}
                    justifyContent="space-between"
                    spacing={1}
                    sx={{ mb: 1 }}
                  >
                    <Tabs
                      value={storyTab}
                      onChange={(_, val) =>
                        handleStoryTabChange(
                          storyKey,
                          (val as "activity" | "story") || "activity"
                        )
                      }
                      variant="standard"
                    >
                      <Tab label="Activity Timeline" value="activity" />
                      <Tab label="Predicted Story" value="story" />
                    </Tabs>
                    <Box sx={{ display: "flex", gap: 1, alignItems: "center", flexWrap: "wrap" }}>
                      {thesis?.overall_fit &&
                        Number.isFinite(thesis.overall_fit.best_path_probability) && (
                          <Typography variant="caption" color="text.secondary">
                            Fit: p(best) {thesis.overall_fit.best_path_probability.toFixed(2)}
                          </Typography>
                        )}
                      {storyTab === "activity" && (
                        <FormControlLabel
                          sx={{ ml: 1 }}
                          control={
                            <Switch
                              size="small"
                              checked={showLatentStory}
                              onChange={(e) => setShowLatentStory(e.target.checked)}
                            />
                          }
                          label={
                            <Typography variant="caption" color="text.secondary">
                              Show inferred
                            </Typography>
                          }
                        />
                      )}
                      <Button
                        size="small"
                        variant="outlined"
                        disabled={isMatchesLoading}
                        onClick={() =>
                          fetchPersonaMatches(selectedProductId!, s.account_id, token!)
                        }
                      >
                        {isMatchesLoading ? "Refreshing…" : "Refresh analysis"}
                      </Button>
                    </Box>
                  </Stack>
                  {storyTab === "activity" ? (
                    <>
                      <Typography variant="subtitle2" sx={{ mb: 1 }}>
                        Likely Activity Story — observed {observedCount}
                        {latentCount > 0 ? ` • inferred ${latentCount}` : ""} — earliest → latest
                      </Typography>
                      {timelineRows.length === 0 ? (
                        <Typography variant="body2" color="text.secondary">
                          No activity yet.
                        </Typography>
                      ) : (
                        <Box>
                          {timelineRows.map((row, i) => (
                            <React.Fragment key={row.key}>
                              {row.type === "latent" ? (
                                <LatentTimelineItem item={row.latent} />
                              ) : (
                                <FeedItem
                                  e={row.observed.engagement}
                                  match={row.observed.annotated?.match}
                                  belief={row.observed.annotated?.belief}
                                />
                              )}
                              {i !== timelineRows.length - 1 && <Divider sx={{ ml: 2 }} />}
                            </React.Fragment>
                          ))}
                        </Box>
                      )}
                    </>
                  ) : (
                    <Box>
                      <Typography variant="subtitle2" sx={{ mb: 1 }}>
                        Predicted Story — ZMOT → Win/Loss narrative
                      </Typography>
                      <StorylineNarrative
                        storyline={accountStoryline}
                        allowEdits
                        edits={storyEditsForAccount}
                        onEdit={(nodeId, value) => handleStoryEditChange(storyKey, nodeId, value)}
                        loading={isMatchesLoading && storyTab === "story"}
                        emptyMessage="No storyline yet. Run analysis to generate one."
                      />
                    </Box>
                  )}
                </Box>
              )}
            </CardContent>
          </Card>
        );
      })}

      <Box display="flex" gap={1} sx={{ mb: 2 }}>
        <Button variant="outlined" onClick={handleAddNewAccountSection}>
          Add New Account Engagements
        </Button>
        <Button variant="contained" onClick={handleSaveAndAnalyze}>
          Save & Analyze Engagements
        </Button>
      </Box>

      <Snackbar
        open={snack.open}
        autoHideDuration={3000}
        onClose={() => setSnack((s) => ({ ...s, open: false }))}
      >
        <Alert
          onClose={() => setSnack((s) => ({ ...s, open: false }))}
          severity={snack.sev}
          sx={{ width: "100%" }}
        >
          {snack.msg}
        </Alert>
      </Snackbar>
    </Box>
  );
}
