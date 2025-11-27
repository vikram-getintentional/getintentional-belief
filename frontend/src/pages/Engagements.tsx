// EngagementsSetup.tsx — FULL REPLACEMENT (with Global Insights + per-account learnings/feed)
import React, { useCallback, useEffect, useMemo, useState } from "react";
import {
  Box,
  Card,
  CardHeader,
  CardContent,
  Typography,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Switch,
  FormControlLabel,
  Button,
  Divider,
  Chip,
  Snackbar,
  Alert,
  CircularProgress,
  Stack,
  Tabs,
  Tab,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
} from "@mui/material";

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
  global: GlobalInsights;
  incremental: any;   // your existing type from incremental_learnings_from_thesis
  activity: any[];    // journey steps
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
};

// Global insights types
type PersonaFieldSummary = {
  field?: string | null;
  band?: LearningBand | null;
  avg_delta?: number | null;
  avg_confidence?: number | null;
  reason?: string | null;
};

export type GlobalPersonaRecommendation = {
  persona_id: string;
  persona_label?: string | null;
  observed_events: number;
  seen?: number;
  recommendation_events?: number;
  current_best_fit?: string | null;
  current_fitness?: number | null;
  avg_confidence?: number | null;
  avg_delta?: number | null;
  predicted_boost_pct?: number | null;
  reasons: string[];
  field_summaries: PersonaFieldSummary[];
  jobs: string[];
  pains: string[];
  triggers: string[];
  account_meta?: string[];
};

export type GlobalEdgeRecommendation = {
  source_id: string;
  target_id: string;
  pair_labels?: string[] | null;
  band?: string | null;
  seen: number;
  avg_confidence?: number | null;
  avg_delta?: number | null;
  predicted_boost_pct?: number | null;
  scope?: number | null;
  reason?: string | null;
  reasons: string[];
  edge_labels: string[][];
  current_relevance?: number | null;
  current_likelihood?: number | null;
  recommended_likelihood?: number | null;
  recommended_relevance?: number | null;
  recommendation_events?: number;
  account_meta?: string[];
};

type GlobalEngagementInsight = {
  classification: string;
  count: number;
  share?: number | null;
  avg_log_loss?: number | null;
};

type ArsenalImpactRow = {
  asset_id: string;
  asset_label?: string | null;
  persona_ids?: string[];
  persona_labels?: string[];
  total_delta?: number | null;
  avg_delta?: number | null;
  avg_confidence?: number | null;
  channels?: string[];
  account_meta?: string[];
  num_engagements: number;
};

export type GlobalInsights = {
  meta: {
    num_accounts: number;
    num_engagements: number;
    num_persona_recommendations: number;
    num_edge_recommendations: number;
    stability_score?: number | null;
    hit_at_1?: number | null;
    hit_at_3?: number | null;
    log_loss?: number | null;
  };
  persona_recommendations: GlobalPersonaRecommendation[];
  edge_recommendations: GlobalEdgeRecommendation[];
  engagement_insights?: GlobalEngagementInsight[];
  arsenal_impact?: ArsenalImpactRow[];
};

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

function fmtScope(value?: number | null) {
  return fmtPercent(value, { inputIsFraction: true, decimals: 0 });
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
  const ls = thesis?.learning_summary;
  const neighborhoods = thesis?.learning_neighborhoods;

  const personaInferences: PersonaPathThesis[] =
    ls?.persona_graph_inferences || [];
  const fullInferences: PersonaPathThesis[] =
    ls?.full_graph_inferences || [];

  const hasAny =
    (personaInferences && personaInferences.length > 0) ||
    (fullInferences && fullInferences.length > 0) ||
    (neighborhoods &&
      (Object.keys(neighborhoods.upstream || {}).length > 0 ||
        Object.keys(neighborhoods.handoff || {}).length > 0 ||
        Object.keys(neighborhoods.downstream || {}).length > 0)) ||
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

  const bandShortLabel: Record<LearningBand, string> = {
    upstream: "Upstream",
    handoff: "Handoff",
    downstream: "Downstream",
  };

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

  const renderNeighborhoodSection = (
    title: string,
    entries: Record<string, LearningNeighborhoodEntry> | undefined
  ) => {
    if (!entries || Object.keys(entries).length === 0) return null;
    return (
      <Box mb={2}>
        <Typography variant="subtitle2" gutterBottom>
          {title}
        </Typography>
        <Stack spacing={1.5}>
          {Object.values(entries).map((entry) => (
            <Box
              key={`${entry.anchor}-${entry.band}`}
              p={1}
              borderRadius={1}
              border={1}
              borderColor="divider"
            >
              <Stack direction="row" justifyContent="space-between" alignItems="flex-start">
                <Box>
                  <Stack direction="row" spacing={1} alignItems="center">
                    <Typography variant="body2">{entry.anchor_label}</Typography>
                    <Chip
                      size="small"
                      label={bandShortLabel[entry.band]}
                      variant="outlined"
                    />
                  </Stack>
                  {entry.rationale && (
                    <Typography
                      variant="caption"
                      color="text.secondary"
                      sx={{ display: "block", mt: 0.5 }}
                    >
                      {entry.rationale}
                    </Typography>
                  )}
                </Box>
                <Typography variant="body2">
                  Δ {entry.delta_sum > 0 ? "+" : ""}
                  {entry.delta_sum.toFixed(3)}
                </Typography>
              </Stack>

              {entry.edge_labels && entry.edge_labels.length > 0 && (
                <Typography
                  variant="caption"
                  color="text.secondary"
                  sx={{ display: "block", mt: 0.75 }}
                >
                  Impacted edges:{" "}
                  {entry.edge_labels
                    .slice(0, 4)
                    .map(([uLabel, vLabel, rel]) => `${uLabel} → ${vLabel} (${rel})`)
                    .join("; ")}
                  {entry.edge_labels.length > 4 ? " …" : ""}
                </Typography>
              )}
            </Box>
          ))}
        </Stack>
      </Box>
    );
  };

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
          {totalEvents === 0 ? (
            <Typography variant="body2" color="text.secondary">
              No engagements recorded yet for this account.
            </Typography>
          ) : (
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
        </Box>

        {poorPersonaFits.length > 0 && (
          <Box sx={{ mb: 2 }}>
            <Typography variant="subtitle2" gutterBottom>
              Personas with weak fit signals
            </Typography>
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
          </Box>
        )}

        {surprisingEngagements.length > 0 && (
          <Box sx={{ mb: 3 }}>
            <Typography variant="subtitle2" gutterBottom>
              Highly surprising engagements
            </Typography>
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
          </Box>
        )}

        {engagementImpactRows.length > 0 && (
          <Box sx={{ mb: 3 }}>
            <Typography variant="subtitle2" gutterBottom>
              Engagement-level insights
            </Typography>
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
          </Box>
        )}

        {personaInferences.length > 0 && (
          <Box sx={{ mb: 2 }}>
            <Stack
              direction="row"
              spacing={1}
              alignItems="center"
              sx={{ mb: 0.5 }}
            >
              <Typography variant="subtitle2">Persona path inferences</Typography>
              <Chip size="small" label={personaInferences.length} />
            </Stack>
            <Stack spacing={1.5}>
              {personaInferences.map((inf, idx) => (
                <Box key={idx}>
                  <Stack direction="row" spacing={1} alignItems="center">
                    <Typography variant="body2">
                      {inf.anchor_label || inf.anchor_persona_id || "Persona"}
                    </Typography>
                    {inf.band && (
                      <Chip
                        size="small"
                        label={bandShortLabel[inf.band]}
                        variant="outlined"
                      />
                    )}
                  </Stack>
                  <Typography variant="caption" color="text.secondary">
                    {inf.diagnosis}
                    {inf.rationale ? ` · ${inf.rationale}` : ""}
                  </Typography>
                </Box>
              ))}
            </Stack>
          </Box>
        )}

        {fullInferences.length > 0 && (
          <Box sx={{ mb: 2 }}>
            <Stack
              direction="row"
              spacing={1}
              alignItems="center"
              sx={{ mb: 0.5 }}
            >
              <Typography variant="subtitle2">Graph-level inferences</Typography>
              <Chip size="small" label={fullInferences.length} />
            </Stack>
            <Stack spacing={1.5}>
              {fullInferences.map((inf, idx) => (
                <Box key={idx}>
                  <Stack direction="row" spacing={1} alignItems="center">
                    <Typography variant="body2">
                      {inf.anchor_label || "Graph motif"}
                    </Typography>
                    {inf.band && (
                      <Chip
                        size="small"
                        label={bandShortLabel[inf.band]}
                        variant="outlined"
                      />
                    )}
                  </Stack>
                  <Typography variant="caption" color="text.secondary">
                    {inf.diagnosis}
                    {inf.rationale ? ` · ${inf.rationale}` : ""}
                  </Typography>
                </Box>
              ))}
            </Stack>
          </Box>
        )}

        {neighborhoods && (
          <>
            <Divider sx={{ my: 2 }} />
            <Typography variant="subtitle2" gutterBottom>
              Neighborhoods of impact
            </Typography>
            <Typography
              variant="caption"
              color="text.secondary"
              sx={{ display: "block", mb: 1.5 }}
            >
              Local regions where edge weights are most likely misestimated for this
              account&apos;s journey.
            </Typography>

            {renderNeighborhoodSection(
              "Upstream (starting personas / triggers)",
              neighborhoods.upstream
            )}
            {renderNeighborhoodSection(
              "Handoff (persona→persona chains)",
              neighborhoods.handoff
            )}
            {renderNeighborhoodSection(
              "Downstream (persona→product influence)",
              neighborhoods.downstream
            )}
          </>
        )}
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

  const [globalInsights, setGlobalInsights] = useState<GlobalInsights | null>(null);
  const [arsenalMetaGaps, setArsenalMetaGaps] = useState<{
    assets: string[];
    channels: string[];
  }>({ assets: [], channels: [] });
  const [showAllEdgeRecs, setShowAllEdgeRecs] = useState(false);

  const refreshGlobalInsights = useCallback(
    async (productId: string | null) => {
      if (!productId || !token) return;
      try {
        const res = await fetch(
          `http://localhost:8000/journey/global-thesis/${productId}`,
          {
            headers: { Authorization: `Bearer ${token}` },
          }
        );
        if (!res.ok) return;
        const data = await res.json();
        if (data?.insights) {
          const insights = data.insights as GlobalInsights;
          insights.persona_recommendations.forEach((rec) => {
            registerPersonaLabel(rec.persona_id, rec.persona_label || null);
          });
          setGlobalInsights(insights);
          setShowAllEdgeRecs(false);
        }
      } catch (err) {
        console.warn("Failed to refresh global insights", err);
      }
    },
    [token]
  );

  const applyGlobalRecommendation = useCallback(
    async (type: "persona" | "edge", recommendation: any) => {
      if (!selectedProductId || !token) return;
      try {
        const res = await fetch(
          "http://localhost:8000/journey/apply-recommendation",
          {
            method: "POST",
            headers: {
              Authorization: `Bearer ${token}`,
              "Content-Type": "application/json",
            },
            body: JSON.stringify({
              product_id: selectedProductId,
              type,
              recommendation,
            }),
          }
        );
        if (!res.ok) {
          throw new Error("Failed to apply recommendation");
        }
        const payload = await res.json();
        if (payload?.applied) {
          console.log("[journey/apply-recommendation] applied", payload.applied);
        }
        setSnack({
          open: true,
          msg: payload?.message || "Graph updated",
          sev: "success",
        });
        await refreshGlobalInsights(selectedProductId);
      } catch (err: any) {
        setSnack({
          open: true,
          msg: err?.message || "Unable to update graph",
          sev: "error",
        });
      }
    },
    [selectedProductId, token, refreshGlobalInsights]
  );

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
        await refreshGlobalInsights(picked);

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
  }, [token, refreshGlobalInsights]);

  const accountsById = useMemo(() => {
    const m = new Map<string, TargetAccount>();
    targets.forEach((t) => m.set(t.id, t));
    return m;
  }, [targets]);

  const freeAccounts = useMemo(() => {
    const used = new Set(sections.map((s) => s.account_id).filter(Boolean));
    return targets.filter((t) => !used.has(t.id));
  }, [targets, sections]);

  const arsenalImpact = useMemo(
    () => globalInsights?.arsenal_impact ?? [],
    [globalInsights]
  );

  useEffect(() => {
    setArsenalMetaGaps({ assets: [], channels: [] });
  }, [selectedProductId]);

  const highScopeEdgeRecs = useMemo(() => {
    if (!globalInsights) return [];
    return globalInsights.edge_recommendations.filter(
      (edge) => (edge.scope ?? 0) >= 0.999
    );
  }, [globalInsights]);

  const edgesForDisplay = useMemo(() => {
    if (!globalInsights) return [];
    if (showAllEdgeRecs) return globalInsights.edge_recommendations;
    if (highScopeEdgeRecs.length === 0) return globalInsights.edge_recommendations;
    return highScopeEdgeRecs;
  }, [globalInsights, showAllEdgeRecs, highScopeEdgeRecs]);

  const hasAdditionalEdgeRecs = useMemo(() => {
    if (!globalInsights) return false;
    return (
      highScopeEdgeRecs.length > 0 &&
      highScopeEdgeRecs.length < globalInsights.edge_recommendations.length
    );
  }, [globalInsights, highScopeEdgeRecs]);

/*const globalInsights = useMemo(
  () => computeGlobalInsights(beliefByAccount, sections),
  [beliefByAccount, sections]
);*/

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
        const insights = (data.global as GlobalInsights) ?? null;
        if (insights) {
          insights.persona_recommendations.forEach((rec) => {
            registerPersonaLabel(rec.persona_id, rec.persona_label || null);
          });
          setGlobalInsights(insights);
          setShowAllEdgeRecs(false);
        } else {
          setGlobalInsights(null);
          setShowAllEdgeRecs(false);
        }
      }

      const payload = data?.thesis ?? data ?? {};

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
      };

      setBeliefByAccount((prev) => ({ ...prev, [accountId]: thesis }));

      // Refresh global rollups so the recommendations card stays in sync with the latest run.
      await refreshGlobalInsights(productId);
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

    const classificationChip =
      belief &&
      (() => {
        const common = {
          size: "small" as const,
          variant: "outlined" as const,
        };
        switch (belief.classification) {
          case "expected":
            return <Chip {...common} color="success" label="On-path" />;
          case "jump_ahead":
            return <Chip {...common} color="warning" label="On-path (jumped)" />;
          case "off_path":
            return <Chip {...common} color="error" label="Off-path" />;
          case "no_persona":
            return <Chip {...common} color="default" label="No persona match" />;
          case "no_path":
          default:
            return <Chip {...common} color="default" label="No belief path yet" />;
        }
      })();

    const qualityChip =
      typeof belief?.quality === "number" ? (
        <Chip
          size="small"
          variant="outlined"
          label={`quality ${(belief.quality * 100).toFixed(0)}%`}
        />
      ) : null;

    const surpriseChip =
    typeof belief?.surpriseScore === "number" ? (
        <Chip
        size="small"
        color={
            belief.surpriseScore > 0.7
            ? "error"
            : belief.surpriseScore > 0.4
            ? "warning"
            : "default"
        }
        variant="outlined"
        label={`surprise ${(belief.surpriseScore * 100).toFixed(0)}%`}
        />
    ) : null;


    return (
      <Box sx={{ display: "flex", gap: 2 }}>
        {/* timeline rail */}
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
              bgcolor: "primary.main",
              mt: 0.75,
            }}
          />
          <Box sx={{ flex: 1, width: 2, bgcolor: "divider" }} />
        </Box>

        <Box sx={{ flex: 1, pb: 2 }}>
          {/* Belief context */}
          {belief && (
            <Box
              sx={{
                mb: 0.5,
                display: "flex",
                flexWrap: "wrap",
                alignItems: "center",
                gap: 1,
              }}
            >
              <Typography variant="caption" color="text.secondary">
                Expected before:&nbsp;
                <strong>
                  {belief.expectedBefore
                    ? personaLabelFromId(belief.expectedBefore)
                    : "—"}
                </strong>
              </Typography>
              {classificationChip}
              {qualityChip}
              {surpriseChip}
              {belief.nextExpectedAfter && (
                <Typography variant="caption" color="text.secondary">
                  Next expected:&nbsp;
                  <strong>{personaLabelFromId(belief.nextExpectedAfter)}</strong>
                </Typography>
              )}
              {belief.note && (
                <Typography
                  variant="caption"
                  color="text.secondary"
                  sx={{ display: "block", width: "100%" }}
                >
                  {belief.note}
                </Typography>
              )}
              {belief.impact && (
                <Box sx={{ width: "100%", mt: 0.75 }}>
                  <Typography variant="caption" color="text.secondary">
                    Graph impact:
                  </Typography>
                  {belief.impact.edges.length === 0 ? (
                    <Typography
                      variant="caption"
                      color="text.secondary"
                      sx={{ display: "block" }}
                    >
                      No meaningful edge adjustments recorded.
                    </Typography>
                  ) : (
                    <Stack spacing={0.25} sx={{ mt: 0.5 }}>
                      {belief.impact.edges.slice(0, 4).map((edge, idx) => (
                        <Typography
                          key={`${edge.from}-${edge.to}-${idx}`}
                          variant="caption"
                          color="text.secondary"
                          sx={{ display: "block" }}
                        >
                          {(edge.fromLabel || personaLabelFromId(edge.from)) ??
                            edge.from}{" "}
                          → {(edge.toLabel || personaLabelFromId(edge.to)) ?? edge.to}{" "}
                          {edge.supports ? "(supports)" : "(contradicts)"}{" "}
                          {typeof edge.deltaLogProb === "number"
                            ? `Δ log p ${edge.deltaLogProb >= 0 ? "+" : ""}${edge.deltaLogProb.toFixed(3)}`
                            : ""}
                        </Typography>
                      ))}
                    </Stack>
                  )}
                </Box>
              )}
            </Box>
          )}

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

      {/* 1. Global Insights */}
      <Card variant="outlined" sx={{ mb: 3 }}>
        <CardHeader title="Global Recommendations" />
        <CardContent>
          {!globalInsights ? (
            <Typography variant="body2" color="text.secondary">
              Global insights populate after you save engagements and run the
              journey analysis.
            </Typography>
          ) : (
            <>
              <Box
                display="flex"
                flexWrap="wrap"
                gap={4}
                alignItems="flex-start"
                mb={3}
              >
                <Box>
                  <Typography variant="overline">Accounts observed</Typography>
                  <Typography variant="h5">
                    {fmtCount(globalInsights.meta.num_accounts)}
                  </Typography>
                </Box>
                <Box>
                  <Typography variant="overline">Total engagements</Typography>
                  <Typography variant="h5">
                    {fmtCount(globalInsights.meta.num_engagements)}
                  </Typography>
                </Box>
                <Box>
                  <Typography variant="overline">Persona recs</Typography>
                  <Typography variant="h6">
                    {fmtCount(globalInsights.meta.num_persona_recommendations)}
                  </Typography>
                </Box>
                <Box>
                  <Typography variant="overline">Edge recs</Typography>
                  <Typography variant="h6">
                    {fmtCount(globalInsights.meta.num_edge_recommendations)}
                  </Typography>
                </Box>
              </Box>

              <Typography variant="subtitle1" gutterBottom>
                Persona addition recommendations
              </Typography>
              {globalInsights.persona_recommendations.length === 0 ? (
                <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                  No persona-level recommendations yet. Feed more qualified
                  journeys to surface patterns.
                </Typography>
              ) : (
                <TableContainer component={Paper} variant="outlined" sx={{ mb: 3 }}>
                  <Table size="small">
                    <TableHead>
                    <TableRow>
                      <TableCell>Persona</TableCell>
                      <TableCell>Current best-fit</TableCell>
                      <TableCell align="right">Fitness</TableCell>
                      <TableCell align="right">Observed</TableCell>
                      <TableCell align="right">Updates</TableCell>
                      <TableCell align="right">Pred. boost</TableCell>
                      <TableCell align="right">Confidence</TableCell>
                      <TableCell align="left">Account meta</TableCell>
                      <TableCell align="right">Action</TableCell>
                    </TableRow>
                    </TableHead>
                    <TableBody>
                      {globalInsights.persona_recommendations.map((rec) => {
                        const personaLabel =
                          rec.persona_label || personaLabelFromId(rec.persona_id);
                        const observedCount =
                          typeof rec.observed_events === "number"
                            ? rec.observed_events
                            : rec.seen ?? 0;
                        const updateCount =
                          typeof rec.recommendation_events === "number"
                            ? rec.recommendation_events
                            : rec.seen ?? 0;
                        return (
                          <TableRow key={rec.persona_id}>
                            <TableCell sx={{ maxWidth: 280 }}>
                              <Typography variant="body2">{personaLabel}</Typography>
                              {rec.reasons.length > 0 && (
                                <Typography
                                  variant="caption"
                                  color="text.secondary"
                                  sx={{ display: "block" }}
                                >
                                  {rec.reasons[0]}
                                </Typography>
                              )}
                            </TableCell>
                            <TableCell sx={{ maxWidth: 200 }}>
                              {rec.current_best_fit || "—"}
                            </TableCell>
                            <TableCell align="right">
                              {fmtPercent(rec.current_fitness, {
                                inputIsFraction: true,
                                decimals: 0,
                              })}
                            </TableCell>
                            <TableCell align="right">
                              {fmtCount(observedCount)}
                            </TableCell>
                            <TableCell align="right">
                              {fmtCount(updateCount)}
                            </TableCell>
                            <TableCell align="right">
                              {fmtPercent(rec.predicted_boost_pct, {
                                sign: true,
                                decimals: 0,
                              })}
                            </TableCell>
                        <TableCell align="right">
                          {fmtPercent(rec.avg_confidence, {
                            inputIsFraction: true,
                            decimals: 0,
                          })}
                        </TableCell>
                        <TableCell sx={{ maxWidth: 260 }}>
                          {rec.account_meta && rec.account_meta.length > 0
                            ? rec.account_meta.join(", ")
                            : "—"}
                        </TableCell>
                        <TableCell align="right">
                          <Button
                            size="small"
                            variant="outlined"
                            disabled={
                              !selectedProductId || !rec.jobs || rec.jobs.length === 0
                            }
                            onClick={() => applyGlobalRecommendation("persona", rec)}
                          >
                            Update graph
                          </Button>
                        </TableCell>
                      </TableRow>
                    );
                  })}
            </TableBody>
          </Table>
                </TableContainer>
              )}

              <Typography variant="subtitle1" gutterBottom>
                Edge update recommendations
              </Typography>
              {globalInsights.edge_recommendations.length === 0 ? (
                <Typography variant="body2" color="text.secondary">
                  No edge-level updates yet. You&apos;ll see recommended handoffs
                  once the engine spots recurring misweighted paths.
                </Typography>
              ) : (
                <TableContainer component={Paper} variant="outlined">
                  <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Relationship</TableCell>
                    <TableCell align="right">Current lik.</TableCell>
                    <TableCell align="right">Current rel.</TableCell>
                    <TableCell align="right">Recommended lik.</TableCell>
                    <TableCell align="right">Recommended rel.</TableCell>
                    <TableCell align="right">Pred. boost</TableCell>
                    <TableCell align="right">Scope</TableCell>
                    <TableCell align="right">Confidence</TableCell>
                    <TableCell align="left">Account meta</TableCell>
                    <TableCell align="right">Action</TableCell>
                  </TableRow>
                </TableHead>
                    <TableBody>
                      {edgesForDisplay.map((edge) => (
                        <TableRow key={`${edge.source_id}-${edge.target_id}`}>
                          <TableCell sx={{ maxWidth: 320 }}>
                            <Typography variant="body2">
                              {(edge.pair_labels || [])
                                .filter(Boolean)
                                .join(" → ") || "Proposed handoff"}
                            </Typography>
                            {edge.reason && (
                              <Typography
                                variant="caption"
                                color="text.secondary"
                                sx={{ display: "block" }}
                              >
                                {edge.reason}
                              </Typography>
                              )}
                            </TableCell>
                            <TableCell align="right">
                              {fmtPercent(edge.current_likelihood, {
                                inputIsFraction: true,
                                decimals: 0,
                              })}
                            </TableCell>
                            <TableCell align="right">
                              {fmtPercent(edge.current_relevance, {
                                inputIsFraction: true,
                                decimals: 0,
                              })}
                            </TableCell>
                            <TableCell align="right">
                              {fmtPercent(edge.recommended_likelihood, {
                                inputIsFraction: true,
                                decimals: 0,
                              })}
                            </TableCell>
                            <TableCell align="right">
                              {fmtPercent(edge.recommended_relevance, {
                                inputIsFraction: true,
                                decimals: 0,
                              })}
                            </TableCell>
                            <TableCell align="right">
                              {fmtPercent(edge.predicted_boost_pct, {
                                sign: true,
                                decimals: 0,
                              })}
                            </TableCell>
                            <TableCell align="right">
                              {fmtScope(edge.scope)}
                            </TableCell>
                            <TableCell align="right">
                              {fmtPercent(edge.avg_confidence, {
                                inputIsFraction: true,
                                decimals: 0,
                              })}
                            </TableCell>
                            <TableCell sx={{ maxWidth: 260 }}>
                              {edge.account_meta && edge.account_meta.length > 0
                                ? edge.account_meta.join(", ")
                                : "—"}
                            </TableCell>
                            <TableCell align="right">
                              <Button
                                size="small"
                                variant="outlined"
                                disabled={!selectedProductId || (edge.scope ?? 0) < 0.95}
                                onClick={() => applyGlobalRecommendation("edge", edge)}
                              >
                                Update graph
                              </Button>
                            </TableCell>
                          </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              )}
              {hasAdditionalEdgeRecs && (
                <Box sx={{ mt: 1 }}>
                  <Button
                    size="small"
                    onClick={() => setShowAllEdgeRecs((prev) => !prev)}
                  >
                    {showAllEdgeRecs
                      ? "Hide lower scope recommendations"
                      : "View all recommendations"}
                  </Button>
                </Box>
              )}

              <Typography variant="subtitle1" gutterBottom sx={{ mt: 3 }}>
                Arsenal impact
              </Typography>
              {arsenalImpact.length === 0 ? (
                <Typography variant="body2" color="text.secondary">
                  No arsenal-level impact recorded yet. Once engagements are logged with assets,
                  we’ll estimate belief shifts per playbook.
                </Typography>
              ) : (
                <TableContainer component={Paper} variant="outlined" sx={{ mt: 1 }}>
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>Arsenal asset</TableCell>
                        <TableCell>Personas impacted</TableCell>
                        <TableCell align="right">Total Δ log&nbsp;p</TableCell>
                        <TableCell align="right">Avg confidence</TableCell>
                        <TableCell>Channels</TableCell>
                        <TableCell>Account meta</TableCell>
                        <TableCell align="right">Engagements</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {arsenalImpact.map((row) => {
                        const personaLabels =
                          (row.persona_labels && row.persona_labels.length > 0
                            ? row.persona_labels
                            : (row.persona_ids || []).map((pid) =>
                                personaLabelFromId(pid)
                              )) || [];
                        const channels = row.channels || [];
                        const accountMeta = row.account_meta || [];
                        return (
                          <TableRow key={row.asset_id}>
                            <TableCell sx={{ maxWidth: 280 }}>
                              <Typography variant="body2">
                                {row.asset_label || row.asset_id}
                              </Typography>
                              <Typography
                                variant="caption"
                                color="text.secondary"
                                sx={{ display: "block" }}
                              >
                                id: {row.asset_id}
                              </Typography>
                            </TableCell>
                            <TableCell sx={{ maxWidth: 260 }}>
                              {personaLabels.length === 0
                                ? "—"
                                : personaLabels.join(", ")}
                            </TableCell>
                            <TableCell align="right">
                              {row.total_delta === undefined ||
                              row.total_delta === null
                                ? "—"
                                : `${row.total_delta >= 0 ? "+" : ""}${row.total_delta.toFixed(3)}`}
                            </TableCell>
                            <TableCell align="right">
                              {fmtPercent(row.avg_confidence, {
                                inputIsFraction: true,
                                decimals: 0,
                              })}
                            </TableCell>
                            <TableCell>
                              {channels.length === 0 ? "—" : channels.join(", ")}
                            </TableCell>
                            <TableCell sx={{ maxWidth: 260 }}>
                              {accountMeta.length === 0
                                ? "—"
                                : accountMeta.join(", ")}
                            </TableCell>
                            <TableCell align="right">
                              {fmtCount(row.num_engagements)}
                            </TableCell>
                          </TableRow>
                        );
                      })}
                    </TableBody>
                  </Table>
                </TableContainer>
              )}
            </>
          )}
        </CardContent>
      </Card>

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

        const annotatedFeed =
          s.added.length && matches
            ? annotateEngagementsForAccount(
                s.added,
                matches,
                candidatePaths,
                journeySteps,
                thesis?.learning_summary?.top_edge_updates
              )
            : [];

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
                    direction="row"
                    alignItems="center"
                    justifyContent="space-between"
                    sx={{ mb: 1 }}
                  >
                    <Typography variant="subtitle2">
                      Activity ({s.added.length}) — earliest → latest
                    </Typography>
                    <Box sx={{ display: "flex", gap: 1, alignItems: "center" }}>
                      {thesis?.overall_fit &&
                        Number.isFinite(thesis.overall_fit.best_path_probability) && (
                          <Typography variant="caption" color="text.secondary">
                            Fit: p(best){" "}
                            {thesis.overall_fit.best_path_probability.toFixed(2)}
                          </Typography>
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
                  {s.added.length === 0 ? (
                    <Typography variant="body2" color="text.secondary">
                      No activity yet.
                    </Typography>
                  ) : (
                    <Box>
                      {annotatedFeed.map(({ engagement, match, belief }, i) => (
                        <React.Fragment key={i}>
                          <FeedItem e={engagement} match={match} belief={belief} />
                          {i !== annotatedFeed.length - 1 && <Divider sx={{ ml: 2 }} />}
                        </React.Fragment>
                      ))}
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
