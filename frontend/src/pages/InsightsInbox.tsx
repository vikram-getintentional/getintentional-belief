import { useCallback, useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  CardHeader,
  Chip,
  CircularProgress,
  IconButton,
  List,
  ListItem,
  ListItemText,
  Paper,
  Stack,
  Tab,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Tabs,
  Tooltip,
  Typography,
} from "@mui/material";
import InfoOutlinedIcon from "@mui/icons-material/InfoOutlined";
import ProvenanceChip, {
  type TextInsight,
} from "../components/ProvenanceChip";
import type { PlanningMode } from "../types/apiContracts";

type PersonaFieldSummary = {
  field?: string | null;
  band?: string | null;
  avg_delta?: number | null;
  avg_confidence?: number | null;
  reason?: string | null;
};

type GlobalPersonaRecommendation = {
  persona_id: string;
  persona_label?: string | null;
  reason?: string | null;
  observed_events: number;
  accounts?: number | null;
  hit_at_1_with?: number | null;
  hit_at_1_without?: number | null;
  seen?: number;
  recommendation_events?: number;
  current_best_fit?: string | null;
  current_fitness?: number | null;
  avg_confidence?: number | null;
  avg_delta?: number | null;
  predicted_boost_pct?: number | null;
  confidence?: number | null;
  reasons: string[];
  field_summaries: PersonaFieldSummary[];
  jobs: string[];
  pains: string[];
  triggers: string[];
  account_meta?: string[];
};

type GlobalEdgeRecommendation = {
  source_id: string;
  target_id: string;
  pair_labels?: string[] | null;
  band?: string | null;
  seen: number;
  avg_confidence?: number | null;
  avg_delta?: number | null;
  predicted_boost_pct?: number | null;
  confidence?: number | null;
  count?: number | null;
  accounts?: number | null;
  impact_pct?: number | null;
  ci_low?: number | null;
  ci_high?: number | null;
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
  personas?: string[];
  from_persona?: string;
  to_persona?: string;
  proposed_likelihood?: number | null;
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
  approx_delta_win_pct?: number | null;
  effect_label?: string | null;
  confidence?: number | null;
  events?: number | null;
  sample_accounts?: number | null;
  personas_impacted?: string[];
  channels?: string[];
  account_meta?: string[];
  num_engagements: number;
  funnel_stage?: {
    code?: string | null;
    label?: string | null;
    avg_perceptibility?: number | null;
    avg_proximity?: number | null;
  } | null;
};

type KeystonePersonaInsight = {
  type?: string;
  persona_id: string;
  persona: string;
  wolves_score?: number | null;
  delta_win_bp?: number | null;
  involvement_rate?: number | null;
  sample_size?: number | null;
  segment?: {
    key?: string;
    label?: string;
    [key: string]: any;
  } | null;
  chain_targets?: string[];
  chain_share?: number | null;
  coalition_path?: string[];
  recommended_moves?: Array<{
    belief_transition?: string | null;
    best_asset?: string | null;
    best_channel?: string | null;
    bps_lift?: number | null;
  }>;
  explanation?: string | null;
};

type ProductInsightsPayload = {
  ideal_customer_patterns?: {
    works_well?: string[];
    gaps?: string[];
    works_well_items?: InsightListItem[];
    gaps_items?: InsightListItem[];
  };
  persona_landscape?: {
    frequency?: string[];
    critical_leads?: string[];
    decision_personas?: string[];
    blockers?: string[];
    coalitions?: string[];
  };
  belief_transitions?: {
    hardest?: TextInsight;
    easiest?: TextInsight;
    top_pains?: TextInsight[];
  };
  asset_channel_effectiveness?: {
    high_assets?: string[];
    underperforming_channels?: string[];
    channel_persona_matches?: string[];
  };
  journey_structure?: {
    common_paths?: string[];
    deviations?: string;
    average_duration_days?: number | string;
    typical_path?: Array<{
    step: number;
    persona: string;
    highlight?: string | null;
    impact?: string | null;
    confidence?: number | null;
    reason?: string | null;
    source?: string;
  }>;
  };
  global_patterns?: {
    biggest_barrier?: TextInsight;
    hidden_blocker?: TextInsight;
    missed_opportunity?: TextInsight;
  };
  product_strengths?: {
    strengths?: string[];
    weaknesses?: string[];
  };
  strategic_moves?: {
    segment_priorities?: string;
    persona_priorities?: string;
    asset_priorities?: string;
    channel_priorities?: string;
  };
  keystone_personas?: KeystonePersonaInsight[];
  wolves_metrics_updated_at?: string | null;
};

type InsightListItem = {
  text: string;
  label?: string | null;
  meta_key?: string | null;
  signals?: number | null;
  support?: Record<string, any>;
};

const makeDefaultInsight = (text: string): TextInsight => ({
  text,
  source: "default",
});

const toTextInsight = (
  value: TextInsight | string | undefined,
  fallbackText: string
): TextInsight => {
  if (!value && !fallbackText) {
    return makeDefaultInsight(NO_DATA_TEXT);
  }
  if (value && typeof value !== "string") {
    return value;
  }
  return makeDefaultInsight(value ?? fallbackText);
};

const toTextInsightList = (
  values: Array<TextInsight | string> | undefined,
  fallback: string[]
): TextInsight[] => {
  if (values && values.length) {
    return values.map((value) =>
      typeof value === "string" ? makeDefaultInsight(value) : value
    );
  }
  return fallback.map((text) => makeDefaultInsight(text));
};

type GlobalInsights = {
  meta: {
    num_accounts: number;
    num_engagements: number;
    num_persona_recommendations: number;
    num_edge_recommendations: number;
    stability_score?: number | null;
    hit_at_1?: number | null;
    hit_at_3?: number | null;
    log_loss?: number | null;
    wolves_metrics_updated_at?: string | null;
  };
  persona_recommendations: GlobalPersonaRecommendation[];
  edge_recommendations: GlobalEdgeRecommendation[];
  engagement_insights?: GlobalEngagementInsight[];
  arsenal_impact?: ArsenalImpactRow[];
  product_insights?: ProductInsightsPayload;
};

type TabKey = "product" | "inbox";

type PersonaIdentifier = string | null | undefined;

const personaLabelOverrides = new Map<string, string>();

function normalizePersonaId(raw?: PersonaIdentifier) {
  if (!raw) return "";
  return String(raw).trim().toLowerCase();
}

function registerPersonaLabel(id?: PersonaIdentifier, label?: string | null) {
  const key = normalizePersonaId(id);
  if (!key || !label) return;
  personaLabelOverrides.set(key, label);
}

function titleCase(value: string) {
  return value
    .split(/[\s_/|]+/)
    .filter(Boolean)
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
    .join(" ");
}

function personaLabelFromId(rawId?: PersonaIdentifier) {
  const id = normalizePersonaId(rawId);
  if (!id) return "";
  const override = personaLabelOverrides.get(id);
  if (override) return override;
  if (id.includes("|")) {
    const [t = "", d = "", s = ""] = id.split("|");
    return [titleCase(t), titleCase(d), titleCase(s)]
      .filter(Boolean)
      .join(" | ");
  }
  const tail = id.includes(":") ? id.split(":").pop() || id : id;
  if (/^[a-f0-9-]+$/i.test(tail)) {
    return tail;
  }
  return titleCase(tail.replace(/[_/|]+/g, " "));
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

function fmtBasisPoints(value?: number | null) {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return `${Number(value).toFixed(1)} bps`;
}

type SignalTier = "high" | "medium" | "low";

const SIGNAL_THRESHOLDS: Record<SignalTier, number> = {
  high: 0.8,
  medium: 0.6,
  low: 0.0, // fallback
};

const SIGNAL_LABELS: Record<SignalTier, string> = {
  high: "High signal",
  medium: "Medium signal",
  low: "Exploratory",
};

const SIGNAL_COLORS: Record<SignalTier, "success" | "warning" | "default"> = {
  high: "success",
  medium: "warning",
  low: "default",
};

function getSignalTier(confidence?: number | null): SignalTier {
  const score = confidence ?? 0;
  if (score >= SIGNAL_THRESHOLDS.high) return "high";
  if (score >= SIGNAL_THRESHOLDS.medium) return "medium";
  return "low";
}

function signalBadgeProps(confidence?: number | null) {
  const tier = getSignalTier(confidence);
  return {
    label: `${SIGNAL_LABELS[tier]}${typeof confidence === "number" ? ` · ${fmtPercent(confidence, { decimals: 0, inputIsFraction: true })}` : ""}`,
    color: SIGNAL_COLORS[tier],
  };
}

function shouldShowTypicalMetrics(value?: number | null) {
  if (value === null || value === undefined) return false;
  if (Number.isNaN(value)) return false;
  return true;
}

function hasMeaningfulPathMetrics(
  perc?: number | null,
  prox?: number | null,
  involvement?: number | null
) {
  if (
    perfEqual(perc, 0) &&
    perfEqual(prox, 1) &&
    perfEqual(involvement, 0)
  ) {
    return false;
  }
  return (
    shouldShowTypicalMetrics(perc) ||
    shouldShowTypicalMetrics(prox) ||
    shouldShowTypicalMetrics(involvement)
  );
}

function perfEqual(value?: number | null, target?: number) {
  if (value === null || value === undefined) return false;
  if (target === undefined) return false;
  return Math.abs(value - target) < 1e-6;
}

function formatArsenalEffect(row: ArsenalImpactRow) {
  const delta = row.approx_delta_win_pct ?? row.avg_delta;
  const label =
    delta === null || delta === undefined
      ? row.effect_label || "Effect unknown"
      : delta > 0
      ? "Helps"
      : delta < 0
      ? "Hurts"
      : "Neutral";
  const detail =
    delta === null || delta === undefined
      ? "Insufficient data"
      : `${delta > 0 ? "+" : ""}${fmtPercent(delta, {
          decimals: 1,
          sign: true,
        })} win prob`;
  return { label, detail };
}

const MAX_PRIMARY_PERSONA_CARDS = 3;
const MAX_MEDIUM_PERSONA_CARDS = 2;
const MIN_EDGE_EVENT_THRESHOLD = 5;

type InsightSupportProps = {
  summary: string | null;
  lowSignal: boolean;
};

function getInsightSupportProps(
  support?: Record<string, any>,
  signals?: number | null
): InsightSupportProps {
  const parts: string[] = [];
  const accounts = support?.accounts;
  const events = support?.events;
  if (typeof accounts === "number" && accounts > 0) {
    parts.push(`${fmtCount(accounts)} accounts`);
  }
  if (typeof events === "number" && events > 0) {
    parts.push(`${fmtCount(events)} events`);
  }
  if (typeof signals === "number" && signals > 0) {
    parts.push(`${fmtCount(signals)} signals`);
  }
  const summary = parts.length ? parts.join(" · ") : null;
  const lowSignal =
    support?.low_signal === true ||
    (typeof events === "number" && events > 0 && events < 3) ||
    (typeof accounts === "number" && accounts > 0 && accounts < 3);
  return { summary, lowSignal };
}

function renderInsightListItems(items: InsightListItem[]) {
  return items.map((item, idx) => {
    const { summary, lowSignal } = getInsightSupportProps(
      item.support,
      item.signals ?? null
    );
    const key = item.text || item.label || `insight-${idx}`;
    return (
      <ListItem key={key} disablePadding sx={{ pl: 0 }}>
        <Stack direction="row" spacing={1} alignItems="flex-start" width="100%">
          <Box sx={{ flex: 1 }}>
            <Typography
              variant="body2"
              color={lowSignal ? "text.secondary" : "text.primary"}
            >
              {item.text}
            </Typography>
            {summary && (
              <Typography variant="caption" color="text.secondary">
                {summary}
              </Typography>
            )}
          </Box>
          {lowSignal && (
            <Chip size="small" variant="outlined" label="Low signal" />
          )}
        </Stack>
      </ListItem>
    );
  });
}

const formatRelativeTime = (iso?: string | null) => {
  if (!iso) return null;
  const parsed = new Date(iso);
  if (Number.isNaN(parsed.getTime())) return null;
  const diffMs = Date.now() - parsed.getTime();
  if (diffMs < 0) return "just now";
  const minutes = Math.floor(diffMs / 60000);
  if (minutes < 1) return "just now";
  if (minutes < 60) return `${minutes} min${minutes === 1 ? "" : "s"} ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours} hr${hours === 1 ? "" : "s"} ago`;
  const days = Math.floor(hours / 24);
  return `${days} day${days === 1 ? "" : "s"} ago`;
};

function formatBadgeLabel(value: number) {
  if (value >= 100) return "99+";
  return String(value);
}

const NO_DATA_TEXT = "Not enough data yet.";
const NO_DATA_LIST = [NO_DATA_TEXT];

const SECTION_A_WORKS_DEFAULT = NO_DATA_LIST;
const SECTION_A_GAPS_DEFAULT = NO_DATA_LIST;
const SECTION_B_FREQUENCY_DEFAULT = NO_DATA_LIST;
const SECTION_B_CRITICAL_LEADS_DEFAULT = NO_DATA_LIST;
const SECTION_B_DECISION_PERSONAS_DEFAULT = NO_DATA_LIST;
const SECTION_B_BLOCKERS_DEFAULT = NO_DATA_LIST;
const SECTION_B_COALITIONS_DEFAULT = NO_DATA_LIST;
const SECTION_C_HARDEST_DEFAULT = NO_DATA_TEXT;
const SECTION_C_EASIEST_DEFAULT = NO_DATA_TEXT;
const SECTION_C_PAIN_NODES_DEFAULT = NO_DATA_LIST;
const SECTION_D_HIGH_IMPACT_ASSETS_DEFAULT = NO_DATA_LIST;
const SECTION_D_UNDERPERFORMING_CHANNELS_DEFAULT = NO_DATA_LIST;
const SECTION_D_CHANNEL_MATCHES_DEFAULT = NO_DATA_LIST;
const SECTION_E_COMMON_PATHWAYS_DEFAULT = NO_DATA_LIST;
const SECTION_E_DEVIATIONS_DEFAULT = NO_DATA_TEXT;
const SECTION_E_DURATION_DEFAULT = "—";
const SECTION_F_BARRIERS_DEFAULT = NO_DATA_TEXT;
const SECTION_F_HIDDEN_DEFAULT = NO_DATA_TEXT;
const SECTION_F_MISSED_DEFAULT = NO_DATA_TEXT;
const SECTION_G_STRENGTHS_DEFAULT = NO_DATA_LIST;
const SECTION_G_WEAKNESSES_DEFAULT = NO_DATA_LIST;
const SECTION_H_SEGMENT_DEFAULT = NO_DATA_TEXT;
const SECTION_H_PERSONA_DEFAULT = NO_DATA_TEXT;
const SECTION_H_ASSET_DEFAULT = NO_DATA_TEXT;
const SECTION_H_CHANNEL_DEFAULT = NO_DATA_TEXT;

const ATTR_COLOR_MAP: Record<
  string,
  "default" | "primary" | "secondary" | "success" | "info" | "warning" | "error"
> = {
  industry: "warning",
  geography: "info",
  revenue_range: "success",
  employee_range: "secondary",
  funding_stage: "primary",
  account_type: "info",
};

const fmtDecimal = (value?: number | null, digits = 2) => {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return Number(value).toFixed(digits).replace(/\.00$/, "");
};

const parseMetaInsight = (text: string) => {
  const match = text.match(/^Attr\s+([^:]+):(.*?)(\s+(shows|drags).*)$/i);
  if (!match) {
    return {
      chipLabel: null,
      chipValue: null,
      remainder: text,
      color: "default" as const,
    };
  }
  const rawAttr = match[1].trim().toLowerCase();
  const chipLabel = rawAttr.replace(/_/g, " ");
  const chipValue = match[2].trim();
  const remainder = match[3].trim();
  const color = ATTR_COLOR_MAP[rawAttr] || "default";
  return { chipLabel, chipValue, remainder, color };
};

const formatInsightsPlanningModeLabel = (mode?: PlanningMode) => {
  switch (mode) {
    case "observed_signal":
      return "Observed signal mode";
    case "learned_strategy":
      return "Learned strategy mode";
    default:
      return "Graph Hypothesis mode";
  }
};

const InsightsInbox = () => {
  const navigate = useNavigate();
  const token = typeof window !== "undefined" ? localStorage.getItem("token") : null;
  const [selectedProductId, setSelectedProductId] = useState<string | null>(null);
  const [globalInsights, setGlobalInsights] = useState<GlobalInsights | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<TabKey>("product");
  const [showAllEdgeRecs, setShowAllEdgeRecs] = useState(false);
  const [showAllHighPersonas, setShowAllHighPersonas] = useState(false);
  const [showMediumSignalPersonas, setShowMediumSignalPersonas] =
    useState(false);
  const [showExploratoryPersonas, setShowExploratoryPersonas] =
    useState(false);
  const [arsenalMetaGaps, setArsenalMetaGaps] = useState<{
    assets: string[];
    channels: string[];
  }>({ assets: [], channels: [] });

  const fetchInsights = useCallback(
    async (productId: string) => {
      if (!token) return;
      try {
        const res = await fetch(
          `http://localhost:8000/journey/global-thesis/${productId}`,
          {
            headers: { Authorization: `Bearer ${token}` },
          }
        );
        if (!res.ok) {
          throw new Error("Failed to load insights");
        }
        const data = await res.json();
        if (data?.insights) {
          const insights = data.insights as GlobalInsights;
          insights.persona_recommendations.forEach((rec) =>
            registerPersonaLabel(rec.persona_id, rec.persona_label || null)
          );
          setGlobalInsights(insights);
          setShowAllEdgeRecs(false);

          const actionableCount =
            (insights.persona_recommendations?.length || 0) +
            (insights.edge_recommendations?.length || 0);
          window.dispatchEvent(new CustomEvent("insights:refresh-count", { detail: actionableCount }));

          setArsenalMetaGaps({
            assets:
              (insights.arsenal_impact || [])
                .filter((row) => (row.total_delta || 0) !== 0 && (row.account_meta || []).length === 0)
                .map((row) => row.asset_label || row.asset_id) ?? [],
            channels: [],
          });
        } else {
          setGlobalInsights(null);
          setArsenalMetaGaps({ assets: [], channels: [] });
          window.dispatchEvent(new CustomEvent("insights:refresh-count", { detail: 0 }));
        }
      } catch (err: any) {
        console.warn("Failed to fetch insights inbox", err);
        setError(err?.message || "Unable to load insights");
        setGlobalInsights(null);
        setArsenalMetaGaps({ assets: [], channels: [] });
        window.dispatchEvent(new CustomEvent("insights:refresh-count", { detail: 0 }));
      }
    },
    [token]
  );

  useEffect(() => {
    if (!token) {
      setError("Missing auth token");
      setLoading(false);
      return;
    }
    (async () => {
      try {
        const meRes = await fetch("http://localhost:8000/me", {
          headers: { Authorization: `Bearer ${token}` },
        });
        if (!meRes.ok) throw new Error("Failed to load profile");
        const me = await meRes.json();

        const productsRes = await fetch(
          `http://localhost:8000/get-products/${me.company_id}`,
          { headers: { Authorization: `Bearer ${token}` } }
        );
        if (!productsRes.ok) throw new Error("Failed to load products");
        const productsData = await productsRes.json();
        const firstProduct = productsData?.products?.[0]?.id;
        if (!firstProduct) {
          setError("No products found. Run Value Prop first.");
          setLoading(false);
          return;
        }
        setSelectedProductId(firstProduct);
        await fetchInsights(firstProduct);
      } catch (err: any) {
        setError(err?.message || "Unable to load insights");
      } finally {
        setLoading(false);
      }
    })();
  }, [token, fetchInsights]);

  const actionableCount = useMemo(() => {
    if (!globalInsights) return 0;
    return (
      (globalInsights.persona_recommendations?.length || 0) +
      (globalInsights.edge_recommendations?.length || 0)
    );
  }, [globalInsights]);

  const edgeInsights = useMemo(() => {
    if (!globalInsights) {
      return {
        display: [] as GlobalEdgeRecommendation[],
        lowSignalCount: 0,
        priorityCount: 0,
        totalCount: 0,
      };
    }
    const all = globalInsights.edge_recommendations || [];
    const priority = all.filter((edge) => {
      const confidence = edge.confidence ?? edge.avg_confidence ?? 0;
      const count = edge.count ?? edge.recommendation_events ?? edge.seen ?? 0;
      return (
        confidence >= SIGNAL_THRESHOLDS.medium && count >= MIN_EDGE_EVENT_THRESHOLD
      );
    });
    const lowSignalCount = Math.max(0, all.length - priority.length);
    return {
      display: showAllEdgeRecs ? all : priority,
      lowSignalCount,
      priorityCount: priority.length,
      totalCount: all.length,
    };
  }, [globalInsights, showAllEdgeRecs]);

  const personaRecommendations = globalInsights?.persona_recommendations ?? [];
  const personaRecsSorted = useMemo(() => {
    return [...personaRecommendations].sort((a, b) => {
      const aScore = a.confidence ?? a.avg_confidence ?? 0;
      const bScore = b.confidence ?? b.avg_confidence ?? 0;
      return bScore - aScore;
    });
  }, [personaRecommendations]);
  const personaHighSignal = useMemo(
    () =>
      personaRecsSorted.filter((rec) => {
        const conf = rec.confidence ?? rec.avg_confidence ?? 0;
        return getSignalTier(conf) === "high";
      }),
    [personaRecsSorted]
  );
  const personaMediumSignal = useMemo(
    () =>
      personaRecsSorted.filter((rec) => {
        const conf = rec.confidence ?? rec.avg_confidence ?? 0;
        return getSignalTier(conf) === "medium";
      }),
    [personaRecsSorted]
  );
  const personaLowSignal = useMemo(
    () =>
      personaRecsSorted.filter((rec) => {
        const conf = rec.confidence ?? rec.avg_confidence ?? 0;
        return getSignalTier(conf) === "low";
      }),
    [personaRecsSorted]
  );
  const personaHighDisplay = showAllHighPersonas
    ? personaHighSignal
    : personaHighSignal.slice(0, MAX_PRIMARY_PERSONA_CARDS);
  const personaHighHiddenCount = Math.max(
    0,
    personaHighSignal.length - personaHighDisplay.length
  );
  const personaMediumDisplay = personaMediumSignal.slice(
    0,
    MAX_MEDIUM_PERSONA_CARDS
  );
  const personaMediumCount = personaMediumSignal.length;
  const personaMediumHiddenCount = Math.max(
    0,
    personaMediumSignal.length - personaMediumDisplay.length
  );
  const personaLowCount = personaLowSignal.length;
  const personaLowDisplay = showExploratoryPersonas ? personaLowSignal : [];

  const productInsights = globalInsights?.product_insights;
  const arsenalImpact = globalInsights?.arsenal_impact ?? [];
  const keystoneInsights = productInsights?.keystone_personas ?? [];
  const wolvesMetricsUpdatedAt =
    productInsights?.wolves_metrics_updated_at ||
    globalInsights?.meta?.wolves_metrics_updated_at ||
    null;
  const wolvesUpdatedLabel = formatRelativeTime(wolvesMetricsUpdatedAt);

  const sectionAWorksItems =
    productInsights?.ideal_customer_patterns?.works_well_items ??
    (productInsights?.ideal_customer_patterns?.works_well ||
      SECTION_A_WORKS_DEFAULT).map((text) => ({
      text,
      label: text,
      meta_key: null,
      signals: null,
      support: {},
    }));
  const sectionAGapsItems =
    productInsights?.ideal_customer_patterns?.gaps_items ??
    (productInsights?.ideal_customer_patterns?.gaps ||
      SECTION_A_GAPS_DEFAULT).map((text) => ({
      text,
      label: text,
      meta_key: null,
      signals: null,
      support: {},
    }));

  const personaFrequency =
    productInsights?.persona_landscape?.frequency || SECTION_B_FREQUENCY_DEFAULT;
  const criticalLeadPersonasList =
    productInsights?.persona_landscape?.critical_leads ||
    SECTION_B_CRITICAL_LEADS_DEFAULT;
  const decisionPersonasList =
    productInsights?.persona_landscape?.decision_personas ||
    SECTION_B_DECISION_PERSONAS_DEFAULT;
  const criticalBlockers =
    productInsights?.persona_landscape?.blockers || SECTION_B_BLOCKERS_DEFAULT;
  const personaCoalitions =
    productInsights?.persona_landscape?.coalitions || SECTION_B_COALITIONS_DEFAULT;

  const highImpactAssets =
    productInsights?.asset_channel_effectiveness?.high_assets ||
    SECTION_D_HIGH_IMPACT_ASSETS_DEFAULT;
  const underperformingChannels =
    productInsights?.asset_channel_effectiveness?.underperforming_channels ||
    SECTION_D_UNDERPERFORMING_CHANNELS_DEFAULT;
  const channelPersonaMatches =
    productInsights?.asset_channel_effectiveness?.channel_persona_matches ||
    SECTION_D_CHANNEL_MATCHES_DEFAULT;
  const typicalPath = productInsights?.journey_structure?.typical_path ?? null;

  const beliefHardest = toTextInsight(
    productInsights?.belief_transitions?.hardest,
    SECTION_C_HARDEST_DEFAULT
  );
  const beliefEasiest = toTextInsight(
    productInsights?.belief_transitions?.easiest,
    SECTION_C_EASIEST_DEFAULT
  );
  const beliefPains = toTextInsightList(
    productInsights?.belief_transitions?.top_pains,
    SECTION_C_PAIN_NODES_DEFAULT
  );

  const commonPathways =
    productInsights?.journey_structure?.common_paths ||
    SECTION_E_COMMON_PATHWAYS_DEFAULT;
  const journeyDeviation =
    productInsights?.journey_structure?.deviations || SECTION_E_DEVIATIONS_DEFAULT;
  const journeyDurationRaw =
    productInsights?.journey_structure?.average_duration_days;
  const journeyDuration =
    typeof journeyDurationRaw === "number"
      ? `${journeyDurationRaw.toFixed(1)} days`
      : typeof journeyDurationRaw === "string" && journeyDurationRaw.trim()
      ? journeyDurationRaw
      : SECTION_E_DURATION_DEFAULT;

  const globalBarrier = toTextInsight(
    productInsights?.global_patterns?.biggest_barrier,
    SECTION_F_BARRIERS_DEFAULT
  );
  const hiddenBlocker = toTextInsight(
    productInsights?.global_patterns?.hidden_blocker,
    SECTION_F_HIDDEN_DEFAULT
  );
  const missedOpportunity = toTextInsight(
    productInsights?.global_patterns?.missed_opportunity,
    SECTION_F_MISSED_DEFAULT
  );

  const productStrengthsList =
    productInsights?.product_strengths?.strengths || SECTION_G_STRENGTHS_DEFAULT;
  const productWeaknessesList =
    productInsights?.product_strengths?.weaknesses || SECTION_G_WEAKNESSES_DEFAULT;

  const strategicSegment =
    productInsights?.strategic_moves?.segment_priorities ||
    SECTION_H_SEGMENT_DEFAULT;
  const strategicPersona =
    productInsights?.strategic_moves?.persona_priorities ||
    SECTION_H_PERSONA_DEFAULT;
  const strategicAsset =
    productInsights?.strategic_moves?.asset_priorities || SECTION_H_ASSET_DEFAULT;
  const strategicChannel =
    productInsights?.strategic_moves?.channel_priorities ||
    SECTION_H_CHANNEL_DEFAULT;

  const handleApplyRecommendation = useCallback(
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
        await fetchInsights(selectedProductId);
      } catch (err) {
        console.error("Unable to apply recommendation", err);
      }
    },
    [selectedProductId, token, fetchInsights]
  );

  const renderPersonaCard = (rec: GlobalPersonaRecommendation) => {
    const label =
      rec.persona_label || personaLabelFromId(rec.persona_id) || rec.persona_id;
    const confidence = rec.confidence ?? rec.avg_confidence ?? null;
    const { label: signalLabelText, color: signalColor } = signalBadgeProps(
      confidence
    );
    const effectLabel = rec.predicted_boost_pct
      ? `${fmtPercent(rec.predicted_boost_pct, { decimals: 1, sign: true })} next-step hit@1`
      : "Effect data pending";
    const confidenceTag =
      typeof confidence === "number"
        ? ` (confidence ${fmtPercent(confidence, {
            decimals: 0,
            inputIsFraction: true,
          })})`
        : "";
    const evidenceParts: string[] = [];
    if (rec.accounts) {
      evidenceParts.push(`${fmtCount(rec.accounts)} accounts`);
    }
    const observedEvents = rec.observed_events ?? rec.recommendation_events ?? 0;
    if (observedEvents) {
      evidenceParts.push(`${fmtCount(observedEvents)} engagements`);
    }
    const evidenceText = evidenceParts.length
      ? `Seen in ${evidenceParts.join(", ")}.`
      : null;
    const hitLine =
      rec.hit_at_1_with != null && rec.hit_at_1_without != null
        ? `Hit@1 ${fmtPercent(rec.hit_at_1_without, {
            decimals: 0,
            inputIsFraction: true,
          })} → ${fmtPercent(rec.hit_at_1_with, {
            decimals: 0,
            inputIsFraction: true,
          })}`
        : null;
    const reason =
      rec.reason || (rec.reasons?.length ? rec.reasons.join(" • ") : null);
    const jobLine =
      rec.jobs && rec.jobs.length ? `Jobs: ${rec.jobs.slice(0, 3).join(", ")}` : null;
    const painLine =
      rec.pains && rec.pains.length ? `Pains: ${rec.pains.slice(0, 3).join(", ")}` : null;

    return (
      <Box
        key={rec.persona_id}
        sx={{
          border: 1,
          borderColor: "divider",
          borderRadius: 1,
          p: 2,
        }}
      >
        <Stack
          direction={{ xs: "column", sm: "row" }}
          spacing={2}
          justifyContent="space-between"
        >
          <Box>
            <Typography variant="subtitle1">{label}</Typography>
            <Typography variant="body2" color="text.secondary">
              Effect: {effectLabel}
              {confidenceTag}
            </Typography>
            {evidenceText && (
              <Typography
                variant="body2"
                color="text.secondary"
                sx={{ mt: 0.5 }}
              >
                {evidenceText}
              </Typography>
            )}
            {hitLine && (
              <Typography variant="caption" color="text.secondary">
                {hitLine}
              </Typography>
            )}
          </Box>
          <Stack
            direction="row"
            spacing={1}
            alignItems="center"
            sx={{ width: { xs: "100%", sm: "auto" } }}
          >
            <Chip
              size="small"
              variant="outlined"
              label={signalLabelText}
              color={signalColor}
            />
            <Button
              variant="contained"
              size="small"
              onClick={() => handleApplyRecommendation("persona", rec)}
            >
              Approve & add persona
            </Button>
          </Stack>
        </Stack>
        {reason && (
          <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
            {reason}
          </Typography>
        )}
        {jobLine && (
          <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5 }}>
            {jobLine}
          </Typography>
        )}
        {painLine && (
          <Typography variant="caption" color="text.secondary">
            {painLine}
          </Typography>
        )}
      </Box>
    );
  };

  if (loading) {
    return (
      <Box sx={{ display: "flex", justifyContent: "center", mt: 6 }}>
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Box sx={{ maxWidth: 960, mx: "auto", p: { xs: 2, md: 3 } }}>
        <Alert severity="error">{error}</Alert>
      </Box>
    );
  }

  return (
    <Box sx={{ maxWidth: 1100, mx: "auto", p: { xs: 2, md: 3 } }}>
      <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 1 }}>
        <Typography variant="h4" fontWeight={700}>
          Insights Inbox
        </Typography>
        <Chip
          size="small"
          variant="outlined"
          label={formatInsightsPlanningModeLabel()}
        />
      </Stack>
      {(arsenalMetaGaps.assets.length > 0 ||
        arsenalMetaGaps.channels.length > 0) && (
        <Alert severity="warning" sx={{ mb: 2 }}>
          {[
            arsenalMetaGaps.assets.length
              ? `${arsenalMetaGaps.assets.length} asset${arsenalMetaGaps.assets.length === 1 ? "" : "s"} missing metadata`
              : null,
            arsenalMetaGaps.channels.length
              ? `${arsenalMetaGaps.channels.length} channel${arsenalMetaGaps.channels.length === 1 ? "" : "s"} missing metadata`
              : null,
          ]
            .filter(Boolean)
            .join(" • ")}
        </Alert>
      )}
      <Tabs
        value={tab}
        onChange={(_, value) => setTab(value as TabKey)}
        sx={{ mb: 3 }}
      >
        <Tab value="product" label="Product insights" />
        <Tab
          value="inbox"
          label={
            <Stack direction="row" spacing={1} alignItems="center">
              <span>Inbox</span>
              {actionableCount > 0 && (
                <Chip
                  size="small"
                  color="primary"
                  label={formatBadgeLabel(actionableCount)}
                />
              )}
            </Stack>
          }
        />
      </Tabs>

      {tab === "product" && (
        <Stack spacing={3}>
          {keystoneInsights.length ? (
            <Card variant="outlined">
              <CardHeader
                title="Keystone personas · Wolves impact"
                subheader="Personas whose presence disproportionately shifts win odds and unlocks downstream coalitions."
                action={
                  wolvesUpdatedLabel ? (
                    <Typography variant="caption" color="text.secondary">
                      Refreshed {wolvesUpdatedLabel}
                    </Typography>
                  ) : null
                }
              />
              <CardContent>
                <Stack spacing={1.5}>
                  {keystoneInsights.map((insight) => {
                    const wolvesScoreLabel =
                      typeof insight.wolves_score === "number"
                        ? fmtPercent(insight.wolves_score, {
                            inputIsFraction: true,
                            decimals: 0,
                          })
                        : null;
                    const deltaLabel = fmtBasisPoints(insight.delta_win_bp);
                    const involvementLabel =
                      typeof insight.involvement_rate === "number"
                        ? fmtPercent(insight.involvement_rate, {
                            inputIsFraction: true,
                            decimals: 0,
                          })
                        : null;
                    const sampleLabel =
                      typeof insight.sample_size === "number" && insight.sample_size > 0
                        ? `${fmtCount(insight.sample_size)} episodes`
                        : null;
                    const segmentLabel =
                      insight.segment?.label || insight.segment?.industry || "Portfolio";
                    const chainTargets =
                      insight.chain_targets && insight.chain_targets.length
                        ? insight.chain_targets.join(" → ")
                        : null;
                    const chainShare =
                      typeof insight.chain_share === "number"
                        ? fmtPercent(insight.chain_share, {
                            inputIsFraction: true,
                            decimals: 0,
                          })
                        : null;
                    return (
                      <Paper
                        key={`keystone-${insight.persona_id}`}
                        variant="outlined"
                        sx={{ p: 1.5 }}
                      >
                        <Stack
                          direction={{ xs: "column", sm: "row" }}
                          spacing={1}
                          alignItems={{ xs: "flex-start", sm: "center" }}
                        >
                          <Box sx={{ flex: 1 }}>
                            <Typography variant="h6">{insight.persona}</Typography>
                            {insight.explanation ? (
                              <Typography variant="body2" color="text.secondary">
                                {insight.explanation}
                              </Typography>
                            ) : null}
                          </Box>
                          <Stack direction="row" spacing={1} sx={{ flexWrap: "wrap" }}>
                            {wolvesScoreLabel && (
                              <Chip size="small" color="success" label={`Wolves ${wolvesScoreLabel}`} />
                            )}
                            {deltaLabel !== "—" && (
                              <Chip size="small" variant="outlined" label={`Δ ${deltaLabel}`} />
                            )}
                            {involvementLabel && (
                              <Chip size="small" variant="outlined" label={`Involvement ${involvementLabel}`} />
                            )}
                            {sampleLabel && (
                              <Chip size="small" variant="outlined" label={sampleLabel} />
                            )}
                          </Stack>
                        </Stack>
                        <Stack direction="row" spacing={1} sx={{ flexWrap: "wrap", mt: 1 }}>
                          <Chip size="small" color="default" label={segmentLabel} />
                          {chainTargets && (
                            <Chip
                              size="small"
                              color="success"
                              variant="outlined"
                              label={
                                chainShare
                                  ? `Unlocks ${chainTargets} (${chainShare})`
                                  : `Unlocks ${chainTargets}`
                              }
                            />
                          )}
                        </Stack>
                        {insight.recommended_moves && insight.recommended_moves.length ? (
                          <Box sx={{ mt: 1 }}>
                            <Typography variant="subtitle2" sx={{ mb: 0.5 }}>
                              Recommended moves
                            </Typography>
                            <List dense>
                              {insight.recommended_moves.slice(0, 2).map((move, idx) => (
                                <ListItem
                                  key={`keystone-move-${insight.persona_id}-${idx}`}
                                  disablePadding
                                  sx={{ pl: 0 }}
                                >
                                  <ListItemText
                                    primary={`${move.best_asset || "Asset"}${
                                      move.best_channel ? ` · ${move.best_channel}` : ""
                                    }`}
                                    secondary={`${move.belief_transition || "Belief transition"} · ${
                                      move.bps_lift !== null && move.bps_lift !== undefined
                                        ? fmtBasisPoints(move.bps_lift)
                                        : "—"
                                    }`}
                                  />
                                </ListItem>
                              ))}
                            </List>
                          </Box>
                        ) : null}
                        <Stack
                          direction={{ xs: "column", sm: "row" }}
                          spacing={1}
                          sx={{ mt: 1.5, flexWrap: "wrap" }}
                        >
                          <Button
                            size="small"
                            variant="contained"
                            color="primary"
                            onClick={() => navigate("/marketing-planner")}
                          >
                            Focus in planner
                          </Button>
                          <Button
                            size="small"
                            variant="outlined"
                            onClick={() => navigate("/account-plan")}
                          >
                            View accounts
                          </Button>
                        </Stack>
                      </Paper>
                    );
                  })}
                </Stack>
              </CardContent>
            </Card>
          ) : null}

          <Card variant="outlined">
            <CardHeader
              title="Section A · Ideal Customer Patterns (ICP Archetypes)"
              subheader="These insights describe which kinds of companies tend to convert strongly vs weakly."
            />
            <CardContent>
              <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1 }}>
                ✔ What Works Well
              </Typography>
              <List dense>
                {sectionAWorksItems.map((item) => {
                  const cleanText = item.text.replace(/^“|”$/g, "");
                  const { chipValue, remainder, color } = parseMetaInsight(cleanText);
                  const metaConfidence =
                    typeof item.support?.meta_confidence === "number"
                      ? Math.round((item.support.meta_confidence ?? 0) * 100)
                      : null;
                  const supportProps = getInsightSupportProps(
                    item.support,
                    item.signals ?? null
                  );
                  const isLowSignal =
                    item.support?.is_significant === false ||
                    supportProps.lowSignal;
                  return (
                  <ListItem
                    key={`sectionA-works-${item.text}`}
                    disablePadding
                    sx={{ pl: 0 }}
                  >
                    <ListItemText
                      primary={
                        <Stack direction="row" spacing={1} alignItems="center">
                          {chipValue ? (
                            <Chip
                              size="small"
                              color={color}
                              label={chipValue}
                              sx={{ textTransform: "capitalize" }}
                            />
                          ) : null}
                          <Typography component="span">{remainder || cleanText}</Typography>
                          {isLowSignal && (
                            <Tooltip title="Signal strength is low — more wins/losses needed">
                              <Chip size="small" color="warning" label="Low signal" variant="outlined" />
                            </Tooltip>
                          )}
                          {item.signals !== null && (
                            <Tooltip
                              title={
                                <Box>
                                  <Typography
                                    variant="body2"
                                    sx={{ fontWeight: 600 }}
                                  >
                                    Supporting signals
                                  </Typography>
                                  <Typography variant="body2">
                                    Total signals: {fmtCount(item.signals)}
                                  </Typography>
                                  {item.support?.wins !== undefined && (
                                    <Typography variant="body2">
                                      Wins: {fmtCount(item.support?.wins || 0)}
                                      {" · "}
                                      Losses: {fmtCount(item.support?.losses || 0)}
                                      {" · "}
                                      Open: {fmtCount(item.support?.open || 0)}
                                    </Typography>
                                  )}
                                  {typeof item.support?.avg_duration_days ===
                                    "number" && (
                                    <Typography variant="body2">
                                      Avg duration:{" "}
                                      {item.support?.avg_duration_days?.toFixed(
                                        1
                                      )}{" "}
                                      days
                                    </Typography>
                                  )}
                                  {Array.isArray(item.support?.accounts) &&
                                    item.support?.accounts.length > 0 && (
                                      <Typography variant="body2">
                                        Accounts:{" "}
                                        {item.support.accounts.join(", ")}
                                      </Typography>
                                    )}
                                  {metaConfidence !== null && (
                                    <Typography variant="body2">
                                      Meta confidence: {metaConfidence}%
                                    </Typography>
                                  )}
                                </Box>
                              }
                              arrow
                            >
                              <IconButton size="small" sx={{ p: 0.25 }}>
                                <InfoOutlinedIcon fontSize="inherit" />
                              </IconButton>
                            </Tooltip>
                          )}
                        </Stack>
                      }
                      secondary={supportProps.summary ?? undefined}
                      secondaryTypographyProps={{
                        component: "span",
                        variant: "caption",
                        color: "text.secondary",
                      }}
                    />
                  </ListItem>
                );})}
              </List>
              <Typography variant="subtitle2" sx={{ fontWeight: 600, mt: 2, mb: 1 }}>
                ✖ What Doesn’t Work Well
              </Typography>
              <List dense>
                {sectionAGapsItems.map((item) => {
                  const cleanText = item.text.replace(/^“|”$/g, "");
                  const { chipValue, remainder, color } = parseMetaInsight(cleanText);
                  const metaConfidence =
                    typeof item.support?.meta_confidence === "number"
                      ? Math.round((item.support.meta_confidence ?? 0) * 100)
                      : null;
                  const supportProps = getInsightSupportProps(
                    item.support,
                    item.signals ?? null
                  );
                  const isLowSignal =
                    item.support?.is_significant === false ||
                    supportProps.lowSignal;
                  return (
                  <ListItem
                    key={`sectionA-gaps-${item.text}`}
                    disablePadding
                    sx={{ pl: 0 }}
                  >
                    <ListItemText
                      primary={
                        <Stack direction="row" spacing={1} alignItems="center">
                          {chipValue ? (
                            <Chip
                              size="small"
                              color={color}
                              label={chipValue}
                              sx={{ textTransform: "capitalize" }}
                            />
                          ) : null}
                          <Typography component="span">{remainder || cleanText}</Typography>
                          {isLowSignal && (
                            <Tooltip title="Signal strength is low — more evidence needed">
                              <Chip size="small" color="warning" label="Low signal" variant="outlined" />
                            </Tooltip>
                          )}
                          {item.signals !== null && (
                            <Tooltip
                              title={
                                <Box>
                                  <Typography
                                    variant="body2"
                                    sx={{ fontWeight: 600 }}
                                  >
                                    Supporting signals
                                  </Typography>
                                  <Typography variant="body2">
                                    Total signals: {fmtCount(item.signals)}
                                  </Typography>
                                  {item.support?.wins !== undefined && (
                                    <Typography variant="body2">
                                      Wins: {fmtCount(item.support?.wins || 0)}
                                      {" · "}
                                      Losses: {fmtCount(item.support?.losses || 0)}
                                      {" · "}
                                      Open: {fmtCount(item.support?.open || 0)}
                                    </Typography>
                                  )}
                                  {typeof item.support?.avg_duration_days ===
                                    "number" && (
                                    <Typography variant="body2">
                                      Avg duration:{" "}
                                      {item.support?.avg_duration_days?.toFixed(
                                        1
                                      )}{" "}
                                      days
                                    </Typography>
                                  )}
                                  {Array.isArray(item.support?.accounts) &&
                                    item.support?.accounts.length > 0 && (
                                      <Typography variant="body2">
                                        Accounts:{" "}
                                        {item.support.accounts.join(", ")}
                                      </Typography>
                                    )}
                                  {metaConfidence !== null && (
                                    <Typography variant="body2">
                                      Meta confidence: {metaConfidence}%
                                    </Typography>
                                  )}
                                </Box>
                              }
                              arrow
                            >
                              <IconButton size="small" sx={{ p: 0.25 }}>
                                <InfoOutlinedIcon fontSize="inherit" />
                              </IconButton>
                            </Tooltip>
                          )}
                        </Stack>
                      }
                      secondary={supportProps.summary ?? undefined}
                      secondaryTypographyProps={{
                        component: "span",
                        variant: "caption",
                        color: "text.secondary",
                      }}
                    />
                  </ListItem>
                );})}
              </List>
            </CardContent>
          </Card>

          <Card variant="outlined">
            <CardHeader
              title="Section B · Persona Landscape Overview"
              subheader="Who matters, who appears, who blocks, who moves the belief needle."
            />
            <CardContent>
              <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1 }}>
                ✔ Persona Frequency Distribution
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                Across all accounts, these personas appear most frequently:
              </Typography>
              <List dense>
                {personaFrequency.map((line) => (
                  <ListItem key={`persona-freq-${line}`} disablePadding sx={{ pl: 0 }}>
                    <ListItemText primary={line} />
                  </ListItem>
                ))}
              </List>

              <Typography variant="subtitle2" sx={{ fontWeight: 600, mt: 2, mb: 1 }}>
                ✔ Critical Lead Personas (High Influence)
              </Typography>
              <List dense>
                {criticalLeadPersonasList.map((line) => (
                  <ListItem key={`lead-${line}`} disablePadding sx={{ pl: 0 }}>
                    <ListItemText primary={line} />
                  </ListItem>
                ))}
              </List>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                These personas consistently move the earliest-stage belief — engage them early for faster momentum.
              </Typography>

              <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1 }}>
                ✔ Decision Personas (Core Deciders)
              </Typography>
              <List dense>
                {decisionPersonasList.map((line) => (
                  <ListItem key={`decision-${line}`} disablePadding sx={{ pl: 0 }}>
                    <ListItemText primary={line} />
                  </ListItem>
                ))}
              </List>

              <Typography variant="subtitle2" sx={{ fontWeight: 600, mt: 2, mb: 1 }}>
                ✔ Critical Blocker Personas
              </Typography>
              <List dense>
                {criticalBlockers.map((line) => (
                  <ListItem key={`blocker-${line}`} disablePadding sx={{ pl: 0 }}>
                    <ListItemText primary={line} />
                  </ListItem>
                ))}
              </List>

              <Typography variant="subtitle2" sx={{ fontWeight: 600, mt: 2, mb: 1 }}>
                ✔ Persona Coalitions (Who Moves Together)
              </Typography>
              <List dense>
                {personaCoalitions.map((line) => (
                  <ListItem key={`coalition-${line}`} disablePadding sx={{ pl: 0 }}>
                    <ListItemText primary={line} />
                  </ListItem>
                ))}
              </List>
            </CardContent>
          </Card>

          <Card variant="outlined">
            <CardHeader
              title="Section C · Belief Transitions"
              subheader="Where belief stalls, accelerates, and which pains surface most."
            />
            <CardContent>
              <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 0.5 }}>
                Hardest Jump
              </Typography>
              <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 1.5 }}>
                <Typography variant="body2">{beliefHardest.text}</Typography>
                <ProvenanceChip provenance={beliefHardest} />
              </Stack>

              <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 0.5 }}>
                Easiest Jump
              </Typography>
              <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 1.5 }}>
                <Typography variant="body2">{beliefEasiest.text}</Typography>
                <ProvenanceChip provenance={beliefEasiest} />
              </Stack>

              <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 0.5 }}>
                Top Pain Signals
              </Typography>
              <List dense>
                {beliefPains.map((pain, idx) => (
                  <ListItem key={`belief-pain-${idx}`} sx={{ pl: 0 }}>
                    <ListItemText primary={pain.text} />
                    <ProvenanceChip provenance={pain} />
                  </ListItem>
                ))}
              </List>
            </CardContent>
          </Card>

          <Card variant="outlined">
            <CardHeader title="Typical conversion path" />
            <CardContent>
              {typicalPath ? (
                <Stack spacing={2}>
                  {typicalPath.map((step) => {
                    const confidencePct =
                      typeof step.confidence === "number"
                        ? Math.round(step.confidence * 100)
                        : null;
                    const supportProps = getInsightSupportProps(
                      step.support,
                      step.signals ?? null
                    );
                    const textColor = supportProps.lowSignal
                      ? "text.secondary"
                      : "text.primary";
                    return (
                      <Box key={step.step}>
                        <Stack direction="row" spacing={1} alignItems="center">
                          <Typography variant="subtitle2" color={textColor}>
                            Step {step.step}: {step.persona}
                          </Typography>
                          {confidencePct !== null && (
                            <Chip
                              size="small"
                              color={
                                confidencePct >= 70
                                  ? "success"
                                  : confidencePct >= 40
                                  ? "warning"
                                  : "default"
                              }
                              label={`confidence ${confidencePct}%`}
                            />
                          )}
                          {supportProps.lowSignal && (
                            <Chip
                              size="small"
                              variant="outlined"
                              label="Low signal"
                            />
                          )}
                          <ProvenanceChip provenance={step} />
                        </Stack>
                        <Typography variant="body2" color={textColor}>
                          {step.highlight && step.highlight.trim()
                            ? step.highlight
                            : "Key activation step"}
                        </Typography>
                        {supportProps.summary && (
                          <Typography
                            variant="caption"
                            color="text.secondary"
                            sx={{ display: "block" }}
                          >
                            {supportProps.summary}
                          </Typography>
                        )}
                        {step.reason && step.reason.trim() && (
                          <Typography
                            variant="caption"
                            color="text.secondary"
                            sx={{ display: "block", mt: 0.5 }}
                          >
                            {step.reason}
                          </Typography>
                        )}
                        <Typography variant="caption" color="text.secondary">
                          {step.impact || "—"}
                        </Typography>
                        {hasMeaningfulPathMetrics(
                          step.perceptibility,
                          step.proximity,
                          step.involvement
                        ) && (
                          <Typography
                            variant="caption"
                            color="text.secondary"
                            sx={{ display: "block" }}
                          >
                            {`Perc ${fmtPercent(step.perceptibility, {
                              decimals: 2,
                              inputIsFraction: true,
                            })} · Prox ${fmtPercent(step.proximity, {
                              decimals: 2,
                              inputIsFraction: true,
                            })} · Involvement ${fmtPercent(
                              step.involvement ?? 0,
                              true
                            )}`}
                          </Typography>
                        )}
                      </Box>
                    );
                  })}
                </Stack>
              ) : (
                <Typography variant="body2" color="text.secondary">
                  Path insights unlock once you run replay on a few deal journeys.
                </Typography>
              )}
            </CardContent>
          </Card>

          <Card variant="outlined">
            <CardHeader
              title="Section F · Global Belief Patterns"
              subheader="Systemic blockers and opportunities surfaced across accounts."
            />
            <CardContent>
              <Stack spacing={1.5}>
                {[globalBarrier, hiddenBlocker, missedOpportunity].map(
                  (insight, idx) => {
                    const entityType = insight.entity_type;
                    const entityLabel =
                      entityType === "concern"
                        ? "Concern"
                        : entityType === "persona"
                        ? "Persona"
                        : null;
                    const entityTooltip =
                      insight.label || insight.persona_id || null;
                    const supportProps = getInsightSupportProps(insight.support);
                    const textColor = supportProps.lowSignal
                      ? "text.secondary"
                      : "text.primary";
                    return (
                      <Stack
                        key={`global-pattern-${idx}`}
                        direction="row"
                        spacing={1}
                        alignItems="center"
                      >
                        <Box sx={{ flex: 1 }}>
                          <Typography variant="body2" color={textColor}>
                            {insight.text}
                          </Typography>
                          {supportProps.summary && (
                            <Typography variant="caption" color="text.secondary">
                              {supportProps.summary}
                            </Typography>
                          )}
                        </Box>
                        {supportProps.lowSignal && (
                          <Chip
                            size="small"
                            variant="outlined"
                            label="Low signal"
                          />
                        )}
                        {entityLabel ? (
                          <Tooltip title={entityTooltip || entityLabel}>
                            <Chip size="small" variant="outlined" label={entityLabel} />
                          </Tooltip>
                        ) : null}
                        <ProvenanceChip provenance={insight} />
                      </Stack>
                    );
                  }
                )}
              </Stack>
            </CardContent>
          </Card>
        </Stack>
      )}

      {tab === "inbox" && (
            <Stack spacing={3}>
              {!globalInsights ? (
                <Typography variant="body2" color="text.secondary">
                  Insights populate after you save engagements and run journey analysis.
                </Typography>
              ) : (
                <>
                  <Box
                    display="flex"
                    flexWrap="wrap"
                    gap={4}
                    alignItems="flex-start"
                    mb={2}
                  >
                    <Box>
                      <Typography variant="overline">Accounts observed</Typography>
                      <Typography variant="h5">
                        {fmtCount(globalInsights.meta.num_accounts)}
                      </Typography>
                    </Box>
                    <Box>
                      <Typography variant="overline">Engagements analyzed</Typography>
                      <Typography variant="h5">
                        {fmtCount(globalInsights.meta.num_engagements)}
                      </Typography>
                    </Box>
                    <Box>
                      <Typography variant="overline">Persona actions</Typography>
                      <Typography variant="h5">
                        {fmtCount(globalInsights.meta.num_persona_recommendations)}
                      </Typography>
                    </Box>
                    <Box>
                      <Typography variant="overline">Edge adjustments</Typography>
                      <Typography variant="h5">
                        {fmtCount(globalInsights.meta.num_edge_recommendations)}
                      </Typography>
                    </Box>
                  </Box>

                  <Card variant="outlined">
                    <CardHeader title="Persona recommendations" />
                    <CardContent>
                      {!personaRecsSorted.length ? (
                        <Typography variant="body2" color="text.secondary">
                          No persona recommendations yet. Save more engagements to surface
                          learning opportunities.
                        </Typography>
                      ) : (
                        <Stack spacing={2}>
                          {personaHighDisplay.length > 0 ? (
                            personaHighDisplay.map((rec) => renderPersonaCard(rec))
                          ) : (
                            <Typography variant="body2">
                              High-confidence personas will appear when journeys gather
                              enough consistent signals.
                            </Typography>
                          )}
                          {personaHighHiddenCount > 0 && (
                            <Box>
                              <Button
                                size="small"
                                variant="text"
                                onClick={() =>
                                  setShowAllHighPersonas((prev) => !prev)
                                }
                              >
                                {showAllHighPersonas
                                  ? "Show fewer high-signal insights"
                                  : `Show ${personaHighHiddenCount} more high-signal insights`}
                              </Button>
                            </Box>
                          )}
                          {personaMediumCount > 0 && (
                            <Box>
                              <Button
                                size="small"
                                variant="text"
                                onClick={() =>
                                  setShowMediumSignalPersonas((prev) => !prev)
                                }
                              >
                                {showMediumSignalPersonas
                                  ? "Hide medium signal insights"
                                  : `Show medium signal insights (${personaMediumCount})`}
                              </Button>
                            </Box>
                          )}
                          {showMediumSignalPersonas &&
                            personaMediumDisplay.map((rec) => renderPersonaCard(rec))}
                          {personaLowCount > 0 && (
                            <Box>
                              <Button
                                size="small"
                                variant="text"
                                onClick={() =>
                                  setShowExploratoryPersonas((prev) => !prev)
                                }
                              >
                                {showExploratoryPersonas
                                  ? "Hide exploratory insights"
                                  : `Show exploratory insights (${personaLowCount})`}
                              </Button>
                            </Box>
                          )}
                          {showExploratoryPersonas &&
                            personaLowDisplay.map((rec) => renderPersonaCard(rec))}
                        </Stack>
                      )}
                    </CardContent>
                  </Card>

                  <Card variant="outlined">
                    <CardHeader
                      title="Edge recommendations"
                      action={
                        edgeInsights.totalCount > 0 ? (
                          <Button
                            size="small"
                            onClick={() => setShowAllEdgeRecs((prev) => !prev)}
                          >
                            {showAllEdgeRecs
                              ? "Show priority only"
                              : edgeInsights.lowSignalCount > 0
                              ? `Show exploratory edges (${edgeInsights.lowSignalCount} hidden)`
                              : "Show all"}
                          </Button>
                        ) : null
                      }
                    />
                    <CardContent>
                      {edgeInsights.totalCount === 0 ? (
                        <Typography variant="body2" color="text.secondary">
                          No edge recommendations available yet.
                        </Typography>
                      ) : (
                        <TableContainer component={Paper} variant="outlined">
                          <Table size="small">
                            <TableHead>
                              <TableRow>
                                <TableCell>Edge</TableCell>
                                <TableCell align="right">Journeys</TableCell>
                                <TableCell align="right">Current Lik.</TableCell>
                                <TableCell align="right">Proposed Lik.</TableCell>
                                <TableCell align="right">Δ Lik.</TableCell>
                                <TableCell align="right">Confidence</TableCell>
                                <TableCell align="center">Action</TableCell>
                              </TableRow>
                            </TableHead>
                            <TableBody>
                              {edgeInsights.display.map((edge) => {
                                const edgeLabel =
                                  (edge.pair_labels || [])
                                    .flat()
                                    .filter(Boolean)
                                    .join(" → ") ||
                                  `${edge.source_id} → ${edge.target_id}`;
                                const journeys =
                                  edge.count ??
                                  edge.recommendation_events ??
                                  edge.seen ??
                                  0;
                                const accounts = edge.accounts ?? 0;
                                const currentLikelihood =
                                  edge.current_likelihood ??
                                  edge.current_relevance ??
                                  0;
                                const proposedLikelihood =
                                  edge.proposed_likelihood ??
                                  edge.recommended_likelihood ??
                                  0;
                                const deltaLikelihood =
                                  proposedLikelihood - currentLikelihood;
                                const deltaLabel =
                                  deltaLikelihood === 0
                                    ? "0pp"
                                    : `${deltaLikelihood >= 0 ? "+" : ""}${(
                                        deltaLikelihood * 100
                                      ).toFixed(1)}pp`;
                                const confidence =
                                  edge.confidence ?? edge.avg_confidence ?? 0;
                                const { label: signalLabelText, color: signalColor } =
                                  signalBadgeProps(confidence);
                                const reasonText =
                                  edge.reason ||
                                  (edge.reasons?.length
                                    ? edge.reasons.slice(0, 2).join(" • ")
                                    : null);
                                const accountText = accounts
                                  ? `${fmtCount(accounts)} accounts`
                                  : null;
                                const ciText =
                                  edge.ci_low != null && edge.ci_high != null
                                    ? `CI ${fmtPercent(edge.ci_low, {
                                        decimals: 0,
                                        inputIsFraction: true,
                                      })}–${fmtPercent(edge.ci_high, {
                                        decimals: 0,
                                        inputIsFraction: true,
                                      })}`
                                    : null;
                                return (
                                  <TableRow
                                    key={`${edge.source_id}-${edge.target_id}`}
                                  >
                                    <TableCell>
                                      <Typography
                                        variant="body2"
                                        fontWeight={600}
                                        gutterBottom
                                      >
                                        {edgeLabel}
                                      </Typography>
                                      {reasonText && (
                                        <Typography
                                          variant="caption"
                                          color="text.secondary"
                                          display="block"
                                        >
                                          {reasonText}
                                        </Typography>
                                      )}
                                      {accountText && (
                                        <Typography
                                          variant="caption"
                                          color="text.secondary"
                                          display="block"
                                        >
                                          {accountText}
                                        </Typography>
                                      )}
                                      {ciText && (
                                        <Typography
                                          variant="caption"
                                          color="text.secondary"
                                          display="block"
                                        >
                                          {ciText}
                                        </Typography>
                                      )}
                                    </TableCell>
                                    <TableCell align="right">
                                      {fmtCount(journeys)}
                                    </TableCell>
                                    <TableCell align="right">
                                      {fmtPercent(currentLikelihood, {
                                        inputIsFraction: true,
                                        decimals: 0,
                                      })}
                                    </TableCell>
                                    <TableCell align="right">
                                      {fmtPercent(proposedLikelihood, {
                                        inputIsFraction: true,
                                        decimals: 0,
                                      })}
                                    </TableCell>
                                    <TableCell align="right">
                                      {deltaLabel}
                                    </TableCell>
                                    <TableCell align="right">
                                      <Chip
                                        size="small"
                                        variant="outlined"
                                        label={signalLabelText}
                                        color={signalColor}
                                      />
                                    </TableCell>
                                    <TableCell align="center">
                                      <Button
                                        size="small"
                                        variant="outlined"
                                        onClick={() =>
                                          handleApplyRecommendation("edge", edge)
                                        }
                                      >
                                        Apply
                                      </Button>
                                    </TableCell>
                                  </TableRow>
                                );
                              })}
                            </TableBody>
                          </Table>
                        </TableContainer>
                      )}
                    </CardContent>
                  </Card>

                  <Card variant="outlined">
                    <CardHeader title="Arsenal impact" />
                    <CardContent>
                      {arsenalImpact.length === 0 ? (
                        <Typography variant="body2" color="text.secondary">
                          No arsenal-level impact recorded yet. Once engagements are logged
                          with assets, we’ll estimate belief shifts per playbook.
                        </Typography>
                      ) : (
                        <TableContainer component={Paper} variant="outlined">
                          <Table size="small">
                            <TableHead>
                              <TableRow>
                                <TableCell>Asset</TableCell>
                                <TableCell>Funnel stage</TableCell>
                                <TableCell align="right">Events</TableCell>
                                <TableCell>Effect</TableCell>
                                <TableCell align="right">Confidence</TableCell>
                                <TableCell>Personas</TableCell>
                                <TableCell>Channels</TableCell>
                              </TableRow>
                            </TableHead>
                            <TableBody>
                              {arsenalImpact.map((row) => {
                                const assetLabel = row.asset_label || row.asset_id;
                                const stageLabel =
                                  row.funnel_stage?.label ||
                                  (typeof row.funnel_stage?.code === "string"
                                    ? titleCase(row.funnel_stage.code)
                                    : "—");
                                const events = row.events ?? row.num_engagements;
                                const confidence =
                                  row.confidence ?? row.avg_confidence ?? null;
                                const { label: signalLabelText, color: signalColor } =
                                  signalBadgeProps(confidence);
                                const effect = formatArsenalEffect(row);
                                const personaCandidates =
                                  row.personas_impacted?.length
                                    ? row.personas_impacted
                                    : row.persona_labels?.length
                                    ? row.persona_labels
                                    : (row.persona_ids || []).map((pid) =>
                                        personaLabelFromId(pid)
                                      );
                                const personaText = personaCandidates.length
                                  ? personaCandidates.slice(0, 3).join(", ")
                                  : "—";
                                const channelCandidates =
                                  row.channels && row.channels.length
                                    ? row.channels
                                    : row.account_meta || [];
                                const channelText = channelCandidates.length
                                  ? channelCandidates.join(", ")
                                  : "—";
                                return (
                                  <TableRow key={row.asset_id}>
                                    <TableCell>
                                      <Typography
                                        variant="body2"
                                        fontWeight={600}
                                      >
                                        {assetLabel}
                                      </Typography>
                                      {row.sample_accounts ? (
                                        <Typography
                                          variant="caption"
                                          color="text.secondary"
                                        >
                                          Sample accounts:{" "}
                                          {fmtCount(row.sample_accounts)}
                                        </Typography>
                                      ) : null}
                                    </TableCell>
                                    <TableCell>
                                      <Typography variant="body2">
                                        {stageLabel}
                                      </Typography>
                                    </TableCell>
                                    <TableCell align="right">
                                      {events ? fmtCount(events) : "—"}
                                    </TableCell>
                                    <TableCell>
                                      <Typography variant="body2" fontWeight={600}>
                                        {effect.label}
                                      </Typography>
                                      <Typography
                                        variant="caption"
                                        color="text.secondary"
                                      >
                                        {effect.detail}
                                      </Typography>
                                    </TableCell>
                                    <TableCell align="right">
                                      <Chip
                                        size="small"
                                        variant="outlined"
                                        label={signalLabelText}
                                        color={signalColor}
                                      />
                                    </TableCell>
                                    <TableCell>
                                      <Typography variant="body2">
                                        {personaText}
                                      </Typography>
                                    </TableCell>
                                    <TableCell>
                                      <Typography
                                        variant="body2"
                                        color="text.secondary"
                                      >
                                        {channelText}
                                      </Typography>
                                    </TableCell>
                                  </TableRow>
                                );
                              })}
                            </TableBody>
                          </Table>
                        </TableContainer>
                      )}
                    </CardContent>
                  </Card>
                </>
              )}
            </Stack>
          )}
    </Box>
  );
};

export default InsightsInbox;
