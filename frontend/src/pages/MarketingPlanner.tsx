import React, { useEffect, useMemo, useState } from "react";
import {
  Alert,
  Box,
  Card,
  CardContent,
  Chip,
  CircularProgress,
  Divider,
  FormControl,
  Grid,
  InputLabel,
  MenuItem,
  Paper,
  Select,
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

// ---------------------------------------------------------------------------
// Types matching backend payload
// ---------------------------------------------------------------------------

type PersonaSummary = {
  id: string;
  order: number;
  label: string;
  title?: string;
  department?: string;
  seniority?: string;
  perceptibility?: number | null;
  proximity?: number | null;
  involvement?: number | null;
  path_probability?: number | null;
};

type PersonaPath = {
  id: string;
  probability?: number | null;
  score?: number | null;
  is_primary?: boolean;
  personas: PersonaSummary[];
};

type JourneyStep = {
  t?: number;
  bucket?: string;
  predicted: string[];
  observed?: string | null;
  win_likelihood?: number | null;
  hit_at_1?: boolean;
  hit_at_3?: boolean;
};

type Transition = {
  persona: PersonaSummary;
  stage_index: number;
  job?: { id: string; label: string };
  pains?: Array<{ id: string; label: string }>;
  capabilities?: Array<{ id: string; label: string }>;
  belief_transition?: {
    persona?: NodeRef;
    problem?: NodeRef;
    execution?: NodeRef;
    pain?: NodeRef | null;
    resolution?: NodeRef | null;
    narrative?: string;
    from_to?: BeliefEdge[];
  };
};

type AssetRec = {
  id?: string;
  name?: string;
  format?: string;
  time_to_effect_days?: number;
};

type ChannelRec = {
  id?: string;
  name?: string;
  type?: string;
  reach_score?: number;
  cadence_hint?: string;
};

type NodeRef = {
  type?: string;
  id?: string;
  label?: string;
};

type BeliefEdge = {
  from_type?: string | null;
  from_id?: string | null;
  from_label?: string | null;
  to_type?: string | null;
  to_id?: string | null;
  to_label?: string | null;
};

type BeliefTransitionDetail = {
  persona?: NodeRef;
  problem?: NodeRef;
  execution?: NodeRef;
  pain?: NodeRef | null;
  resolution?: NodeRef | null;
  narrative?: string;
  from_to?: BeliefEdge[];
};

type CampaignAssetChannel = ChannelRec & {
  engagement_score?: number | null;
  expected_delta_bp?: number;
  mode?: string | null;
  modes?: string[];
  exploration_weight?: number | null;
};

type CampaignAssetRow = {
  asset: AssetRec;
  asset_fit_score?: number | null;
  expected_delta_bp: number;
  channels: CampaignAssetChannel[];
};

type Play = {
  persona_id: string;
  persona_label: string;
  stage_index: number;
  belief_transition?: Transition["belief_transition"];
  asset: AssetRec;
  channel: ChannelRec;
  expected_delta_bp: number;
  expected_delta_pct: number;
  confidence: number;
  asset_score?: number;
  channel_score?: number;
  start_day?: number;
  end_day?: number;
  quarter?: string;
  theme?: string;
  mode?: "broad" | "focused";
  exploration_weight?: number;
  path_probability?: number;
  path_id?: string;
};

type Campaign = {
  quarter: string;
  theme: string;
  focus_personas: string[];
  focus_pains: string[];
  start_day: number;
  end_day: number;
  total_delta_bp: number;
  confidence: number;
  plays: Play[];
  mode_mix: { broad: number; focused: number };
  belief_transitions?: BeliefTransitionDetail[];
  asset_table?: CampaignAssetRow[];
};

type RandomizationPolicy = {
  path_allocation?: Array<{
    path_id: string;
    probability: number;
    exploration_weight: number;
    persona_ids: string[];
  }>;
  belief_spread?: {
    early_cycle_avg?: number;
    late_cycle_avg?: number;
  };
  notes?: string;
};

type AccountPlan = {
  account_id: string;
  account_name: string;
  deal_status?: string;
  meta?: Record<string, any>;
  prediction: {
    persona_paths: PersonaPath[];
    fit: Record<string, any>;
    expected_next?: any;
    observed_personas: string[];
    journey: {
      steps: JourneyStep[];
      total_steps: number;
    };
  };
  scatter?: { personas: PersonaSummary[] };
  transitions?: Transition[];
  execution?: { plays: Play[] };
  campaigns?: Campaign[];
  randomization?: RandomizationPolicy;
  learning?: any;
  error?: string;
};

type PortfolioTheme = {
  quarter: string;
  theme: string;
  accounts: string[];
  focus_personas: string[];
  focus_pains: string[];
  total_delta_bp: number;
  avg_confidence: number;
  play_count: number;
};

type PortfolioPlan = {
  campaign_themes: PortfolioTheme[];
  scatter: Array<{
    label?: string;
    perceptibility?: number | null;
    proximity?: number | null;
    involvement?: number | null;
  }>;
  broad_focus_mix: {
    broad: number;
    focused: number;
    broad_ratio: number;
  };
  randomization: {
    avg_path_probability: number;
    path_count_sampled: number;
  };
};

type PlanSummary = {
  account_count: number;
  avg_best_path_probability: number;
  avg_accuracy: number;
  top_personas: Array<{ label: string; count: number }>;
  top_pains: Array<{ label: string; count: number }>;
  top_capabilities: Array<{ label: string; count: number }>;
  total_expected_delta_bp: number;
};

type MarketingPlan = {
  product_id: string;
  generated_at: string;
  summary: PlanSummary;
  accounts: AccountPlan[];
  portfolio_plan: PortfolioPlan;
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const fmtPercent = (value?: number | null) => {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return `${Math.round(value * 100)}%`;
};

const fmtNumber = (value?: number | null, digits = 2) => {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return Number(value).toFixed(digits);
};

const fmtBasisPoints = (value?: number | null) => {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return `${Math.round(value)} bps`;
};

const toWeeks = (days?: number) => {
  if (days === null || days === undefined || Number.isNaN(days)) return "—";
  return `${Math.round(days / 7)}w`;
};

const renderChipList = (
  items: Array<{ label: string; count: number }>,
  placeholder = "—"
) => {
  if (!items.length) {
    return (
      <Typography variant="body2" color="text.secondary">
        {placeholder}
      </Typography>
    );
  }
  return (
    <Stack direction="row" spacing={1} sx={{ flexWrap: "wrap" }}>
      {items.map((item) => (
        <Chip
          key={`${item.label}-${item.count}`}
          label={`${item.label} (${item.count})`}
          size="small"
          variant="outlined"
        />
      ))}
    </Stack>
  );
};

const uniqueStrings = (values: Array<string | undefined | null>) =>
  Array.from(new Set(values.filter(Boolean) as string[]));

const topFromMap = (map: Map<string, number>, limit = 6) =>
  Array.from(map.entries())
    .sort((a, b) => b[1] - a[1])
    .slice(0, limit)
    .map(([label, count]) => ({ label, count }));

const beliefLabel = (transition?: Transition["belief_transition"]) =>
  transition?.pain?.label ||
  transition?.problem?.label ||
  transition?.resolution?.label ||
  "Belief Shift";

const personaTooltip = (persona: PersonaSummary) => {
  const pieces = [
    persona.perceptibility !== null && persona.perceptibility !== undefined
      ? `Perceptibility: ${fmtNumber(persona.perceptibility)}`
      : null,
    persona.proximity !== null && persona.proximity !== undefined
      ? `Proximity: ${fmtNumber(persona.proximity)}`
      : null,
    persona.involvement !== null && persona.involvement !== undefined
      ? `Involvement: ${fmtNumber(persona.involvement)}`
      : null,
    persona.path_probability !== null && persona.path_probability !== undefined
      ? `Path probability: ${fmtPercent(persona.path_probability)}`
      : null,
  ].filter(Boolean);
  return pieces.join(" · ") || persona.label;
};

const SummaryMetric: React.FC<{ label: string; value: string | number }> = ({
  label,
  value,
}) => (
  <Box sx={{ minWidth: 120 }}>
    <Typography variant="caption" color="text.secondary">
      {label}
    </Typography>
    <Typography variant="h6">{value}</Typography>
  </Box>
);

type CampaignRow = {
  id: string;
  accountId: string;
  accountName: string;
  quarter: string;
  theme: string;
  startDay: number;
  endDay: number;
  totalDelta: number;
  confidence: number;
  personas: string[];
  beliefs: string[];
  beliefTransitions: BeliefTransitionDetail[];
  assetTable: CampaignAssetRow[];
};

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

const MarketingPlanner: React.FC = () => {
  const token =
    typeof window !== "undefined" ? localStorage.getItem("token") : null;

  const [products, setProducts] = useState<Array<{ id: string; name: string }>>(
    []
  );
  const [selectedProductId, setSelectedProductId] = useState<string>("");
  const [accounts, setAccounts] = useState<Array<{ id: string; name: string }>>(
    []
  );
  const [selectedAccountId, setSelectedAccountId] = useState<string>("");
  const [plan, setPlan] = useState<MarketingPlan | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [statusMsg, setStatusMsg] = useState<string>("");
  const [activeTab, setActiveTab] = useState<"execution" | "thesis">("execution");
  const [sortField, setSortField] = useState<
    "timeline" | "theme" | "bps" | "confidence"
  >("timeline");
  const [sortDirection, setSortDirection] = useState<"asc" | "desc">("asc");
  const [filterQuarter, setFilterQuarter] = useState<string>("");
  const [filterTheme, setFilterTheme] = useState<string>("");
  const [filterAccount, setFilterAccount] = useState<string>("");

  useEffect(() => {
    if (token) {
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
            setStatusMsg("No products found. Please run Value Prop first.");
            return;
          }
          setProducts(list);
          setSelectedProductId((prev) => prev || list[0].id);
        } catch (err: any) {
          console.error(err);
          setError(err.message || "Failed to load company or products.");
        }
      })();
    }
  }, [token]);

  useEffect(() => {
    if (!selectedProductId || !token) return;
    (async () => {
      try {
        const res = await fetch(
          `http://localhost:8000/get-accounts-for-rcs/${selectedProductId}`,
          { headers: { Authorization: `Bearer ${token}` } }
        );
        if (!res.ok) throw new Error(await res.text());
        const data = await res.json();
        const list = (data.accounts ?? []).map((a: any) => ({
          id: a.id,
          name: a.account_name || a.name || a.id,
        }));
        setAccounts(list);
        setSelectedAccountId("");
      } catch (err) {
        console.error(err);
        setAccounts([]);
      }
    })();
  }, [selectedProductId, token]);

  useEffect(() => {
    if (!selectedProductId || !token) return;
    setLoading(true);
    setError(null);
    (async () => {
      try {
        const query = selectedAccountId
          ? `?account_id=${encodeURIComponent(selectedAccountId)}`
          : "";
        const res = await fetch(
          `http://localhost:8000/get-comprehensive-execution-plan/${selectedProductId}${query}`,
          { headers: { Authorization: `Bearer ${token}` } }
        );
        if (!res.ok) throw new Error(await res.text());
        const data: MarketingPlan = await res.json();
        setPlan(data);
        setStatusMsg(
          `Marketing blueprint generated ${new Date(
            data.generated_at
          ).toLocaleString()}`
        );
      } catch (err: any) {
        console.error(err);
        setError(err.message || "Failed to build marketing plan.");
        setPlan(null);
      } finally {
        setLoading(false);
      }
    })();
  }, [selectedProductId, selectedAccountId, token]);

  useEffect(() => {
    setActiveTab("execution");
  }, [selectedAccountId]);

  const selectedAccountPlan = useMemo(
    () =>
      plan && selectedAccountId
        ? plan.accounts.find((a) => a.account_id === selectedAccountId) || null
        : null,
  [plan, selectedAccountId]
);

  const campaignRows = useMemo<CampaignRow[]>(() => {
    if (!plan) return [];
    const rows: CampaignRow[] = [];
    plan.accounts.forEach((account) => {
      (account.campaigns || []).forEach((campaign, campaignIdx) => {
        const beliefTransitions = campaign.belief_transitions ?? [];
        const personaNames = uniqueStrings([
          ...(campaign.focus_personas || []),
          ...beliefTransitions
            .map((bt) => bt.persona?.label)
            .filter((label): label is string => Boolean(label)),
        ]);
        const beliefNames = uniqueStrings([
          ...(campaign.focus_pains || []),
          ...beliefTransitions
            .map((bt) => bt.pain?.label || bt.problem?.label || bt.resolution?.label)
            .filter((label): label is string => Boolean(label)),
        ]);
        const assetTable = (campaign.asset_table ?? []).map((assetRow) => ({
          ...assetRow,
          channels: assetRow.channels ?? [],
        }));
        rows.push({
          id: `${account.account_id}-${campaign.quarter}-${campaign.theme}-${campaignIdx}`,
          accountId: account.account_id,
          accountName: account.account_name || account.account_id,
          quarter: campaign.quarter,
          theme: campaign.theme,
          startDay: campaign.start_day,
          endDay: campaign.end_day,
          totalDelta: campaign.total_delta_bp,
          confidence: campaign.confidence,
          personas: personaNames,
          beliefs: beliefNames,
          beliefTransitions,
          assetTable,
        });
      });
    });
    return rows;
  }, [plan]);

  const uniqueQuarters = useMemo(
    () =>
      Array.from(new Set(campaignRows.map((row) => row.quarter))).sort(
        (a, b) => a.localeCompare(b)
      ),
    [campaignRows]
  );

  const uniqueThemes = useMemo(
    () =>
      Array.from(new Set(campaignRows.map((row) => row.theme))).sort((a, b) =>
        a.localeCompare(b)
      ),
    [campaignRows]
  );

  const uniqueAccounts = useMemo(
    () => {
      const map = new Map<string, string>();
      campaignRows.forEach((row) => {
        if (row.accountId) {
          map.set(row.accountId, row.accountName);
        }
      });
      return Array.from(map.entries())
        .map(([id, name]) => ({ id, name }))
        .sort((a, b) => a.name.localeCompare(b.name));
    },
    [campaignRows]
  );

  useEffect(() => {
    if (selectedAccountPlan) {
      setFilterAccount(selectedAccountPlan.account_id);
    } else {
      setFilterAccount("");
    }
  }, [selectedAccountPlan]);

  const filteredRows = useMemo(() => {
    return campaignRows.filter((row) => {
      if (selectedAccountPlan && row.accountId !== selectedAccountPlan.account_id) {
        return false;
      }
      if (!selectedAccountPlan && filterAccount && row.accountId !== filterAccount) {
        return false;
      }
      if (filterQuarter && row.quarter !== filterQuarter) {
        return false;
      }
      if (filterTheme && row.theme !== filterTheme) {
        return false;
      }
      return true;
    });
  }, [
    campaignRows,
    filterAccount,
    filterQuarter,
    filterTheme,
    selectedAccountPlan,
  ]);

  const sortedFilteredRows = useMemo(() => {
    const rows = [...filteredRows];
    rows.sort((a, b) => {
      let cmp = 0;
      switch (sortField) {
        case "timeline":
          cmp =
            a.startDay - b.startDay ||
            a.endDay - b.endDay ||
            a.theme.localeCompare(b.theme);
          break;
        case "theme":
          cmp = a.theme.localeCompare(b.theme);
          break;
        case "bps":
          cmp = (a.totalDelta || 0) - (b.totalDelta || 0);
          break;
        case "confidence":
          cmp = (a.confidence || 0) - (b.confidence || 0);
          break;
        default:
          cmp = 0;
      }
      return sortDirection === "asc" ? cmp : -cmp;
    });
    return rows;
  }, [filteredRows, sortField, sortDirection]);

  const accountSummary = useMemo(() => {
    if (!selectedAccountPlan) return null;
    const personaCounts = new Map<string, number>();
    const painCounts = new Map<string, number>();
    const capabilityCounts = new Map<string, number>();

    selectedAccountPlan.transitions?.forEach((transition) => {
      if (transition.persona?.label) {
        personaCounts.set(
          transition.persona.label,
          (personaCounts.get(transition.persona.label) || 0) + 1
        );
      }
      transition.pains?.forEach((pain) => {
        if (!pain.label) return;
        painCounts.set(pain.label, (painCounts.get(pain.label) || 0) + 1);
      });
      transition.capabilities?.forEach((cap) => {
        if (!cap.label) return;
        capabilityCounts.set(
          cap.label,
          (capabilityCounts.get(cap.label) || 0) + 1
        );
      });
    });

    const totalDelta =
      selectedAccountPlan.campaigns?.reduce(
        (sum, campaign) => sum + (campaign.total_delta_bp || 0),
        0
      ) ?? 0;

    return {
      personas: topFromMap(personaCounts),
      pains: topFromMap(painCounts),
      capabilities: topFromMap(capabilityCounts),
      bestPathProbability:
        selectedAccountPlan.prediction.persona_paths?.[0]?.probability ?? null,
      accuracy: selectedAccountPlan.prediction.fit?.accuracy ?? null,
      totalDelta,
    };
  }, [selectedAccountPlan]);

  const accountMix = useMemo(() => {
    if (!selectedAccountPlan?.execution?.plays?.length) return null;
    let broad = 0;
    let focused = 0;
    selectedAccountPlan.execution.plays.forEach((play) => {
      if (play.mode === "broad") broad += 1;
      else focused += 1;
    });
    const total = broad + focused;
    return {
      broad,
      focused,
      ratio: total ? broad / total : 0,
    };
  }, [selectedAccountPlan]);

  if (loading) {
    return (
      <Box sx={{ display: "flex", justifyContent: "center", mt: 8 }}>
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Card variant="outlined" sx={{ p: 2, bgcolor: "#ffebee" }}>
        <Typography color="error" sx={{ whiteSpace: "pre-wrap" }}>
          {error}
        </Typography>
      </Card>
    );
  }

  if (!plan) {
    return (
      <Typography variant="body1" color="text.secondary" sx={{ mt: 4 }}>
        No marketing plan available.
      </Typography>
    );
  }

  const summaryTitle = selectedAccountPlan ? "Account Summary" : "Portfolio Summary";
  const personaItems = selectedAccountPlan
    ? accountSummary?.personas ?? []
    : plan.summary.top_personas;
  const painItems = selectedAccountPlan
    ? accountSummary?.pains ?? []
    : plan.summary.top_pains;
  const capabilityItems = selectedAccountPlan
    ? accountSummary?.capabilities ?? []
    : plan.summary.top_capabilities;

  const bestPathValue = selectedAccountPlan
    ? fmtPercent(accountSummary?.bestPathProbability ?? null)
    : fmtPercent(plan.summary.avg_best_path_probability);
  const accuracyValue = selectedAccountPlan
    ? fmtPercent(accountSummary?.accuracy ?? null)
    : fmtPercent(plan.summary.avg_accuracy);
  const liftValue = selectedAccountPlan
    ? fmtBasisPoints(accountSummary?.totalDelta ?? null)
    : fmtBasisPoints(plan.summary.total_expected_delta_bp);

  const describeBeliefTransition = (bt: BeliefTransitionDetail): string => {
    if (bt.narrative) return bt.narrative;
    const persona = bt.persona?.label || "Target persona";
    const pain = bt.pain?.label;
    const job = bt.problem?.label;
    const resolution = bt.resolution?.label;
    let text = `Get ${persona}`;
    if (pain) {
      text += ` to realise ${pain}`;
    } else {
      text += " to internalize the required belief";
    }
    if (job) {
      text += ` when they ${job}`;
    }
    if (resolution) {
      text += ` and reinforce with ${resolution}`;
    }
    return text;
  };

  const summarizeList = (items: string[], limit = 3) => {
    if (!items.length) return "";
    const unique = Array.from(new Set(items));
    if (unique.length <= limit) return unique.join(", ");
    const head = unique.slice(0, limit).join(", ");
    return `${head} (+${unique.length - limit} more)`;
  };

  const describeCampaignSummary = (row: CampaignRow): string => {
    const personaSummary = summarizeList(row.personas);
    const beliefSummary = summarizeList(row.beliefs);
    const timelineLabel = `${toWeeks(row.startDay)} → ${toWeeks(row.endDay)}`;
    let text = `By ${row.quarter} (${timelineLabel}) we expect ~${fmtNumber(
      row.totalDelta
    )} bps lift at ${fmtPercent(row.confidence)} confidence.`;
    if (personaSummary) {
      text += ` Personas: ${personaSummary}.`;
    }
    if (beliefSummary) {
      text += ` Belief shifts: ${beliefSummary}.`;
    }
    text += ` Campaign theme ${row.theme} across ${row.accountName}.`;
    return text;
  };

  const renderExecutionContent = () => {
    if (!campaignRows.length) {
      return (
        <Typography variant="body2" color="text.secondary">
          No campaigns generated yet.
        </Typography>
      );
    }

    const hasRows = sortedFilteredRows.length > 0;

    return (
      <Stack spacing={2}>
        <Paper
          variant="outlined"
          sx={{
            p: 1.5,
            display: "flex",
            flexWrap: "wrap",
            gap: 1.5,
            alignItems: "center",
          }}
        >
          <FormControl size="small" sx={{ minWidth: 160 }}>
            <InputLabel id="filter-quarter-label">Timeline</InputLabel>
            <Select
              labelId="filter-quarter-label"
              label="Timeline"
              value={filterQuarter}
              onChange={(e) => setFilterQuarter(String(e.target.value))}
            >
              <MenuItem value="">All</MenuItem>
              {uniqueQuarters.map((quarter) => (
                <MenuItem key={quarter} value={quarter}>
                  {quarter}
                </MenuItem>
              ))}
            </Select>
          </FormControl>

          <FormControl size="small" sx={{ minWidth: 200 }}>
            <InputLabel id="filter-theme-label">Campaign Theme</InputLabel>
            <Select
              labelId="filter-theme-label"
              label="Campaign Theme"
              value={filterTheme}
              onChange={(e) => setFilterTheme(String(e.target.value))}
            >
              <MenuItem value="">All</MenuItem>
              {uniqueThemes.map((theme) => (
                <MenuItem key={theme} value={theme}>
                  {theme}
                </MenuItem>
              ))}
            </Select>
          </FormControl>

          {!selectedAccountPlan && (
            <FormControl size="small" sx={{ minWidth: 200 }}>
              <InputLabel id="filter-account-label">Target Account</InputLabel>
              <Select
                labelId="filter-account-label"
                label="Target Account"
                value={filterAccount}
                onChange={(e) => setFilterAccount(String(e.target.value))}
              >
                <MenuItem value="">All</MenuItem>
                {uniqueAccounts.map((account) => (
                  <MenuItem key={account.id} value={account.id}>
                    {account.name}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          )}

          <Box sx={{ flexGrow: 1 }} />

          <FormControl size="small" sx={{ minWidth: 160 }}>
            <InputLabel id="sort-field-label">Sort by</InputLabel>
            <Select
              labelId="sort-field-label"
              label="Sort by"
              value={sortField}
              onChange={(e) =>
                setSortField(e.target.value as typeof sortField)
              }
            >
              <MenuItem value="timeline">Timeline</MenuItem>
              <MenuItem value="theme">Campaign Theme</MenuItem>
              <MenuItem value="bps">Δ (bps)</MenuItem>
              <MenuItem value="confidence">Confidence</MenuItem>
            </Select>
          </FormControl>

          <FormControl size="small" sx={{ minWidth: 120 }}>
            <InputLabel id="sort-direction-label">Order</InputLabel>
            <Select
              labelId="sort-direction-label"
              label="Order"
              value={sortDirection}
              onChange={(e) =>
                setSortDirection(e.target.value as typeof sortDirection)
              }
            >
              <MenuItem value="asc">Ascending</MenuItem>
              <MenuItem value="desc">Descending</MenuItem>
            </Select>
          </FormControl>
        </Paper>

        {hasRows ? (
          <Stack spacing={2}>
            {sortedFilteredRows.map((row) => (
              <Paper
                key={row.id}
                variant="outlined"
                sx={{
                  p: 3,
                  display: "flex",
                  flexDirection: "column",
                  gap: 2,
                }}
              >
                <Stack direction="row" spacing={1.5} sx={{ flexWrap: "wrap" }}>
                  <Chip label={row.quarter} size="small" color="primary" />
                  <Chip
                    label={`${toWeeks(row.startDay)} → ${toWeeks(row.endDay)}`}
                    size="small"
                    variant="outlined"
                  />
                  <Chip label={row.theme} size="small" color="default" />
                </Stack>

                <Box>
                  <Typography variant="subtitle1" sx={{ mb: 0.5 }}>
                    {row.accountName}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    {describeCampaignSummary(row)}
                  </Typography>
                </Box>

                <Grid container spacing={2}>
                  <Grid size={12}>
                    <Typography
                      variant="overline"
                      color="text.secondary"
                      sx={{ display: "block" }}
                    >
                      Target Personas
                    </Typography>
                    <Stack direction="row" spacing={0.75} sx={{ flexWrap: "wrap" }}>
                      {row.personas.map((persona) => (
                        <Chip key={persona} label={persona} size="small" />
                      ))}
                    </Stack>
                  </Grid>
                  <Grid size={12}>
                    <Typography
                      variant="overline"
                      color="text.secondary"
                      sx={{ display: "block" }}
                    >
                      Target Belief Shifts
                    </Typography>
                    <Stack direction="row" spacing={0.75} sx={{ flexWrap: "wrap" }}>
                      {row.beliefs.map((belief) => (
                        <Chip key={belief} label={belief} size="small" />
                      ))}
                    </Stack>
                  </Grid>
                  <Grid size={12}>
                    <Typography
                      variant="overline"
                      color="text.secondary"
                      sx={{ display: "block", mb: 0.75 }}
                    >
                      Belief Transitions
                    </Typography>
                    {row.beliefTransitions.length ? (
                      <Stack spacing={0.75}>
                        {row.beliefTransitions.map((bt, idx) => (
                          <Box key={`${row.id}-belief-${idx}`}>
                            <Typography variant="body2">
                              {describeBeliefTransition(bt)}
                            </Typography>
                            {bt.from_to?.length ? (
                              <Typography variant="caption" color="text.secondary">
                                {bt.from_to
                                  .map(
                                    (edge) =>
                                      `${edge.from_label || edge.from_type || "Source"} → ${
                                        edge.to_label || edge.to_type || "Destination"
                                      }`
                                  )
                                  .join(" · ")}
                              </Typography>
                            ) : null}
                          </Box>
                        ))}
                      </Stack>
                    ) : (
                      <Typography variant="body2" color="text.secondary">
                        No belief transitions identified for this campaign.
                      </Typography>
                    )}
                  </Grid>
                </Grid>

                {row.assetTable.length ? (
                  <TableContainer component={Paper} variant="outlined">
                    <Table size="small">
                      <TableHead>
                        <TableRow>
                          <TableCell>Asset</TableCell>
                          <TableCell align="right">Δ (bps)</TableCell>
                          <TableCell align="right">Fit score</TableCell>
                          <TableCell>Channels (top)</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {row.assetTable.map((assetRow, idx) => {
                          const assetLabel =
                            assetRow.asset?.name || assetRow.asset?.id || `Asset ${idx + 1}`;
                          const channelSlice = (assetRow.channels || []).slice(0, 3);
                          return (
                            <TableRow
                              key={
                                assetRow.asset?.id ||
                                assetRow.asset?.name ||
                                `${row.id}-asset-${idx}`
                              }
                            >
                              <TableCell sx={{ maxWidth: 240 }}>
                                <Typography variant="body2">{assetLabel}</Typography>
                                {assetRow.asset?.id && assetRow.asset?.id !== assetLabel && (
                                  <Typography variant="caption" color="text.secondary">
                                    id: {assetRow.asset?.id}
                                  </Typography>
                                )}
                              </TableCell>
                              <TableCell align="right">
                                {fmtBasisPoints(assetRow.expected_delta_bp)}
                              </TableCell>
                              <TableCell align="right">
                                {assetRow.asset_fit_score === null ||
                                assetRow.asset_fit_score === undefined
                                  ? "—"
                                  : fmtNumber(assetRow.asset_fit_score)}
                              </TableCell>
                              <TableCell>
                                <Stack direction="row" spacing={0.75} sx={{ flexWrap: "wrap" }}>
                                  {channelSlice.length === 0 ? (
                                    <Typography variant="caption" color="text.secondary">
                                      —
                                    </Typography>
                                  ) : (
                                    channelSlice.map((channel, cIdx) => {
                                      const label =
                                        channel.name || channel.id || `Channel ${cIdx + 1}`;
                                      const fit =
                                        channel.engagement_score !== null &&
                                        channel.engagement_score !== undefined
                                          ? ` · fit ${fmtNumber(channel.engagement_score)}`
                                          : "";
                                      return (
                                        <Chip
                                          key={`${row.id}-asset-${idx}-channel-${
                                            channel.id || channel.name || cIdx
                                          }`}
                                          label={`${label}${fit}`}
                                          size="small"
                                          variant="outlined"
                                        />
                                      );
                                    })
                                  )}
                                </Stack>
                              </TableCell>
                            </TableRow>
                          );
                        })}
                      </TableBody>
                    </Table>
                  </TableContainer>
                ) : (
                  <Typography variant="body2" color="text.secondary">
                    No arsenal recommendations available yet.
                  </Typography>
                )}

                <Stack direction="row" spacing={3}>
                  <Box>
                    <Typography variant="caption" color="text.secondary">
                      Expected Δ by {row.quarter}
                    </Typography>
                    <Typography variant="h6">
                      {fmtNumber(row.totalDelta)} bps
                    </Typography>
                  </Box>
                  <Box>
                    <Typography variant="caption" color="text.secondary">
                      Confidence
                    </Typography>
                    <Typography variant="h6">
                      {fmtPercent(row.confidence)}
                    </Typography>
                  </Box>
                </Stack>
              </Paper>
            ))}
          </Stack>
        ) : (
          <Paper variant="outlined" sx={{ p: 4 }}>
            <Typography variant="body2" color="text.secondary" align="center">
              No campaigns match the current filters.
            </Typography>
          </Paper>
        )}
      </Stack>
    );
  };

  const renderThesisContent = () => {
    if (!selectedAccountPlan) {
      return (
        <Stack spacing={2}>
          <Card variant="outlined">
            <CardContent>
              <Typography variant="subtitle1" sx={{ mb: 1 }}>
                Persona Scatter (Portfolio)
              </Typography>
              <TableContainer component={Paper} variant="outlined">
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Persona</TableCell>
                      <TableCell align="right">Perceptibility</TableCell>
                      <TableCell align="right">Proximity</TableCell>
                      <TableCell align="right">Involvement</TableCell>
                      <TableCell align="right">Path Probability</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {plan.portfolio_plan.scatter.slice(0, 40).map((row, idx) => (
                      <TableRow key={`${row.id || row.label || idx}`}>
                        <TableCell>{row.label || "Persona"}</TableCell>
                        <TableCell align="right">
                          {fmtNumber(row.perceptibility)}
                        </TableCell>
                        <TableCell align="right">
                          {fmtNumber(row.proximity)}
                        </TableCell>
                        <TableCell align="right">
                          {fmtNumber(row.involvement)}
                        </TableCell>
                        <TableCell align="right">
                          {fmtPercent(
                            row.path_probability === undefined
                              ? null
                              : row.path_probability
                          )}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </CardContent>
          </Card>
          <Card variant="outlined">
            <CardContent>
              <Typography variant="subtitle1" sx={{ mb: 1 }}>
                Explore vs Exploit Strategy (Portfolio)
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Avg path probability:{" "}
                {fmtPercent(plan.portfolio_plan.randomization.avg_path_probability)}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Paths sampled: {plan.portfolio_plan.randomization.path_count_sampled}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Broad engagements: {plan.portfolio_plan.broad_focus_mix.broad} ·
                Focused engagements: {plan.portfolio_plan.broad_focus_mix.focused} ·
                Broad ratio: {fmtPercent(plan.portfolio_plan.broad_focus_mix.broad_ratio)}
              </Typography>
            </CardContent>
          </Card>
        </Stack>
      );
    }

    return (
      <Stack spacing={2}>
        <Card variant="outlined">
          <CardContent>
            <Typography variant="subtitle1" sx={{ mb: 1 }}>
              Persona Paths
            </Typography>
            <Stack spacing={1.5}>
              {selectedAccountPlan.prediction.persona_paths.map((path) => (
                <Box
                  key={path.id}
                  sx={{
                    border: "1px solid",
                    borderColor: path.is_primary ? "primary.main" : "divider",
                    borderRadius: 1,
                    p: 1,
                  }}
                >
                  <Stack direction="row" spacing={1} alignItems="center">
                    <Typography variant="subtitle2">
                      {path.is_primary ? "Primary path" : "Alternate path"}
                    </Typography>
                    <Chip
                      size="small"
                      label={`P=${fmtPercent(path.probability)}`}
                      color={path.is_primary ? "primary" : "default"}
                    />
                    {path.score !== undefined && (
                      <Chip
                        size="small"
                        label={`Score ${fmtNumber(path.score)}`}
                        variant="outlined"
                      />
                    )}
                  </Stack>
                  <Stack direction="row" spacing={1} sx={{ mt: 1, flexWrap: "wrap" }}>
                    {path.personas.map((persona) => (
                      <Tooltip key={persona.id} title={personaTooltip(persona)}>
                        <Chip
                          label={`${persona.order + 1}. ${persona.label}`}
                          size="small"
                          variant={path.is_primary ? "filled" : "outlined"}
                          color={path.is_primary ? "primary" : "default"}
                        />
                      </Tooltip>
                    ))}
                  </Stack>
                </Box>
              ))}
            </Stack>
          </CardContent>
        </Card>

        <Card variant="outlined">
          <CardContent>
            <Typography variant="subtitle1" sx={{ mb: 1 }}>
              Persona Scatter (Proximity × Perceptibility)
            </Typography>
            <TableContainer component={Paper} variant="outlined">
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Persona</TableCell>
                    <TableCell align="right">Perceptibility</TableCell>
                    <TableCell align="right">Proximity</TableCell>
                    <TableCell align="right">Involvement</TableCell>
                    <TableCell align="right">Path Prob.</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {(selectedAccountPlan.scatter?.personas || []).map((persona) => (
                    <TableRow key={`${persona.id}-${persona.order}`}>
                      <TableCell>{persona.label}</TableCell>
                      <TableCell align="right">
                        {fmtNumber(persona.perceptibility)}
                      </TableCell>
                      <TableCell align="right">
                        {fmtNumber(persona.proximity)}
                      </TableCell>
                      <TableCell align="right">
                        {fmtNumber(persona.involvement)}
                      </TableCell>
                      <TableCell align="right">
                        {fmtPercent(persona.path_probability)}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </CardContent>
        </Card>

        <Card variant="outlined">
          <CardContent>
            <Typography variant="subtitle1" sx={{ mb: 1 }}>
              Belief Transitions
            </Typography>
            <Stack spacing={1.5}>
              {(selectedAccountPlan.transitions || []).map((transition, idx) => (
                <Box
                  key={`${transition.persona.id}-${idx}`}
                  sx={{
                    border: "1px solid",
                    borderColor: "divider",
                    borderRadius: 1,
                    p: 1,
                  }}
                >
                  <Typography variant="subtitle2">
                    Stage {transition.stage_index + 1}: {transition.persona.label}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Problem: {transition.belief_transition?.problem || "—"} · Pain:{" "}
                    {transition.belief_transition?.pain || "—"} · Resolution:{" "}
                    {transition.belief_transition?.resolution || "—"}
                  </Typography>
                </Box>
              ))}
            </Stack>
          </CardContent>
        </Card>

        <Card variant="outlined">
          <CardContent>
            <Typography variant="subtitle1" sx={{ mb: 1 }}>
              Explore vs Exploit Strategy
            </Typography>
            {selectedAccountPlan.randomization?.path_allocation?.length ? (
              <TableContainer component={Paper} variant="outlined">
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Path</TableCell>
                      <TableCell align="right">Probability</TableCell>
                      <TableCell align="right">Exploration</TableCell>
                      <TableCell>Personas</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {selectedAccountPlan.randomization.path_allocation.map((row) => (
                      <TableRow key={row.path_id}>
                        <TableCell>{row.path_id}</TableCell>
                        <TableCell align="right">
                          {fmtPercent(row.probability)}
                        </TableCell>
                        <TableCell align="right">
                          {fmtPercent(row.exploration_weight)}
                        </TableCell>
                        <TableCell>{row.persona_ids.join(" → ") || "—"}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            ) : (
              <Typography variant="body2" color="text.secondary">
                No randomization policy available yet.
              </Typography>
            )}
            <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
              Early-cycle exploration:{" "}
              {fmtPercent(
                selectedAccountPlan.randomization?.belief_spread?.early_cycle_avg
              )}{" "}
              · Late-cycle:{" "}
              {fmtPercent(
                selectedAccountPlan.randomization?.belief_spread?.late_cycle_avg
              )}
            </Typography>
            <Typography variant="caption" color="text.secondary">
              {selectedAccountPlan.randomization?.notes || "—"}
            </Typography>
          </CardContent>
        </Card>
      </Stack>
    );
  };

  return (
    <Box sx={{ p: 3 }}>
      <Stack direction="row" spacing={2} alignItems="center" sx={{ mb: 2 }}>
        <Typography variant="h5">Marketing Planner</Typography>

        {products.length > 0 && (
          <FormControl size="small" sx={{ minWidth: 220, ml: "auto" }}>
            <InputLabel id="product-select-label">Product</InputLabel>
            <Select
              labelId="product-select-label"
              label="Product"
              value={selectedProductId}
              onChange={(e) => setSelectedProductId(String(e.target.value))}
            >
              {products.map((product) => (
                <MenuItem key={product.id} value={product.id}>
                  {product.name || product.id}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        )}

        <FormControl size="small" sx={{ minWidth: 260 }}>
          <InputLabel id="account-select-label">View</InputLabel>
          <Select
            labelId="account-select-label"
            label="View"
            value={selectedAccountId}
            onChange={(e) => setSelectedAccountId(String(e.target.value))}
          >
            <MenuItem value="">Portfolio Overview</MenuItem>
            {accounts.map((account) => (
              <MenuItem key={account.id} value={account.id}>
                Plan for {account.name}
              </MenuItem>
            ))}
          </Select>
        </FormControl>
      </Stack>

      {!!statusMsg && (
        <Alert severity="info" sx={{ mb: 2 }}>
          {statusMsg}
        </Alert>
      )}

      <Grid container spacing={2} sx={{ mb: 3 }}>
        <Grid size={{ xs: 12, md: 6 }}>
          <Card variant="outlined">
            <CardContent>
              <Typography variant="subtitle1" sx={{ mb: 1 }}>
                {summaryTitle}
              </Typography>
              <Stack direction="row" spacing={3} sx={{ flexWrap: "wrap" }}>
                {selectedAccountPlan ? (
                  <>
                    <SummaryMetric label="Best Path Probability" value={bestPathValue} />
                    <SummaryMetric label="Prediction Accuracy" value={accuracyValue} />
                    <SummaryMetric label="Expected Lift" value={liftValue} />
                  </>
                ) : (
                  <>
                    <SummaryMetric label="Accounts" value={plan.summary.account_count} />
                    <SummaryMetric label="Avg Best Path Probability" value={bestPathValue} />
                    <SummaryMetric label="Avg Accuracy" value={accuracyValue} />
                    <SummaryMetric label="Total Expected Lift" value={liftValue} />
                  </>
                )}
              </Stack>

              <Divider sx={{ my: 2 }} />

              <Typography variant="subtitle2" sx={{ mb: 1 }}>
                Persona Focus
              </Typography>
              {renderChipList(personaItems)}

              <Typography variant="subtitle2" sx={{ mt: 2, mb: 1 }}>
                Key Pains
              </Typography>
              {renderChipList(painItems)}

              <Typography variant="subtitle2" sx={{ mt: 2, mb: 1 }}>
                Capability Themes
              </Typography>
              {renderChipList(capabilityItems)}
            </CardContent>
          </Card>
        </Grid>
        <Grid size={{ xs: 12, md: 6 }}>
          <Card variant="outlined">
            <CardContent>
              <Typography variant="subtitle1" sx={{ mb: 1 }}>
                {selectedAccountPlan ? "Campaign Mix" : "Portfolio Mix"}
              </Typography>
              {selectedAccountPlan ? (
                accountMix ? (
                  <>
                    <Typography variant="body2" color="text.secondary">
                      Broad engagements: {accountMix.broad} · Focused engagements:{" "}
                      {accountMix.focused} · Broad ratio: {fmtPercent(accountMix.ratio)}
                    </Typography>
                  </>
                ) : (
                  <Typography variant="body2" color="text.secondary">
                    No executions generated yet.
                  </Typography>
                )
              ) : (
                <Typography variant="body2" color="text.secondary">
                  Broad engagements: {plan.portfolio_plan.broad_focus_mix.broad} ·
                  Focused engagements: {plan.portfolio_plan.broad_focus_mix.focused} ·
                  Broad ratio:{" "}
                  {fmtPercent(plan.portfolio_plan.broad_focus_mix.broad_ratio)}
                </Typography>
              )}
              <Divider sx={{ my: 2 }} />
              <Typography variant="subtitle2" sx={{ mb: 1 }}>
                Explore / Exploit Snapshot
              </Typography>
              <Typography variant="body2" color="text.secondary">
                {selectedAccountPlan
                  ? `Paths tracked: ${
                      selectedAccountPlan.randomization?.path_allocation?.length ?? 0
                    }`
                  : `Paths sampled: ${plan.portfolio_plan.randomization.path_count_sampled}`}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      <Box sx={{ borderBottom: 1, borderColor: "divider", mb: 2 }}>
        <Tabs
          value={activeTab}
          onChange={(_, value) => setActiveTab(value)}
          aria-label="marketing planner tabs"
        >
          <Tab label="Execution Blueprint" value="execution" />
          <Tab label="Thesis" value="thesis" />
        </Tabs>
      </Box>

      {activeTab === "execution" ? renderExecutionContent() : renderThesisContent()}
    </Box>
  );
};

export default MarketingPlanner;
