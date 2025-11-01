import React, { useEffect, useMemo, useState } from "react";
import {
  Box,
  Typography,
  Card,
  CardContent,
  Chip,
  Stack,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Table,
  TableHead,
  TableBody,
  TableRow,
  TableCell,
  TableContainer,
  Paper,
  CircularProgress,
  Divider,
  Tooltip,
  Grid,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Alert,
} from "@mui/material";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";

/* =======================
   Types (v2-first)
   ======================= */
interface AssetRef {
  id?: string;
  name?: string;
  format?: string;
  evergreen?: boolean;
}
interface ChannelRef {
  id?: string;
  name?: string;
  type?: string;
  reach?: number;
  reach_score?: number;
}
interface ArsenalRow {
  asset?: AssetRef | string;
  channel?: ChannelRef | string; // single channel
  channels?: string;             // fallback: comma-joined channels string
  fitment?: string;
  engagement?: string;
  expectedLift?: string;
  concernsAddressed?: string[];
  why?: string;
  breadthScore?: number;
}
interface Campaign {
  id: string;
  description: string;
  timeframe?: { startDate?: string; endDate?: string };
  personas?: string[];
  arsenalTable?: ArsenalRow[];
}
interface Theme {
  id: string;
  name: string;
  explanation: string;
  objective: string;
  targetAccounts?: (string | { id: string })[];
  campaigns?: Campaign[];
}
interface Quarter {
  label: string;
  timeframe?: { startDate?: string; endDate?: string };
  successMetrics?: { personasEngaged?: number; expectedBeliefShift?: number };
  campaigns: Campaign[];
}
interface Plan {
  meta?: { version?: string; generatedAt?: string; beliefScale?: string[]; debug?: Record<string, any> };
  portfolio?: {
    keyStats?: {
      totalTargetAccounts?: number;
      totalPersonasToEngage?: number;
      expectedWinsPct?: number; // 0..1
      averageAccountBelief?: string | number;
      timeToWinMonths?: number;
    };
  };
  themes?: Theme[];
  quarters?: Quarter[];
  // tolerate backend variants
  phases?: Array<{ name?: string; label?: string; explanation?: string; objective?: string; campaigns?: Campaign[] }>;
  execution_plan?: { campaigns?: Campaign[] };
}

/* =======================
   Helpers
   ======================= */
const fmt = (v?: string | number | null) => (v === 0 ? "0" : v ? String(v) : "—");
const dateRange = (s?: string, e?: string) => `${fmt(s)} → ${fmt(e)}`;
const pct = (x?: number) => (typeof x === "number" ? `${Math.round(x * 100)}%` : "—");

const renderAsset = (a?: AssetRef | string) => {
  if (!a) return "—";
  if (typeof a === "string") return a;
  const suffix = a.format ? ` · ${a.format}` : "";
  return `${a.name ?? a.id ?? "Asset"}${suffix}`;
};
const renderChannel = (c?: ChannelRef | string) => {
  if (!c) return "";
  if (typeof c === "string") return c;
  const suffix = c.type ? ` · ${c.type}` : "";
  return `${c.name ?? c.id ?? "Channel"}${suffix}`;
};

/** Normalize any legacy/partial responses into v2 `{ themes: [...] }` */
function normalizeToV2(input: any): Plan {
  // Accept top-level plan OR { plan } OR { data }
  const body = input?.plan ?? input?.data ?? input ?? {};
  const plan: Plan = typeof body === "object" && body ? { ...body } : {};

  // defaults
  plan.meta = plan.meta ?? { version: "2.0" };
  plan.meta.version = plan.meta.version ?? "2.0";
  plan.portfolio = plan.portfolio ?? { keyStats: {} };
  plan.portfolio.keyStats = plan.portfolio.keyStats ?? {
    totalTargetAccounts: 0,
    totalPersonasToEngage: 0,
    expectedWinsPct: 0,
    averageAccountBelief: "Unaware",
    timeToWinMonths: 12,
  };

  // (1) Already v2
  if (Array.isArray(plan.themes) && plan.themes.length) return plan;

  // (2) quarters → themes
  if (Array.isArray(plan.quarters) && plan.quarters.length) {
    plan.themes = plan.quarters.map((q, i) => ({
      id: `theme:${q.label ?? `Q${i + 1}`}`,
      name: q.label ?? `Q${i + 1}`,
      explanation: "Auto-converted from quarters",
      objective: "",
      targetAccounts: [],
      campaigns: (q.campaigns ?? []).map((c) => ({ ...c })),
    }));
    return plan;
  }

  // (3) phases → themes (frozen strategy shape)
  if (Array.isArray((plan as any).phases) && (plan as any).phases.length) {
    const phases = (plan as any).phases as any[];
    plan.themes = phases.map((ph, i) => ({
      id: `theme:phase_${i + 1}`,
      name: ph?.name || ph?.label || `Phase ${i + 1}`,
      explanation: ph?.explanation || "Auto-converted from phases",
      objective: ph?.objective || "",
      targetAccounts: [],
      campaigns: Array.isArray(ph?.campaigns) ? ph.campaigns : [],
    }));
    return plan;
  }

  // (4) scaffold.execution_plan.campaigns → single theme
  const campaignsFromScaffold = body?.execution_plan?.campaigns;
  if (Array.isArray(campaignsFromScaffold) && campaignsFromScaffold.length) {
    plan.themes = [
      {
        id: "theme:scaffold",
        name: "Account Plan",
        explanation: body?.zmot_theme ? `ZMOT: ${String(body.zmot_theme)}` : "Auto-converted from scaffold",
        objective: "",
        targetAccounts: [],
        campaigns: campaignsFromScaffold,
      },
    ];
    return plan;
  }

  // (5) nothing usable
  if (typeof window !== "undefined" && !(plan as any)._normalizedLogged) {
    console.warn("[normalizeToV2] No themes/quarters/phases/campaigns found in payload:", body);
    (plan as any)._normalizedLogged = true;
  }
  plan.themes = [];
  return plan;
}

/** Compute personas to engage on the fly if backend didn’t fill it. */
function computePersonasFromThemes(themes: Theme[] | undefined): number {
  if (!themes?.length) return 0;
  const set = new Set<string>();
  themes.forEach((t) =>
    (t.campaigns ?? []).forEach((c) =>
      (c.personas ?? []).forEach((p) => set.add(String(p)))
    )
  );
  return set.size;
}

/* =======================
   Arsenal Table
   ======================= */
const ArsenalTable: React.FC<{ rows?: ArsenalRow[] }> = ({ rows }) => (
  <TableContainer component={Paper} variant="outlined" sx={{ mt: 1 }}>
    <Table size="small">
      <TableHead>
        <TableRow>
          <TableCell>Asset</TableCell>
          <TableCell>Channel(s)</TableCell>
          <TableCell>Fitment</TableCell>
          <TableCell>Engagement</TableCell>
          <TableCell>Expected&nbsp;Lift</TableCell>
          <TableCell>Concerns</TableCell>
          <TableCell align="right">Why / Breadth</TableCell>
        </TableRow>
      </TableHead>
      <TableBody>
        {rows && rows.length ? (
          rows.map((r, i) => {
            const breadthPct =
              r.breadthScore != null ? `${Math.round((r.breadthScore || 0) * 100)}%` : "—";
            const channelsText =
              r.channels && r.channels.trim().length
                ? r.channels
                : renderChannel(r.channel) || "—";
            return (
              <TableRow key={i}>
                <TableCell>{renderAsset(r.asset)}</TableCell>
                <TableCell>{channelsText}</TableCell>
                <TableCell>{fmt(r.fitment)}</TableCell>
                <TableCell>{fmt(r.engagement)}</TableCell>
                <TableCell>{fmt(r.expectedLift)}</TableCell>
                <TableCell>
                  <Stack direction="row" spacing={0.5} sx={{ flexWrap: "wrap" }}>
                    {(r.concernsAddressed ?? []).slice(0, 6).map((c, idx) => (
                      <Chip key={`${c}-${idx}`} label={c} size="small" />
                    ))}
                    {(r.concernsAddressed?.length ?? 0) > 6 && (
                      <Chip
                        label={`+${(r.concernsAddressed!.length - 6)} more`}
                        size="small"
                        variant="outlined"
                      />
                    )}
                  </Stack>
                </TableCell>
                <TableCell align="right">
                  <Stack direction="row" spacing={1} justifyContent="flex-end">
                    <Tooltip title={r.why || "No reasoning available"}>
                      <Chip label="Why" size="small" variant="outlined" />
                    </Tooltip>
                    <Chip label={`Breadth ${breadthPct}`} size="small" />
                  </Stack>
                </TableCell>
              </TableRow>
            );
          })
        ) : (
          <TableRow>
            <TableCell colSpan={7} align="center" sx={{ color: "text.secondary" }}>
              No assets defined.
            </TableCell>
          </TableRow>
        )}
      </TableBody>
    </Table>
  </TableContainer>
);

/* =======================
   Campaign Card
   ======================= */
const CampaignCard: React.FC<{ campaign: Campaign }> = ({ campaign }) => (
  <Card variant="outlined" sx={{ mb: 2 }}>
    <CardContent>
      <Grid container spacing={2}>
        <Grid item xs={12} md={8}>
          <Typography variant="subtitle1">{campaign.description}</Typography>
          <Stack direction="row" spacing={1} sx={{ flexWrap: "wrap", mt: 1 }}>
            {(campaign.personas ?? []).map((p) => (
              <Chip key={p} label={p} size="small" />
            ))}
          </Stack>
        </Grid>
        <Grid item xs={12} md={4}>
          <Typography variant="caption" color="text.secondary">
            Execution timeframe
          </Typography>
          <Typography variant="body2">
            {dateRange(campaign.timeframe?.startDate, campaign.timeframe?.endDate)}
          </Typography>
        </Grid>
      </Grid>

      <Divider sx={{ my: 1.5 }} />

      <Typography variant="subtitle2" sx={{ mb: 0.5 }}>
        Assets + Channels
      </Typography>
      <ArsenalTable rows={campaign.arsenalTable} />
    </CardContent>
  </Card>
);

/* =======================
   Theme (v2 primary)
   ======================= */
const ThemeAccordion: React.FC<{ theme: Theme }> = ({ theme }) => (
  <Accordion defaultExpanded disableGutters sx={{ mb: 2 }}>
    <AccordionSummary expandIcon={<ExpandMoreIcon />}>
      <Typography variant="h6">{theme.name}</Typography>
    </AccordionSummary>
    <AccordionDetails>
      {(theme.explanation || theme.objective || (theme.targetAccounts?.length ?? 0) > 0) && (
        <Grid container spacing={1} sx={{ mb: 2 }}>
          <Grid item xs={12} md={8}>
            {theme.explanation && (
              <>
                <Typography variant="subtitle2">Explanation</Typography>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  {theme.explanation}
                </Typography>
              </>
            )}
            {theme.objective && (
              <>
                <Typography variant="subtitle2">Objective</Typography>
                <Typography variant="body2">{theme.objective}</Typography>
              </>
            )}
          </Grid>
          <Grid item xs={12} md={4}>
            {(theme.targetAccounts?.length ?? 0) > 0 && (
              <>
                <Typography variant="subtitle2">Target Accounts</Typography>
                <Stack direction="row" spacing={1} sx={{ flexWrap: "wrap", mt: 0.5 }}>
                  {(theme.targetAccounts ?? []).map((a, i) => {
                    const label = typeof a === "string" ? a : a?.id ?? `acct-${i}`;
                    return <Chip key={label} label={label} size="small" />;
                  })}
                </Stack>
              </>
            )}
          </Grid>
        </Grid>
      )}

      {(theme.campaigns ?? []).map((c) => (
        <CampaignCard key={c.id} campaign={c} />
      ))}
    </AccordionDetails>
  </Accordion>
);

/* =======================
   Stat card
   ======================= */
const Stat: React.FC<{ label: string; value: string | number }> = ({ label, value }) => (
  <Card variant="outlined" sx={{ minWidth: 180 }}>
    <CardContent sx={{ py: 1.5 }}>
      <Typography variant="caption" color="text.secondary">
        {label}
      </Typography>
      <Typography variant="h6" sx={{ mt: 0.5 }}>
        {String(value)}
      </Typography>
    </CardContent>
  </Card>
);

/* =======================
   Main
   ======================= */
const MarketingPlanner: React.FC = () => {
  const token = typeof window !== "undefined" ? localStorage.getItem("token") : null;

  // ⬇️ these MUST be inside the component
  const [accounts, setAccounts] = useState<{ id: string; name: string }[]>([]);
  const [selectedAccountId, setSelectedAccountId] = useState<string>("");

  const [plan, setPlan] = useState<Plan | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [statusMsg, setStatusMsg] = useState("");
  const [products, setProducts] = useState<{ id: string; name: string }[]>([]);
  const [selectedProductId, setSelectedProductId] = useState<string>("");
  const [companyId, setCompanyId] = useState<string | null>(null);

  // 1) Company + Products
  useEffect(() => {
    const fetchCompanyAndProducts = async () => {
      try {
        const meRes = await fetch("http://localhost:8000/me", {
          headers: { Authorization: `Bearer ${token}` },
        });
        if (!meRes.ok) throw new Error(await meRes.text());
        const meData = await meRes.json();
        setCompanyId(meData.company_id);

        const prodRes = await fetch(
          `http://localhost:8000/get-products/${meData.company_id}`,
          { headers: { Authorization: `Bearer ${token}` } }
        );
        if (!prodRes.ok) throw new Error(await prodRes.text());
        const prodData = await prodRes.json();

        const list = prodData.products ?? [];
        if (list.length === 0) {
          setStatusMsg("No products found. Please run Value Prop first.");
          return;
        }
        setProducts(list);
        setSelectedProductId((prev) => prev || list[0].id); // ensure a selection
      } catch (err: any) {
        console.error(err);
        setError(err.message || "Error fetching company or products.");
      }
    };
    if (token) fetchCompanyAndProducts();
  }, [token]);

  // 2) Fetch Accounts for selected product
  useEffect(() => {
    if (!selectedProductId) return;
    (async () => {
      try {
        const res = await fetch(`http://localhost:8000/get-accounts-for-rcs/${selectedProductId}`, {
          headers: { Authorization: `Bearer ${token}` },
        });
        if (!res.ok) throw new Error(await res.text());
        const data = await res.json();
        console.log("Tgt accts recd:", data);
        const list = (data.accounts ?? []).map((a: any) => ({
          id: a.id,
          name: a.account_name || a.name || a.id,
        }));
        setAccounts(list);
        // default to consolidated on product change
        setSelectedAccountId("");
      } catch (e) {
        console.error(e);
        setAccounts([]);
      }
    })();
  }, [selectedProductId, token]);

  // 3) Fetch Plan (refetch when view changes)
  useEffect(() => {
    if (!companyId || !selectedProductId) return;
    (async () => {
      setLoading(true);
      try {
        const q = selectedAccountId ? `?account_id=${encodeURIComponent(selectedAccountId)}` : "";
        console.log("Fetching plan with q=", q);
        const res = await fetch(
          `http://localhost:8000/get-comprehensive-execution-plan/${selectedProductId}${q}`,
          { headers: { Authorization: `Bearer ${token}` } }
        );
        if (!res.ok) throw new Error(await res.text());
        const raw = await res.json();
        const normalized = normalizeToV2(raw);
        setPlan(normalized);
        setStatusMsg(`themes=${normalized.themes?.length ?? 0}`);
      } catch (e: any) {
        console.error(e);
        setError(e.message || "Failed to fetch plan");
      } finally {
        setLoading(false);
      }
    })();
  }, [companyId, selectedProductId, selectedAccountId, token]); // include selectedAccountId

  const themes = useMemo(() => plan?.themes ?? [], [plan]);
  const computedPersonas = useMemo(() => computePersonasFromThemes(themes), [themes]);

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
        No plan data available.
      </Typography>
    );
  }

  const stats = plan.portfolio?.keyStats ?? {};
  const personasToEngage =
    typeof stats.totalPersonasToEngage === "number" && stats.totalPersonasToEngage > 0
      ? stats.totalPersonasToEngage
      : computedPersonas;

  return (
    <Box sx={{ p: 3 }}>
      <Stack direction="row" spacing={2} alignItems="center" sx={{ mb: 2 }}>
        <Typography variant="h5">Comprehensive Execution Plan</Typography>

        {/* Product selector */}
        {products.length > 0 && (
          <FormControl size="small" sx={{ minWidth: 240, ml: "auto" }}>
            <InputLabel id="product-select-label">Product</InputLabel>
            <Select
              labelId="product-select-label"
              label="Product"
              value={selectedProductId}
              onChange={(e) => setSelectedProductId(String(e.target.value))}
            >
              {products.map((p) => (
                <MenuItem key={p.id} value={p.id}>
                  {p.name || p.id}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        )}

        {/* Account selector */}
        {selectedProductId && (
          <FormControl size="small" sx={{ minWidth: 260 }}>
            <InputLabel id="account-select-label">View</InputLabel>
            <Select
              labelId="account-select-label"
              label="View"
              value={selectedAccountId}
              onChange={(e) => setSelectedAccountId(String(e.target.value))}
            >
              <MenuItem value="">Consolidated Plan</MenuItem>
              {accounts.map((a) => (
                <MenuItem key={a.id} value={a.id}>
                  Plan for {a.name}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        )}
      </Stack>

      {!!statusMsg && (
        <Alert severity="info" sx={{ mb: 2 }}>
          {statusMsg}
        </Alert>
      )}

      {/* Portfolio Section */}
      <Card variant="outlined" sx={{ p: 2, mb: 3 }}>
        <Typography variant="subtitle1" sx={{ mb: 1 }}>
          Portfolio Key Stats
        </Typography>
        <Stack direction="row" spacing={2} sx={{ flexWrap: "wrap" }}>
          <Stat label="Target Accounts" value={stats.totalTargetAccounts ?? 0} />
          <Stat label="Personas to Engage" value={personasToEngage ?? 0} />
          <Stat label="Time to Win" value={`${stats.timeToWinMonths ?? 0} months`} />
          <Stat label="Expected Wins" value={pct(stats.expectedWinsPct)} />
          <Stat label="Avg Belief" value={stats.averageAccountBelief ?? "—"} />
        </Stack>
      </Card>

      {/* Themes view */}
      {themes.length > 0 ? (
        themes.map((t) => <ThemeAccordion key={t.id} theme={t} />)
      ) : (
        <Typography variant="body2" color="text.secondary">
          No plan data available.
        </Typography>
      )}
    </Box>
  );
};

export default MarketingPlanner;
