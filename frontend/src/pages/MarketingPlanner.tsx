import React, { useEffect, useState } from "react";
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
} from "@mui/material";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";

// ---------- Types (aligned to new backend) ----------
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
  reach_score?: number;
}
interface ArsenalRow {
  asset?: AssetRef | string;
  channel?: ChannelRef | string;
  fitment?: string;
  engagement?: string;
  expectedLift?: string;
  concernsAddressed?: string[];      // <- array of labels/stage labels
  why?: string;                      // optional rationale
  breadthScore?: number;             // optional breadth metric (0..1)
}

interface Campaign {
  id: string;
  description: string;
  timeframe?: { startDate?: string; endDate?: string };
  personas?: string[];
  arsenalTable?: ArsenalRow[];
}

interface Quarter {
  label: string; // "Q1" | "Q2" | ...
  timeframe?: { startDate?: string; endDate?: string };
  successMetrics?: {
    personasEngaged?: number;
    expectedBeliefShift?: number; // e.g. 0.1 => +10%
  };
  campaigns: Campaign[];
}

interface Theme {                     // fallback support (old shape)
  id: string;
  name: string;
  explanation: string;
  objective: string;
  targetAccounts?: string[];
  campaigns?: Campaign[];
}

interface Plan {
  meta?: {
    version?: string;
    generatedAt?: string;
    beliefScale?: string[];
  };
  portfolio?: {
    keyStats?: {
      totalTargetAccounts?: number;
      totalPersonasToEngage?: number;
      expectedWinsPct?: number;      // 0..1
      averageAccountBelief?: string;
      timeToWinMonths?: number;
    };
  };
  // NEW preferred shape
  quarters?: Quarter[];
  // BACK-COMPAT
  themes?: Theme[];
}

// ---------- Helpers ----------
const fmt = (v?: string | number | null) => (v === 0 ? "0" : v ? String(v) : "—");
const dateRange = (s?: string, e?: string) => `${fmt(s)} → ${fmt(e)}`;
const pct = (x?: number) =>
  typeof x === "number" ? `${Math.round(x * 100)}%` : "—";

const renderAsset = (a?: AssetRef | string) => {
  if (!a) return "—";
  if (typeof a === "string") return a;
  const suffix = a.format ? ` · ${a.format}` : "";
  return `${a.name ?? a.id ?? "Asset"}${suffix}`;
};

const renderChannel = (c?: ChannelRef | string) => {
  if (!c) return "—";
  if (typeof c === "string") return c;
  const suffix = c.type ? ` · ${c.type}` : "";
  return `${c.name ?? c.id ?? "Channel"}${suffix}`;
};

// ---------- Arsenal Table ----------
const ArsenalTable: React.FC<{ rows?: ArsenalRow[] }> = ({ rows }) => (
  <TableContainer component={Paper} variant="outlined" sx={{ mt: 1 }}>
    <Table size="small">
      <TableHead>
        <TableRow>
          <TableCell>Asset</TableCell>
          <TableCell>Channel</TableCell>
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
            return (
              <TableRow key={i}>
                <TableCell>{renderAsset(r.asset)}</TableCell>
                <TableCell>{renderChannel(r.channel)}</TableCell>
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

// ---------- Campaign Card ----------
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

// ---------- Quarter Section ----------
const QuarterSection: React.FC<{ quarter: Quarter }> = ({ quarter }) => (
  <Accordion defaultExpanded disableGutters sx={{ mb: 2 }}>
    <AccordionSummary expandIcon={<ExpandMoreIcon />}>
      <Stack direction="row" spacing={2} alignItems="center">
        <Typography variant="h6">{quarter.label}</Typography>
        <Typography variant="body2" color="text.secondary">
          {dateRange(quarter.timeframe?.startDate, quarter.timeframe?.endDate)}
        </Typography>
      </Stack>
    </AccordionSummary>
    <AccordionDetails>
      {/* Success metrics */}
      <Card variant="outlined" sx={{ mb: 2 }}>
        <CardContent sx={{ py: 1.5 }}>
          <Stack direction="row" spacing={2} sx={{ flexWrap: "wrap" }}>
            <Stack>
              <Typography variant="caption" color="text.secondary">
                Personas Engaged
              </Typography>
              <Typography variant="h6">
                {quarter.successMetrics?.personasEngaged ?? "—"}
              </Typography>
            </Stack>
            <Stack>
              <Typography variant="caption" color="text.secondary">
                Expected Belief Shift
              </Typography>
              <Typography variant="h6">
                {pct(quarter.successMetrics?.expectedBeliefShift)}
              </Typography>
            </Stack>
          </Stack>
        </CardContent>
      </Card>

      {(quarter.campaigns ?? []).map((c) => (
        <CampaignCard key={c.id} campaign={c} />
      ))}
    </AccordionDetails>
  </Accordion>
);

// ---------- Theme Accordion (fallback for old shape) ----------
const ThemeAccordion: React.FC<{ theme: Theme }> = ({ theme }) => (
  <Accordion defaultExpanded disableGutters sx={{ mb: 2 }}>
    <AccordionSummary expandIcon={<ExpandMoreIcon />}>
      <Typography variant="h6">{theme.name}</Typography>
    </AccordionSummary>
    <AccordionDetails>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
        {theme.explanation}
      </Typography>
      <Grid container spacing={1} sx={{ mb: 2 }}>
        <Grid item xs={12} md={8}>
          <Typography variant="subtitle2">Objective</Typography>
          <Typography variant="body2">{theme.objective}</Typography>
        </Grid>
        <Grid item xs={12} md={4}>
          <Typography variant="subtitle2">Target Accounts</Typography>
          <Stack direction="row" spacing={1} sx={{ flexWrap: "wrap", mt: 0.5 }}>
            {(theme.targetAccounts ?? []).map((a) => (
              <Chip key={a} label={a} size="small" />
            ))}
          </Stack>
        </Grid>
      </Grid>
      {(theme.campaigns ?? []).map((c) => (
        <CampaignCard key={c.id} campaign={c} />
      ))}
    </AccordionDetails>
  </Accordion>
);

// ---------- Stat card ----------
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

// ---------- Main ----------
const MarketingPlanner: React.FC = () => {
  const token = typeof window !== "undefined" ? localStorage.getItem("token") : null;
  const [plan, setPlan] = useState<Plan | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [statusMsg, setStatusMsg] = useState("");
  const [products, setProducts] = useState<{ id: string; name: string }[]>([]);
  const [selectedProductId, setSelectedProductId] = useState<string>("");
  const [companyId, setCompanyId] = useState<string | null>(null);

 // 1. Get company ID on mount
   useEffect(() => {
     const fetchCompanyAndProducts = async () => {
       try {
         const meRes = await fetch("http://localhost:8000/me", {
           headers: { Authorization: `Bearer ${token}` },
         });
         const meData = await meRes.json();
         setCompanyId(meData.company_id);
 
         // 2. Get product IDs for this company
         const prodRes = await fetch(
           `http://localhost:8000/get-products/${meData.company_id}`,
           {
             headers: { Authorization: `Bearer ${token}` },
           }
         );
         const prodData = await prodRes.json();
         if (!prodData.products || prodData.products.length === 0) {
           setStatusMsg("No products found. Please run Value Prop first.");
           return;
         }
         console.log("✅ Product IDs set:", prodData.products.map((p: any) => p.id));
         setProducts(prodData.products);
 
         // 3. If only one product, select it automatically
         if (prodData.products.length === 1) {
           console.log("✅ Automatically selecting single product:", prodData.products[0].id);
           setSelectedProductId(prodData.products[0].id);
         }
       } catch (err) {
         setStatusMsg("Error fetching company or products.");
       }
     };
     fetchCompanyAndProducts();
   }, [token]);
   
   // Fetch entry parameters from graph - to be designed
   
 
   // Fetch Marketing Plan
   useEffect(() => {
     if (!companyId || !selectedProductId) return;
     (async () => {
       setLoading(true);
       try {
         const res = await fetch(
           `http://localhost:8000/get-comprehensive-execution-plan/${selectedProductId}`,
           { headers: { Authorization: `Bearer ${token}` } }
         );
        const data = await res.json();
        console.log("✅ Fetched Marketing Plan:", data);
        // Map backend fields to frontend fields
        setPlan(data.plan || data);
        console.log("Plan data:", data);
         
       } catch (e) {
         setError(e.message);
       } finally {
         setLoading(false);
       }
     })();
   }, [companyId, selectedProductId, token]);


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
        <Typography color="error">{error}</Typography>
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

  const stats = plan?.portfolio?.keyStats ?? {
    totalTargetAccounts: 0,
    totalPersonasToEngage: 0,
    expectedWinsPct: 0,
    averageAccountBelief: "—",
    timeToWinMonths: 0,
  };

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h5" sx={{ mb: 2 }}>
        Comprehensive Execution Plan
      </Typography>

      {/* Portfolio Section */}
      {plan.portfolio?.keyStats && (
        <Card variant="outlined" sx={{ p: 2, mb: 3 }}>
          <Typography variant="subtitle1" sx={{ mb: 1 }}>
            Portfolio Key Stats
          </Typography>
          <Stack direction="row" spacing={2} sx={{ flexWrap: "wrap" }}>
            <Stat label="Target Accounts" value={stats.totalTargetAccounts} />
            <Stat label="Personas to Engage" value={stats.totalPersonasToEngage} />
            <Stat label="Time to Win" value={`${stats.timeToWinMonths} months`} />
            <Stat label="Expected Wins" value={pct(stats.expectedWinsPct)} />
            <Stat label="Avg Belief" value={stats.averageAccountBelief ?? "—"} />
          </Stack>
        </Card>
      )}

      {/* Preferred: Quarters view */}
      {Array.isArray(plan.quarters) && plan.quarters.length > 0 ? (
        plan.quarters.map((q, idx) => <QuarterSection key={`${q.label}-${idx}`} quarter={q} />)
      ) : // Fallback: legacy themes view
      Array.isArray(plan.themes) && plan.themes.length > 0 ? (
        plan.themes.map((t) => <ThemeAccordion key={t.id} theme={t} />)
      ) : (
        <Typography variant="body2" color="text.secondary">
          No plan data available.
        </Typography>
      )}
    </Box>
  );
};

export default MarketingPlanner;
