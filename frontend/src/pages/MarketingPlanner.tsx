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
  Grid,
  CircularProgress,
  Divider,
} from "@mui/material";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";

// ---------- Types ----------
interface ArsenalRow {
  asset: string;
  channels: string;
  fitment: string;
  engagement: string;
  expectedLift: string;
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
      expectedWinsPct?: number;
      averageAccountBelief?: string;
      timeToWinMonths?: number;
    };
  };
  themes?: Theme[];
}

// ---------- Helpers ----------
const fmt = (v?: string | number | null) => (v === 0 ? "0" : v ? String(v) : "—");
const dateRange = (s?: string, e?: string) => `${fmt(s)} → ${fmt(e)}`;
const pct = (x?: number) =>
  typeof x === "number" ? `${Math.round(x * 100)}%` : "—";

// ---------- Arsenal Table ----------
const ArsenalTable: React.FC<{ rows?: ArsenalRow[] }> = ({ rows }) => (
  <TableContainer component={Paper} variant="outlined" sx={{ mt: 1 }}>
    <Table size="small">
      <TableHead>
        <TableRow>
          <TableCell>Asset</TableCell>
          <TableCell>Recommended Channels</TableCell>
          <TableCell>Fitment</TableCell>
          <TableCell>Engagement</TableCell>
          <TableCell>Expected&nbsp;Lift</TableCell>
        </TableRow>
      </TableHead>
      <TableBody>
        {rows && rows.length ? (
          rows.map((r, i) => (
            <TableRow key={i}>
              <TableCell>{r.asset}</TableCell>
              <TableCell>{r.channels}</TableCell>
              <TableCell>{r.fitment}</TableCell>
              <TableCell>{r.engagement}</TableCell>
              <TableCell>{r.expectedLift}</TableCell>
            </TableRow>
          ))
        ) : (
          <TableRow>
            <TableCell colSpan={5} align="center" sx={{ color: "text.secondary" }}>
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
        <Grid size={{ xs: 12, md: 8 }}>
          <Typography variant="subtitle1">{campaign.description}</Typography>
          <Stack direction="row" spacing={1} sx={{ flexWrap: "wrap", mt: 1 }}>
            {(campaign.personas ?? []).map((p) => (
              <Chip key={p} label={p} size="small" />
            ))}
          </Stack>
        </Grid>
        <Grid size={{ xs: 12, md: 4 }}>
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

// ---------- Theme Accordion ----------
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
        <Grid size={{ xs: 12, md: 8 }}>
          <Typography variant="subtitle2">Objective</Typography>
          <Typography variant="body2">{theme.objective}</Typography>
        </Grid>
        <Grid size={{ xs: 12, md: 4 }}>
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
const Stat: React.FC<{ label: string; value: string | number }> = ({
  label,
  value,
}) => (
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
  const token =
    typeof window !== "undefined" ? localStorage.getItem("token") : null;
  const [plan, setPlan] = useState<Plan | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!token) return;
    (async () => {
      try {
        setLoading(true);
        setError(null);
        const res = await fetch(
          "http://localhost:8000/get-comprehensive-execution-plan/5a742a1a-fa9c-47f3-bca2-72e49cd10cd6",
          { headers: { Authorization: `Bearer ${token}` } }
        );
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        setPlan(data.plan || data);
        console.log("Plan data:", data);
      } catch (e: any) {
        setError(e.message);
      } finally {
        setLoading(false);
      }
    })();
  }, [token]);

  if (loading)
    return (
      <Box sx={{ display: "flex", justifyContent: "center", mt: 8 }}>
        <CircularProgress />
      </Box>
    );

  if (error)
    return (
      <Card variant="outlined" sx={{ p: 2, bgcolor: "#ffebee" }}>
        <Typography color="error">{error}</Typography>
      </Card>
    );

  if (!plan)
    return (
      <Typography variant="body1" color="text.secondary" sx={{ mt: 4 }}>
        No plan data available.
      </Typography>
    );

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
            <Stat
              label="Personas to Engage"
              value={stats.totalPersonasToEngage}
            />
            <Stat label="Time to Win" value={`${stats.timeToWinMonths} months`} />
            <Stat label="Expected Wins" value={pct(stats.expectedWinsPct)} />
            <Stat label="Avg Belief" value={stats.averageAccountBelief ?? "—"} />
          </Stack>
        </Card>
      )}

      {/* Themes */}
      {Array.isArray(plan.themes) && plan.themes.length > 0 ? (
        plan.themes.map((t) => <ThemeAccordion key={t.id} theme={t} />)
      ) : (
        <Typography variant="body2" color="text.secondary">
          No themes available.
        </Typography>
      )}
    </Box>
  );
};

export default MarketingPlanner;
