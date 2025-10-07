import React, { useEffect, useMemo, useState } from "react";
import {
  Box,
  Typography,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  CircularProgress,
  Alert,
  Chip,
  Divider,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Tooltip,
  LinearProgress,
  Stack,
} from "@mui/material";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";

type AlignMetrics = {
  MAE_models?: number;
  R2_models?: number;
  Brier_graph?: number;
  Brier_crm?: number;
};

type AlignRow = {
  account_id: string;
  industry?: string;
  revenue_range?: string;
  employee_range?: string;
  geography?: string;
  funding_stage?: string;
  p_crm_win: number;
  p_graph_win: number;
  residual: number;
};

type Proposal = {
  edge_from: string;       // node id or readable label
  edge_to: string;         // node id or readable label
  edge_type: string;       // e.g., LIKELIHOOD
  layer: string;           // e.g., "attr→pain_trigger"
  current_weight: number;
  proposed_weight: number;
  delta: number;
  support_n: number;
  confidence: number;      // 0..1
  reason?: string;
};

type ModelStats = {
  features_and_coefficients: { feature: string; coefficient: number }[];
  intercept?: number[] | null;
  df_augmented?: any[];

  // NEW (optional)
  alignment?: AlignMetrics;
  df_align?: AlignRow[];
  proposals?: Proposal[];
};

interface CRMWinModelsProps {
  productId: string;
  token?: string;
}

type FeatureRow = {
  raw: string;
  dimension: string;
  value: string;
  coefficient: number;
  oddsRatio: number;
};

const DIM_LABELS: Record<string, string> = {
  industry: "Industry",
  revenue_range: "Revenue Range",
  employee_range: "Employee Range",
  geography: "Geography",
  funding_stage: "Funding Stage",
  competitor_used: "Competitor Used",
  other_tech_stack: "Other Tech Stack",
};

function parseFeature(raw: string, coefficient: number): FeatureRow {
  const idx = raw.indexOf("_");
  const dimension = idx > -1 ? raw.slice(0, idx) : raw;
  const value = idx > -1 ? raw.slice(idx + 1) : "";
  const oddsRatio = Math.exp(coefficient);
  return { raw, dimension, value, coefficient, oddsRatio };
}

const numberFmt = (x: number, d = 4) => (Number.isFinite(x) ? x.toFixed(d) : "—");
const pctFmt = (x: number, d = 1) => (Number.isFinite(x) ? `${(x * 100).toFixed(d)}%` : "—");

function mean(values: number[]) {
  if (!values.length) return NaN;
  return values.reduce((a, b) => a + b, 0) / values.length;
}

function groupArchetypes(rows: AlignRow[]) {
  // Group by the 5 attributes we show (adjust if you want fewer)
  const keyOf = (r: AlignRow) =>
    [r.industry, r.revenue_range, r.employee_range, r.geography, r.funding_stage]
      .map((v) => v ?? "—")
      .join(" | ");

  const map = new Map<
    string,
    { count: number; pGraph: number[]; pCrm: number[]; anyRow: AlignRow | null }
  >();

  rows.forEach((r) => {
    const k = keyOf(r);
    const prev = map.get(k) ?? { count: 0, pGraph: [], pCrm: [], anyRow: null };
    prev.count += 1;
    prev.pGraph.push(r.p_graph_win);
    prev.pCrm.push(r.p_crm_win);
    prev.anyRow = prev.anyRow ?? r;
    map.set(k, prev);
  });

  const table = Array.from(map.entries()).map(([k, v]) => {
    const p_graph = mean(v.pGraph);
    const p_crm = mean(v.pCrm);
    return {
      archetype: k,
      count: v.count,
      p_graph,
      p_crm,
      delta: p_crm - p_graph,
      sample: v.anyRow!,
    };
  });

  // sort by absolute delta descending
  table.sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta));
  return table;
}

const CRMWinModels: React.FC<CRMWinModelsProps> = ({ productId, token }) => {
  const [modelStats, setModelStats] = useState<ModelStats | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!productId) return;
    setLoading(true);
    setError(null);
    (async () => {
      try {
        const res = await fetch(`http://localhost:8000/show-crm-win-model/${productId}`, {
          method: "GET",
          headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
        });
        if (!res.ok) throw new Error("Failed to load model stats");
        const data = await res.json();
        setModelStats(data);
        setError(null);
      } catch (e) {
        setError("Failed to load model stats");
      } finally {
        setLoading(false);
      }
    })();
  }, [productId, token]);

  const rows: FeatureRow[] = useMemo(() => {
    if (!modelStats?.features_and_coefficients) return [];
    return modelStats.features_and_coefficients.map(({ feature, coefficient }) =>
      parseFeature(feature, coefficient)
    );
  }, [modelStats]);

  const grouped = useMemo(() => {
    const m = new Map<string, FeatureRow[]>();
    rows.forEach((r) => {
      const key = r.dimension;
      if (!m.has(key)) m.set(key, []);
      m.get(key)!.push(r);
    });
    for (const [k, arr] of m) {
      arr.sort((a, b) => Math.abs(b.coefficient) - Math.abs(a.coefficient));
      m.set(k, arr);
    }
    return m;
  }, [rows]);

  const intercept = Array.isArray(modelStats?.intercept) ? modelStats!.intercept![0] : null;

  // ======= New: Alignment + df_align-derived summaries =======
  const align = modelStats?.alignment;
  const dfAlign = modelStats?.df_align ?? [];

  const overallMeans = useMemo(() => {
    if (!dfAlign.length) return null;
    return {
      pGraphMean: mean(dfAlign.map((r) => r.p_graph_win)),
      pCrmMean: mean(dfAlign.map((r) => r.p_crm_win)),
      delta: mean(dfAlign.map((r) => r.p_crm_win - r.p_graph_win)),
      n: dfAlign.length,
    };
  }, [dfAlign]);

  const archetypeTable = useMemo(() => {
    if (!dfAlign.length) return [];
    return groupArchetypes(dfAlign).slice(0, 12); // top 12 gaps
  }, [dfAlign]);

  const proposals = modelStats?.proposals ?? [];

  if (!productId) return <Typography>Please select a product.</Typography>;
  if (loading)
    return (
      <Box display="flex" alignItems="center" justifyContent="center" minHeight={120}>
        <CircularProgress />
        <Typography sx={{ ml: 2 }}>Loading model stats...</Typography>
      </Box>
    );
  if (error)
    return (
      <Alert severity="error" sx={{ my: 2 }}>
        {error}
      </Alert>
    );
  if (!modelStats)
    return <Typography>No model data available.</Typography>;

  return (
    <Box>
      <Typography variant="h5" sx={{ mb: 2, fontWeight: "bold" }}>
        CRM Win Model & Graph Reconciliation
      </Typography>

      {/* ===== Reconciliation Summary (optional) ===== */}
      {(align || overallMeans) && (
        <Paper sx={{ p: 2, mb: 2 }}>
          <Typography variant="subtitle1" sx={{ fontWeight: "bold", mb: 1 }}>
            Reconciliation Summary
          </Typography>
          <Stack direction={{ xs: "column", md: "row" }} spacing={2}>
            {overallMeans && (
              <Paper variant="outlined" sx={{ p: 1.5, flex: 1 }}>
                <Typography variant="body2" sx={{ fontWeight: "bold" }}>
                  Assumed vs Learned (overall)
                </Typography>
                <Box sx={{ display: "flex", gap: 2, mt: 1 }}>
                  <Chip
                    label={`Graph mean p(win): ${numberFmt(overallMeans.pGraphMean, 3)}`}
                    variant="outlined"
                  />
                  <Chip
                    color="primary"
                    label={`CRM mean p(win): ${numberFmt(overallMeans.pCrmMean, 3)}`}
                    variant="outlined"
                  />
                  <Chip
                    color={overallMeans.delta >= 0 ? "success" : "warning"}
                    label={`Δ (CRM-Graph): ${numberFmt(overallMeans.delta, 3)}`}
                    variant="outlined"
                  />
                  <Chip label={`n=${overallMeans.n}`} variant="outlined" />
                </Box>
              </Paper>
            )}
            {align && (
              <Paper variant="outlined" sx={{ p: 1.5, flex: 1 }}>
                <Typography variant="body2" sx={{ fontWeight: "bold" }}>
                  Fit Metrics
                </Typography>
                <Box sx={{ display: "flex", flexWrap: "wrap", gap: 1.5, mt: 1 }}>
                  <Chip label={`MAE: ${numberFmt(align.MAE_models ?? NaN, 3)}`} />
                  <Chip label={`R²: ${numberFmt(align.R2_models ?? NaN, 3)}`} />
                  {"Brier_graph" in align && (
                    <Chip label={`Brier (Graph): ${numberFmt(align.Brier_graph!, 3)}`} />
                  )}
                  {"Brier_crm" in align && (
                    <Chip color="primary" label={`Brier (CRM): ${numberFmt(align.Brier_crm!, 3)}`} />
                  )}
                </Box>
              </Paper>
            )}
          </Stack>
        </Paper>
      )}

      {/* ===== Assumed vs Learned by archetype (optional) ===== */}
      {!!archetypeTable.length && (
        <Paper sx={{ p: 2, mb: 2 }}>
          <Typography variant="subtitle1" sx={{ fontWeight: "bold", mb: 1 }}>
            Assumed vs Learned — Top Gaps (by Archetype)
          </Typography>
          <TableContainer component={Paper} variant="outlined">
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell sx={{ fontWeight: "bold" }}>Archetype</TableCell>
                  <TableCell sx={{ fontWeight: "bold" }} align="right">
                    n
                  </TableCell>
                  <TableCell sx={{ fontWeight: "bold" }} align="right">
                    Graph p(win)
                  </TableCell>
                  <TableCell sx={{ fontWeight: "bold" }} align="right">
                    CRM p(win)
                  </TableCell>
                  <TableCell sx={{ fontWeight: "bold" }} align="right">
                    Δ (CRM-Graph)
                  </TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {archetypeTable.map((r, i) => (
                  <TableRow key={i}>
                    <TableCell>{r.archetype}</TableCell>
                    <TableCell align="right">{r.count}</TableCell>
                    <TableCell align="right">{numberFmt(r.p_graph, 3)}</TableCell>
                    <TableCell align="right">{numberFmt(r.p_crm, 3)}</TableCell>
                    <TableCell align="right">
                      <Typography
                        component="span"
                        color={r.delta >= 0 ? "success.main" : "warning.main"}
                        sx={{ fontWeight: "bold" }}
                      >
                        {numberFmt(r.delta, 3)}
                      </Typography>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </Paper>
      )}

      {/* ===== Proposed Graph Edge Updates (optional) ===== */}
      {!!proposals.length && (
        <Paper sx={{ p: 2, mb: 2 }}>
          <Typography variant="subtitle1" sx={{ fontWeight: "bold", mb: 1 }}>
            Proposed Graph Edge Updates
          </Typography>
          <TableContainer component={Paper} variant="outlined">
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell sx={{ fontWeight: "bold" }}>Layer</TableCell>
                  <TableCell sx={{ fontWeight: "bold" }}>Edge</TableCell>
                  <TableCell sx={{ fontWeight: "bold" }} align="right">
                    Current
                  </TableCell>
                  <TableCell sx={{ fontWeight: "bold" }} align="right">
                    Proposed
                  </TableCell>
                  <TableCell sx={{ fontWeight: "bold" }} align="right">
                    Δ
                  </TableCell>
                  <TableCell sx={{ fontWeight: "bold" }} align="right">
                    n
                  </TableCell>
                  <TableCell sx={{ fontWeight: "bold" }} width={220}>
                    Confidence
                  </TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {proposals.map((p, i) => (
                  <TableRow key={i}>
                    <TableCell>{p.layer}</TableCell>
                    <TableCell>
                      <Typography variant="body2" sx={{ fontFamily: "monospace" }}>
                        {p.edge_from} → {p.edge_to}
                      </Typography>
                      {p.reason && (
                        <Typography variant="caption" color="text.secondary">
                          {p.reason}
                        </Typography>
                      )}
                    </TableCell>
                    <TableCell align="right">{numberFmt(p.current_weight, 3)}</TableCell>
                    <TableCell align="right">{numberFmt(p.proposed_weight, 3)}</TableCell>
                    <TableCell align="right">
                      <Typography
                        component="span"
                        color={p.delta >= 0 ? "success.main" : "warning.main"}
                        sx={{ fontWeight: "bold" }}
                      >
                        {numberFmt(p.delta, 3)}
                      </Typography>
                    </TableCell>
                    <TableCell align="right">{p.support_n}</TableCell>
                    <TableCell>
                      <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                        <Box sx={{ flex: 1 }}>
                          <LinearProgress
                            variant="determinate"
                            value={Math.max(0, Math.min(100, p.confidence * 100))}
                          />
                        </Box>
                        <Typography variant="caption" sx={{ minWidth: 38, textAlign: "right" }}>
                          {pctFmt(p.confidence, 0)}
                        </Typography>
                      </Box>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </Paper>
      )}

      {/* ===== Existing: Intercept & Coeffs ===== */}
      <Paper sx={{ p: 2, mb: 2 }}>
        <Typography variant="subtitle1" sx={{ fontWeight: "bold" }}>
          Intercept
        </Typography>
        <Box sx={{ ml: 0.5, display: "flex", alignItems: "center", gap: 2 }}>
          <Typography variant="body2">
            {intercept !== null ? numberFmt(intercept, 4) : "N/A"}{" "}
            <Tooltip title="Log-odds baseline; odds ratio = exp(intercept)">
              <Chip
                size="small"
                label={`OR = ${intercept !== null ? numberFmt(Math.exp(intercept), 3) : "—"}`}
                variant="outlined"
              />
            </Tooltip>
          </Typography>
        </Box>
      </Paper>

      {/* Coefficients grouped by dimension */}
      {[...grouped.entries()].map(([dim, arr]) => (
        <Accordion key={dim} defaultExpanded>
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Typography sx={{ fontWeight: "bold" }}>
              {(DIM_LABELS[dim] ?? dim) + ` (${arr.length})`}
            </Typography>
          </AccordionSummary>
          <AccordionDetails>
            <TableContainer component={Paper} variant="outlined">
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell sx={{ fontWeight: "bold" }}>{DIM_LABELS[dim] ?? dim}</TableCell>
                    <TableCell sx={{ fontWeight: "bold" }} align="right">
                      Coef (log-odds)
                    </TableCell>
                    <TableCell sx={{ fontWeight: "bold" }} align="right">
                      Odds Ratio (exp)
                    </TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {arr.map((r) => (
                    <TableRow key={r.raw}>
                      <TableCell>{r.value}</TableCell>
                      <TableCell align="right">{numberFmt(r.coefficient)}</TableCell>
                      <TableCell align="right">{numberFmt(r.oddsRatio, 3)}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </AccordionDetails>
        </Accordion>
      ))}

      {/* Optional: sample of df_augmented */}
      {Array.isArray(modelStats.df_augmented) && modelStats.df_augmented.length > 0 && (
        <>
          <Divider sx={{ my: 3 }} />
          <Typography variant="subtitle1" sx={{ fontWeight: "bold", mb: 1 }}>
            Training Rows (sample)
          </Typography>
          <TableContainer component={Paper} variant="outlined">
            <Table size="small">
              <TableHead>
                <TableRow>
                  {[
                    "account_id",
                    "deal_status",
                    "won",
                    "industry",
                    "revenue_range",
                    "employee_range",
                    "funding_stage",
                    "geography",
                  ].map((h) => (
                    <TableCell key={h} sx={{ fontWeight: "bold" }}>
                      {DIM_LABELS[h as keyof typeof DIM_LABELS] ?? h}
                    </TableCell>
                  ))}
                </TableRow>
              </TableHead>
              <TableBody>
                {modelStats.df_augmented.slice(0, 5).map((r, i) => (
                  <TableRow key={i}>
                    <TableCell>{r.account_id}</TableCell>
                    <TableCell>{r.deal_status}</TableCell>
                    <TableCell>{r.won}</TableCell>
                    <TableCell>{r.industry}</TableCell>
                    <TableCell>{r.revenue_range}</TableCell>
                    <TableCell>{r.employee_range}</TableCell>
                    <TableCell>{r.funding_stage}</TableCell>
                    <TableCell>{r.geography}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </>
      )}
    </Box>
  );
};

export default CRMWinModels;
