import React, { useEffect, useId, useMemo, useState } from "react";
import {
  Box,
  Chip,
  FormControl,
  InputLabel,
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
  Typography,
} from "@mui/material";

export type AssetCadenceRow = {
  timeframe?: string | null;
  campaign_theme?: string | null;
  persona_id?: string | null;
  persona_label?: string | null;
  funnel_phase?: string | null;
  cadence_phase?: string | null;
  asset?: string | null;
  channel?: string | null;
  mode?: string | null;
  expected_delta_bp?: number | null;
  confidence?: number | null;
  duration_days?: number | null;
  target_stage?: string | null;
  target_concern?: string | null;
  segment_fit_score?: number | null;
  belief_probability?: number | null;
  evidence?: {
    segment_fit?: string[];
    belief_alignment?: string[];
    historical_win_lift?: number | null;
    evidence_count?: number | null;
    channel_stage_fit?: string[];
    channel_concern_fit?: string[];
    concern_match?: string | null;
  } | null;
};

type Props = {
  rows?: AssetCadenceRow[];
  personaLabelLookup?: Record<string, string>;
  enableFilters?: boolean;
  emptyCopy?: string;
};

type FunnelBand = "Early" | "Mid" | "Late";

const cadenceBand = (phase?: string | null): FunnelBand => {
  const normalized = (phase || "").toLowerCase();
  if (normalized === "probe" || normalized === "ramp") return "Early";
  if (normalized === "sustain") return "Mid";
  return "Late";
};

const titleize = (value?: string | null) => {
  if (!value) return "";
  return value
    .split(/[\s_-]+/)
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1).toLowerCase())
    .join(" ");
};

const fmtPercent = (value?: number | null) => {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  const scaled = value > 1 ? value : value * 100;
  const normalized = Math.min(100, Math.max(0, scaled));
  return `${normalized.toFixed(normalized >= 10 ? 0 : 1)}%`;
};

const fmtBasisPoints = (value?: number | null) => {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return `${Number(value).toFixed(1)} bp`;
};

const fmtDays = (value?: number | null) => {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  if (value < 1) return "<1 day";
  if (value === 1) return "1 day";
  return `${Math.round(value)} days`;
};

const uniqueList = (values: Array<string | null | undefined>) =>
  Array.from(new Set(values.filter(Boolean) as string[]));

const AssetCadenceTable: React.FC<Props> = ({
  rows,
  personaLabelLookup = {},
  enableFilters = false,
  emptyCopy = "Asset cadence will populate once campaign plays are generated.",
}) => {
  const data = rows || [];
  const [timeframeFilter, setTimeframeFilter] = useState("all");
  const [themeFilter, setThemeFilter] = useState("all");
  const timeframes = useMemo(
    () => uniqueList(data.map((row) => row.timeframe)),
    [data]
  );
  const themes = useMemo(
    () => uniqueList(data.map((row) => row.campaign_theme)),
    [data]
  );

  useEffect(() => {
    if (
      timeframeFilter !== "all" &&
      timeframes.length &&
      !timeframes.includes(timeframeFilter)
    ) {
      setTimeframeFilter("all");
    }
  }, [timeframes, timeframeFilter]);

  useEffect(() => {
    if (themeFilter !== "all" && themes.length && !themes.includes(themeFilter)) {
      setThemeFilter("all");
    }
  }, [themes, themeFilter]);

  const filteredRows = useMemo(() => {
    return data.filter((row) => {
      const matchesTimeframe =
        timeframeFilter === "all" ||
        (row.timeframe || "").toLowerCase() === timeframeFilter.toLowerCase();
      const matchesTheme =
        themeFilter === "all" ||
        (row.campaign_theme || "").toLowerCase() === themeFilter.toLowerCase();
      return matchesTimeframe && matchesTheme;
    });
  }, [data, timeframeFilter, themeFilter]);

  const timeframeId = useId();
  const themeId = useId();

  const highlightPlay = (row: AssetCadenceRow) => {
    const lift = row.expected_delta_bp ?? 0;
    const fit = row.segment_fit_score ?? 0;
    const belief = row.belief_probability ?? 0;
    return lift >= 35 && fit >= 0.6 && belief >= 0.4;
  };

  const renderEvidence = (row: AssetCadenceRow) => {
    const evidence = row.evidence;
    if (!evidence) {
      return (
        <Typography variant="body2" color="text.secondary">
          —
        </Typography>
      );
    }
    const chips: string[] = [];
    if (evidence.segment_fit?.length) {
      chips.push(`Segment: ${evidence.segment_fit.join(", ")}`);
    }
    if (evidence.belief_alignment?.length) {
      chips.push(`Belief tokens: ${evidence.belief_alignment.join(", ")}`);
    }
    if (typeof evidence.historical_win_lift === "number") {
      chips.push(`Historical lift ${fmtBasisPoints(evidence.historical_win_lift)}`);
    }
    if (typeof evidence.evidence_count === "number" && evidence.evidence_count > 0) {
      chips.push(`${evidence.evidence_count} wins logged`);
    }
    if (evidence.channel_stage_fit?.length) {
      chips.push(`Channel stage: ${evidence.channel_stage_fit.join(", ")}`);
    }
    if (evidence.channel_concern_fit?.length) {
      chips.push(`Channel concern: ${evidence.channel_concern_fit.join(", ")}`);
    }
    if (evidence.concern_match) {
      chips.push(`Concern match: ${evidence.concern_match}`);
    }
    if (!chips.length) {
      return (
        <Typography variant="body2" color="text.secondary">
          —
        </Typography>
      );
    }
    return (
      <Stack direction="row" spacing={0.5} sx={{ flexWrap: "wrap" }}>
        {chips.map((label) => (
          <Chip key={`evidence-${label}`} size="small" variant="outlined" label={label} />
        ))}
      </Stack>
    );
  };

  if (!data.length) {
    return (
      <Typography variant="body2" color="text.secondary">
        {emptyCopy}
      </Typography>
    );
  }

  return (
    <Box>
      {enableFilters && (timeframes.length > 1 || themes.length > 1) ? (
        <Stack
          direction={{ xs: "column", sm: "row" }}
          spacing={1.5}
          sx={{ mb: 1.5 }}
        >
          {timeframes.length > 1 ? (
            <FormControl size="small" sx={{ minWidth: 180 }}>
              <InputLabel id={`${timeframeId}-label`}>Timeframe</InputLabel>
              <Select
                labelId={`${timeframeId}-label`}
                value={timeframeFilter}
                label="Timeframe"
                onChange={(event) => setTimeframeFilter(event.target.value)}
              >
                <MenuItem value="all">All timeframes</MenuItem>
                {timeframes.map((value) => (
                  <MenuItem key={`timeframe-${value}`} value={value}>
                    {value}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          ) : null}
          {themes.length > 1 ? (
            <FormControl size="small" sx={{ minWidth: 220 }}>
              <InputLabel id={`${themeId}-label`}>Campaign theme</InputLabel>
              <Select
                labelId={`${themeId}-label`}
                value={themeFilter}
                label="Campaign theme"
                onChange={(event) => setThemeFilter(event.target.value)}
              >
                <MenuItem value="all">All themes</MenuItem>
                {themes.map((value) => (
                  <MenuItem key={`theme-${value}`} value={value}>
                    {value}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          ) : null}
        </Stack>
      ) : null}
      <TableContainer component={Paper} variant="outlined">
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>Timeframe</TableCell>
              <TableCell>Campaign Theme</TableCell>
              <TableCell>Persona</TableCell>
              <TableCell>Belief Stage</TableCell>
              <TableCell>Concern</TableCell>
              <TableCell>Cadence Phase</TableCell>
              <TableCell>Asset</TableCell>
              <TableCell>Channel</TableCell>
              <TableCell>Mode</TableCell>
              <TableCell align="right">Δ (bps)</TableCell>
              <TableCell align="right">Confidence</TableCell>
              <TableCell align="right">Duration</TableCell>
              <TableCell>Why this asset?</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {filteredRows.length ? (
              filteredRows.map((row, idx) => {
                const personaLabel =
                  row.persona_label ||
                  (row.persona_id ? personaLabelLookup[row.persona_id] : null) ||
                  row.persona_id ||
                  "Persona";
                const funnelLabel = row.funnel_phase || "—";
                const beliefStage = row.target_stage || funnelLabel;
                const concernLabel = row.target_concern || "—";
                const phaseBand = cadenceBand(row.cadence_phase);
                const rowHighlight = highlightPlay(row);
                return (
                  <TableRow
                    key={`${row.timeframe || "timeframe"}-${row.campaign_theme || "theme"}-${idx}`}
                    sx={
                      rowHighlight
                        ? {
                            backgroundColor: "rgba(76, 175, 80, 0.08)",
                          }
                        : undefined
                    }
                  >
                    <TableCell>{row.timeframe || "—"}</TableCell>
                    <TableCell>{row.campaign_theme || "—"}</TableCell>
                    <TableCell>{personaLabel}</TableCell>
                    <TableCell>{beliefStage}</TableCell>
                    <TableCell>{concernLabel}</TableCell>
                    <TableCell>
                      <Chip
                        size="small"
                        color={
                          phaseBand === "Early"
                            ? "info"
                            : phaseBand === "Mid"
                            ? "success"
                            : "warning"
                        }
                        label={titleize(row.cadence_phase || "Phase")}
                      />
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2">
                        {row.asset || "Asset"}
                      </Typography>
                    </TableCell>
                    <TableCell>{row.channel || "—"}</TableCell>
                    <TableCell>
                      {row.mode ? titleize(row.mode) : "—"}
                    </TableCell>
                    <TableCell align="right">
                      {fmtBasisPoints(row.expected_delta_bp)}
                    </TableCell>
                    <TableCell align="right">
                      {fmtPercent(row.confidence ?? null)}
                    </TableCell>
                    <TableCell align="right">
                      {fmtDays(row.duration_days)}
                    </TableCell>
                    <TableCell>{renderEvidence(row)}</TableCell>
                  </TableRow>
                );
              })
            ) : (
              <TableRow>
                <TableCell colSpan={13}>
                  <Typography variant="body2" color="text.secondary">
                    No entries match the selected filters.
                  </Typography>
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </TableContainer>
    </Box>
  );
};

export default AssetCadenceTable;
