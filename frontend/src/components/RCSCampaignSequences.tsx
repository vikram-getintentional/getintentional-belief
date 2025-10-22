import React from "react";
import {
  Box,
  Card,
  CardContent,
  Chip,
  Divider,
  Stack,
  Typography,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  Tooltip,
  LinearProgress,
} from "@mui/material";

/** ===== Types kept minimal for resilience to backend drift ===== */
type LabelMap = {
  personaLabelById: Record<string, string>;
  concernLabelById: Record<string, string>;
};

type StrategyShape = {
  frozen_strategy?: {
    phases?: Array<{
      phaseIndex?: number;
      stageLabel?: string;
      stage?: string;
      timeframe?: { startDate?: string; endDate?: string; start?: string; end?: string };
      personas?: string[];
      campaigns?: Array<{
        id?: string;
        campaign_id?: string;
        description?: string;
        title?: string;
        arsenalTable?: any[];
        arsenal_table?: any[];
      }>;
    }>;
    personas?: any[];
    concerns_flat?: any[];
  };
  personas?: any[];
  concerns_flat?: any[];
};

function FitnessBar({ value }: { value?: number }) {
  const v = Math.max(0, Math.min(1, value ?? 0));
  const pct = Math.round(v * 100);
  return (
    <Typography variant="caption" color="text.secondary" sx={{ minWidth: 48 }}>
      {pct}%
    </Typography>
  );
}

/** ===== Arsenal table (Asset, Channel, Concerns, Engagement, Fitness) ===== */
function ArsenalTable({
  rows,
  labelMap,
}: {
  rows: any[];
  labelMap: LabelMap;
}) {
  const [expandedRows, setExpandedRows] = React.useState<Set<number>>(new Set());

  const toggleExpand = (idx: number) => {
    setExpandedRows(prev => {
      const next = new Set(prev);
      if (next.has(idx)) next.delete(idx);
      else next.add(idx);
      return next;
    });
  };

  if (!Array.isArray(rows) || rows.length === 0) {
    return (
      <Typography variant="caption" color="text.secondary">
        No arsenal rows for this campaign.
      </Typography>
    );
  }

  return (
    <Table size="small">
      <TableHead>
        <TableRow>
          <TableCell>Asset</TableCell>
          <TableCell>Channel (reach)</TableCell>
          <TableCell>Targeted</TableCell>
          <TableCell>Concerns Addressed</TableCell>
          <TableCell>Engagement</TableCell>
          <TableCell>Fitness</TableCell>
        </TableRow>
      </TableHead>
      <TableBody>
        {rows.map((row: any, i: number) => {
          // canonicalize asset / channel objects
          const asset = row.asset || row.play || row || {};
          const assetName =
            asset.asset_name || asset.name || asset.title || asset.id || "—";
          const assetFormat = asset.format || asset.type || asset.format_label || "";

          const channel = row.channel || row.play_channel || asset.channel || {};
          const channelName =
            channel.name || channel.channel_name || channel.id || "—";
          const channelReach =
            channel.reach ??
            channel.reach_score ??
            channel.reachScore ??
            channel.reach_score_value ??
            null;

          // support multiple shapes for concerns
          const concerns =
            row.concernsAddressed ??
            row.concerns_addressed ??
            row.concerns ??
            asset.concerns ??
            [];

          // engagement may be numeric or descriptive label
          const engagementRaw =
            row.engagement ??
            row.engagement_rate ??
            row.engagementLevel ??
            row.engagement_level ??
            row.eng ??
            null;
          const engagement =
            typeof engagementRaw === "number" ? engagementRaw : engagementRaw;

          // fitness / messaging: numeric fitness if present, otherwise try to infer from fit/fitment fields
          let fitnessNum: number | undefined =
            row.fitness ?? row.fit ?? asset.fitness ?? undefined;
          if (fitnessNum == null) {
            // try fitmentScore / expectedLift heuristics (strings like "+10%")
            const maybe = row.fitmentScore ?? row.fitment_score ?? row.expectedLift ?? row.expected_lift;
            if (typeof maybe === "string" && maybe.includes("%")) {
              const n = parseFloat(maybe.replace(/[^\d.-]/g, ""));
              if (!Number.isNaN(n)) {
                // map percent lift to 0..1 roughly
                fitnessNum = Math.min(1, Math.max(0, n / 100));
              }
            }
          }
          if (typeof fitnessNum === "string") {
            const parsed = parseFloat(String(fitnessNum));
            fitnessNum = Number.isFinite(parsed) ? parsed : undefined;
          }

          // fit label (human friendly) if available
          const fitLabel =
            row.fitment ?? row.fitmentLabel ?? row.fitment_label ?? row.fit ?? asset.fitment ?? asset.fit_label ?? "";

          const why = row.why ?? row.note ?? asset.why ?? "";

          // concerns display logic: show top 3, expand to show all
          const MAX_VISIBLE = 1;
          const isExpanded = expandedRows.has(i);
          const visibleConcerns = Array.isArray(concerns)
            ? (isExpanded ? concerns : concerns.slice(0, MAX_VISIBLE))
            : [];

          return (
            <TableRow hover key={`arsenal-row-${i}`}>
              <TableCell>
                <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap">
                  <Tooltip title={why || "No explanation provided"} arrow placement="top">
                    <span style={{ display: "inline-flex", alignItems: "center", cursor: why ? "pointer" : "default" }}>
                      <Typography variant="body2" sx={{ fontWeight: 600 }}>
                        {assetName}
                      </Typography>
                    </span>
                  </Tooltip>
                  {assetFormat ? <Chip size="small" variant="outlined" label={assetFormat} /> : null}
                </Stack>
              </TableCell>

              <TableCell>
                <Typography variant="body2">{channelName}</Typography>
                {channelReach != null && (
                  <Typography variant="caption" color="text.secondary" display="block">
                    reach: {channelReach}
                  </Typography>
                )}
              </TableCell>

              <TableCell>
                {typeof fitnessNum === "number" && fitnessNum >= 0.6 ? (
                  <Chip size="small" color="success" label="Targeted" />
                ) : (
                  <Chip size="small" label="—" />
                )}
              </TableCell>

              <TableCell>
                {Array.isArray(concerns) && concerns.length ? (
                  <Stack direction="row" spacing={0.5} flexWrap="wrap" alignItems="center">
                    {visibleConcerns.map((c: any, idx: number) => {
                      const label =
                        typeof c === "string"
                          ? labelMap.concernLabelById?.[c] || c
                          : c.concern_label || c.label || c.concern_id || JSON.stringify(c);
                      return <Chip key={`concern-${i}-${idx}`} size="small" variant="outlined" label={label} />;
                    })}
                    {Array.isArray(concerns) && concerns.length > MAX_VISIBLE ? (
                      <Chip
                        key={`concern-more-${i}`}
                        size="small"
                        variant="outlined"
                        clickable
                        onClick={() => toggleExpand(i)}
                        label={isExpanded ? "Show less" : `+${concerns.length - MAX_VISIBLE} more`}
                      />
                    ) : null}
                  </Stack>
                ) : (
                  "—"
                )}
              </TableCell>

              <TableCell>
                {engagement == null ? "—" : (typeof engagement === "number" ? engagement.toFixed(3) : String(engagement))}
              </TableCell>

              <TableCell>
                <Stack direction="row" spacing={1} alignItems="center">
                  <FitnessBar value={fitnessNum} />
                  {fitLabel ? <Typography variant="caption" color="text.secondary">{fitLabel}</Typography> : null}
                </Stack>
              </TableCell>
            </TableRow>
          );
        })}
      </TableBody>
    </Table>
  );
}

/** ===== Main: Phase > Campaigns > Asset Table (from frozen_strategy.phases) ===== */
export default function RCSCampaignSequences({strategy}: {strategy?: StrategyShape;}) 
{
  console.log("RCSCampaignSequences: strategy (raw):", strategy);

  // build label maps from strategy/personas/concerns so component can be self-contained
  function buildLabelMap(strat: any): LabelMap {
    const personaLabelById: Record<string, string> = {};
    const concernLabelById: Record<string, string> = {};

    const personas = strat?.personas ?? strat?.frozen_strategy?.personas ?? [];
    if (Array.isArray(personas)) {
      for (const p of personas) {
        // p may be string id or object { id, label }
        if (typeof p === "string") {
          personaLabelById[p] = p;
          const short = String(p).split(":").slice(-1)[0];
          personaLabelById[short] = personaLabelById[short] || p;
        } else if (p && typeof p === "object") {
          const id = String(p.id ?? p.persona ?? p.label ?? "");
          const label = p.label ?? p.name ?? id;
          if (id) personaLabelById[id] = label;
          const short = String(id).split(":").slice(-1)[0];
          if (short) personaLabelById[short] = personaLabelById[short] || label;
        }
      }
    }

    const concerns = strat?.concerns_flat ?? strat?.frozen_strategy?.concerns_flat ?? strat?.concerns ?? [];
    if (Array.isArray(concerns)) {
      for (const c of concerns) {
        if (typeof c === "string") {
          concernLabelById[c] = c;
        } else if (c && typeof c === "object") {
          const id = String(c.id ?? c.concern_id ?? c.key ?? c.label ?? "");
          const label = c.label ?? c.concern_label ?? c.name ?? id;
          if (id) concernLabelById[id] = label;
          const short = String(id).split(":").slice(-1)[0];
          if (short) concernLabelById[short] = concernLabelById[short] || label;
        }
      }
    }

    return { personaLabelById, concernLabelById };
  }

  function resolvePhases(strat: any): any[] {
    if (!strat) return [];
    // common places the backend might put phases
    const candidates = [
      strat.frozen_strategy?.phases,
      strat.phases,
      strat.frozen_strategy,
      strat.frozen_strategy?.data?.phases,
      strat?.data?.frozen_strategy?.phases,
    ];
    for (const cand of candidates) {
      if (Array.isArray(cand) && cand.length) return cand;
      if (cand && typeof cand === "object" && !Array.isArray(cand)) {
        // keyed object like { "0": {...}, "1": {...} } -> return values
        const vals = Object.values(cand).filter(Boolean);
        if (vals.length) return vals;
      }
    }
    return [];
  }

  const labelMap = buildLabelMap(strategy ?? {});
  const phases = resolvePhases(strategy ?? {});

  // eslint-disable-next-line no-console
  console.log("RCSCampaignSequences: resolved phases length:", phases.length, "sample:", phases[0]);

  if (!phases.length) {
    // show helpful message instead of silently returning
    return (
      <Typography color="text.secondary">
        No phases found in <code>frozen_strategy.phases</code>. Check console for the received strategy shape.
      </Typography>
    );
  }

  return (
    <Stack spacing={2}>
      {phases.map((phase, pi) => {
        const stageLabel = phase.stageLabel || phase.stage || `Phase ${pi + 1}`;
        const tf = phase.timeframe || {};
        const start = tf.startDate || tf.start || "";
        const end = tf.endDate || tf.end || "";
        const personas = Array.isArray(phase.personas) ? phase.personas : [];
        const campaigns = Array.isArray(phase.campaigns) ? phase.campaigns : [];

        return (
          <Card key={phase.phaseIndex ?? pi} variant="outlined" sx={{ overflow: "hidden" }}>
            <CardContent>
              <Stack spacing={2}>
                {/* Phase header */}
                <Stack direction="row" justifyContent="space-between" alignItems="center">
                  <Stack spacing={0.25}>
                    <Typography variant="h6">{stageLabel}</Typography>
                    {(start || end) && (
                      <Typography variant="caption" color="text.secondary">
                        {start} {start && end ? "—" : ""} {end}
                      </Typography>
                    )}
                  </Stack>
                  <Chip size="small" label={`Phase ${phase.phaseIndex ?? pi + 1}`} />
                </Stack>

                <Divider />

                {/* Personas */}
                <Stack spacing={1}>
                  <Typography variant="body2" color="text.secondary">
                    Personas
                  </Typography>
                  <Stack direction="row" spacing={1} flexWrap="wrap">
                    {personas.map((p: string | number, idx: any) => (
                      <Chip
                        key={`${String(p)}::${idx}`}
                        size="small"
                        variant="outlined"
                        label={labelMap.personaLabelById?.[p] || String(p)}
                      />
                    ))}
                    {!personas.length && (
                      <Chip size="small" label="—" variant="outlined" />
                    )}
                  </Stack>
                </Stack>

                {/* Campaigns */}
                <Stack spacing={1}>
                  {campaigns.map((camp: { arsenalTable: any; arsenal_table: any; description: any; title: any; id: any; campaign_id: any; }, ci: number) => {
                    const table = Array.isArray(camp.arsenalTable)
                      ? camp.arsenalTable
                      : camp.arsenal_table || [];
                    const title = camp.description || camp.title || `Campaign ${ci + 1}`;
                    return (
                      <Card key={camp.id || camp.campaign_id || ci} variant="outlined" sx={{ mb: 1 }}>
                        <CardContent>
                          <Stack spacing={1.25}>
                            <Stack direction="row" spacing={1} alignItems="center">
                              <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
                                {title}
                              </Typography>
                            </Stack>

                            <ArsenalTable rows={table} labelMap={labelMap} />
                          </Stack>
                        </CardContent>
                      </Card>
                    );
                  })}
                  {!campaigns.length && (
                    <Typography variant="caption" color="text.secondary">
                      No campaigns in this phase.
                    </Typography>
                  )}
                </Stack>
              </Stack>
            </CardContent>
          </Card>
        );
      })}
    </Stack>
  );
}
