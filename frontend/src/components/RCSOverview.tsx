import React, { useMemo, useState } from "react";
import {
  Box,
  Card,
  CardContent,
  Chip,
  Divider,
  Grid,
  LinearProgress,
  Stack,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  Tooltip,
  Typography,
} from "@mui/material";

/** -------- helpers -------- */
function pct(n?: number) {
  const v = Number.isFinite(n as number) ? (n as number) : 0;
  return `${Math.round(v * 100)}%`;
}
function safe<T>(v: T | undefined | null, d: T): T {
  return v ?? d;
}
function truncate(str: string, n = 80) {
  if (!str) return "";
  return str.length > n ? str.slice(0, n - 1) + "…" : str;
}

function median(xs: number[]) {
  if (!xs.length) return 0;
  const a = [...xs].sort((a, b) => a - b);
  const mid = Math.floor(a.length / 2);
  return a.length % 2 ? a[mid] : (a[mid - 1] + a[mid]) / 2;
}

/** Extractors that tolerate old/new report shapes */
function pickReport(rcs: any) {
  return rcs?.frozen_strategy?.report || rcs?.report || {};
}
function pickFrozenPersonas(rcs: any) {
  // canonical: frozen_strategy.frozen_personas (object keyed by persona_id)
  const fp = rcs?.frozen_strategy?.frozen_personas;
  if (fp && typeof fp === "object") return Object.values(fp);
  // older shapes fallbacks
  return rcs?.frozen_strategy?.personas || rcs?.personas || [];
}
function pickGraphwin(rcs: any) {
  const report = pickReport(rcs);
  return Number(report?.graphwin ?? 0);
}
function pickOrgCoalitions(rcs: any) {
  const report = pickReport(rcs);
  return Array.isArray(report?.org_coalitions) ? report.org_coalitions : [];
}
function pickTotalCampaigns(rcs: any) {
  const t = rcs?.tactical_update?.next_campaigns;
  if (Array.isArray(t)) return t.length;
  // fallback: 1 if stage plan exists in tactical_update
  return Array.isArray(rcs?.tactical_update?.phases) ? rcs.tactical_update.phases.length : 0;
}

type Persona = {
  id?: string;
  persona?: string;
  persona_label?: string;
  label?: string;
  involvement?: number;
  activation?: number;
  strength?: number;
};

type Coalition = {
  coalition_id?: string;
  dominant_phase?: string;
  members?: any[];
  concerns?: any[];
  org_rank_score?: number;
  keyness_score?: number;
  stats?: any;
};

/** Build the 2x2 buckets from frozen personas using median thresholds */
function bucketRoles(personas: Persona[]) {
  const I = personas.map((p) => Number(p.involvement ?? 0));
  const A = personas.map((p) => Number(p.activation ?? 0));
  const iCut = median(I);
  const aCut = median(A);

  const labelOf = (p: Persona) =>
    p.persona_label || p.label || p.persona || p.id || "—";

  const champions: string[] = [];
  const blockers: string[] = [];
  const operators: string[] = [];
  const influencers: string[] = [];

  personas.forEach((p) => {
    const inv = Number(p.involvement ?? 0);
    const act = Number(p.activation ?? 0);
    const name = labelOf(p);

    if (inv >= iCut && act >= aCut) champions.push(name);
    else if (inv >= iCut && act < aCut) blockers.push(name);
    else if (inv < iCut && act >= aCut) operators.push(name);
    else influencers.push(name);
  });

  return { champions, blockers, operators, influencers, iCut, aCut };
}

/** Coalition render helpers for new shape */
function coalitionMemberLabels(c: Coalition): string[] {
  const src = Array.isArray(c?.members) ? c.members : [];
  const names = src
    .map((m) => {
      if (m?.persona_label) return m.persona_label;
      if (m?.persona) return m.persona;
      if (m?.label) return m.label;
      return typeof m === "string" ? m : "";
    })
    .filter(Boolean);
  // de-dupe, keep order
  return Array.from(new Set(names));
}
function coalitionConcernLabels(c: Coalition): string[] {
  // primary source: coalition.concerns[]
  if (Array.isArray(c?.concerns) && c.concerns.length) {
    return c.concerns
      .map((x) => x?.label || x?.concern_label || (typeof x === "string" ? x : ""))
      .filter(Boolean);
  }
  // fallback: collect member.concern_label
  const fromMembers =
    Array.isArray(c?.members) &&
    c.members
      .map((m: any) => m?.concern_label)
      .filter(Boolean);
  return (fromMembers || []).slice(0, 6); // cap
}

/** ---------- Component ---------- */
export default function RCSOverview({ rcs }: { rcs: any }) {
  const graphwin = pickGraphwin(rcs);
  const frozenPersonas: Persona[] = pickFrozenPersonas(rcs);
  const coalitions: Coalition[] = pickOrgCoalitions(rcs);
  const totalCampaigns = pickTotalCampaigns(rcs);

  const roles = useMemo(() => bucketRoles(frozenPersonas), [frozenPersonas]);

  return (
    <Stack spacing={3}>
      <Typography variant="h5" sx={{ fontWeight: 600 }}>
        Overview
      </Typography>

      {/* Snapshot */}
      <Card variant="outlined" sx={{ borderRadius: 2 }}>
        <CardContent>
          <Typography variant="subtitle1" sx={{ mb: 2, fontWeight: 600 }}>
            Snapshot
          </Typography>
          <Grid container spacing={2}>
            <Grid size={{ xs: 12, sm: 3 }}>
              <KPI label="Graph Win Likelihood" value={pct(graphwin)} helper="" />
            </Grid>
            <Grid size={{ xs: 12, sm: 3 }}>
              <KPI label="Personas to Analyze" value={frozenPersonas.length} />
            </Grid>
            <Grid size={{ xs: 12, sm: 3 }}>
              <KPI label="Coalitions to Convert" value={coalitions.length} />
            </Grid>
            <Grid size={{ xs: 12, sm: 3 }}>
              <KPI label="Total Campaigns" value={totalCampaigns} />
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      {/* Persona Roles */}
      <Card variant="outlined" sx={{ borderRadius: 2 }}>
        <CardContent>
          <Typography variant="subtitle1" sx={{ mb: 2, fontWeight: 600 }}>
            Persona Roles (Involvement × Activation)
          </Typography>

          <Grid container spacing={2}>
            {/* Champions (High x High) */}
            <Grid size={12}>
              <RoleBox title="Potential Champion (High I × High A)" color="success" items={roles.champions} />
            </Grid>

            {/* Blockers & Operators side-by-side for compactness */}
            <Grid size={{ xs: 12, md: 6 }}>
              <RoleBox title="Blocker (High I × Low A)" color="error" items={roles.blockers} />
            </Grid>
            <Grid size={{ xs: 12, md: 6 }}>
              <RoleBox title="Operator (Low I × High A)" color="info" items={roles.operators} />
            </Grid>

            {/* Influencers */}
            <Grid size={12}>
              <RoleBox title="Passive Influencer (Low I × Low A)" color="default" items={roles.influencers} />
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      {/* Potential Coalitions */}
      <Card variant="outlined" sx={{ borderRadius: 2 }}>
        <CardContent>
          <Typography variant="subtitle1" sx={{ mb: 2, fontWeight: 600 }}>
            Potential Coalitions
          </Typography>

          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>Coalition</TableCell>
                <TableCell>Concerns</TableCell>
                <TableCell align="right">Org Rank</TableCell>
                <TableCell align="right">Keyness</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {coalitions.map((c, idx) => {
                const people = coalitionMemberLabels(c);
                const concerns = coalitionConcernLabels(c);
                return (
                  <TableRow key={c.coalition_id || idx}>
                    <TableCell sx={{ maxWidth: 520 }}>
                      <Stack direction="row" spacing={1} useFlexGap flexWrap="wrap">
                        {people.length === 0 ? (
                          <Muted>—</Muted>
                        ) : (
                          people.map((p) => (
                            <Chip
                              key={p + idx}
                              size="small"
                              label={truncate(p, 40)}
                              sx={{ bgcolor: "success.main", color: "white" }}
                            />
                          ))
                        )}
                      </Stack>
                    </TableCell>
                    <TableCell sx={{ maxWidth: 520 }}>
                      <Stack direction="row" spacing={1} useFlexGap flexWrap="wrap">
                        {concerns.length === 0 ? (
                          <Muted>—</Muted>
                        ) : (
                          concerns.map((lab, i) => (
                            <Tooltip key={lab + i} title={lab}>
                              <Chip size="small" variant="outlined" label={truncate(lab, 36)} />
                            </Tooltip>
                          ))
                        )}
                      </Stack>
                    </TableCell>
                    <TableCell align="right" sx={{ minWidth: 120 }}>
                      <Bar
                        value={
                          typeof c?.org_rank_score === "number"
                            ? c.org_rank_score
                            : typeof c?.stats?.avg_perc === "number"
                            ? c.stats.avg_perc
                            : 0
                        }
                        mode="bps"
                      />

                    </TableCell>
                    <TableCell align="right" sx={{ minWidth: 120 }}>
                      <Bar
                        value={
                          typeof c?.stats?.avg_key === "number"
                            ? c.stats.avg_key
                            : safe(c.keyness_score, 0)
                        }
                      />
                    </TableCell>

                  </TableRow>
                );
              })}
              {coalitions.length === 0 && (
                <TableRow>
                  <TableCell colSpan={4}>
                    <Muted>No coalitions returned for this scenario.</Muted>
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </Stack>
  );
}

/** --------- small presentational bits ---------- */
function KPI({ label, value, helper }: { label: string; value: React.ReactNode; helper?: string }) {
  return (
    <Box
      sx={{
        px: 2,
        py: 1.5,
        borderRadius: 2,
        border: "1px solid",
        borderColor: "divider",
        bgcolor: "background.paper",
      }}
    >
      <Typography sx={{ fontSize: 28, fontWeight: 700, lineHeight: 1 }}>{value}</Typography>
      <Typography variant="body2" color="text.secondary">
        {label}
      </Typography>
      {helper && (
        <Typography variant="caption" color="text.disabled">
          {helper}
        </Typography>
      )}
    </Box>
  );
}

function RoleBox({
  title,
  items,
  color,
}: {
  title: string;
  items: string[];
  color: "success" | "error" | "info" | "default";
}) {
  const palette =
    color === "success"
      ? { bg: "success.main", fg: "common.white", variant: "filled" as const }
      : color === "error"
      ? { bg: "error.main", fg: "common.white", variant: "filled" as const }
      : color === "info"
      ? { bg: "info.main", fg: "common.white", variant: "filled" as const }
      : { bg: "grey.200", fg: "text.primary", variant: "outlined" as const };

  return (
    <Box
      sx={{
        p: 2,
        borderRadius: 2,
        border: "1px solid",
        borderColor: "divider",
      }}
    >
      <Typography variant="subtitle2" sx={{ mb: 1.25, color: "text.secondary" }}>
        {title}
      </Typography>
      <Stack direction="row" spacing={1} useFlexGap flexWrap="wrap">
        {items.length === 0 ? (
          <Muted>—</Muted>
        ) : (
          items.map((t, i) => (
            <Chip
              key={t + i}
              size="small"
              label={t}
              sx={palette.variant === "filled" ? { bgcolor: palette.bg, color: palette.fg } : {}}
              variant={palette.variant}
            />
          ))
        )}
      </Stack>
    </Box>
  );
}

function Bar({ value, mode = "percent" }: { value: number; mode?: "percent" | "bps" }) {
  const v = Number(value || 0);

  if (mode === "bps") {
    // convert float (0.0061) → 61 bps
    const bps = v * 10000;
    const bounded = Math.min(100, bps / 100); // simple scale bar visually up to 100 bps
    return (
      <Stack alignItems="flex-end" spacing={0.5}>
        <Typography variant="caption" color="text.secondary">
          {bps.toFixed(0)} bps
        </Typography>
        <LinearProgress
          variant="determinate"
          value={bounded}
          sx={{ width: 120, height: 6, borderRadius: 4 }}
        />
      </Stack>
    );
  }

  // default percent view
  const pct = Math.max(0, Math.min(1, v));
  return (
    <Stack alignItems="flex-end" spacing={0.5}>
      <Typography variant="caption" color="text.secondary">
        {Math.round(pct * 100)}%
      </Typography>
      <LinearProgress
        variant="determinate"
        value={pct * 100}
        sx={{ width: 120, height: 6, borderRadius: 4 }}
      />
    </Stack>
  );
}


function Muted({ children }: { children: React.ReactNode }) {
  return (
    <Typography variant="body2" color="text.disabled">
      {children}
    </Typography>
  );
}
