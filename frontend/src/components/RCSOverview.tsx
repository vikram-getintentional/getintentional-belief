// ============================
// Component: RCSOverview.tsx
// ============================
import React from "react";
import {
  Box, Card, CardContent, Chip, Divider, Grid, Stack, Typography,
  Tooltip,
  Table,
  TableHead,
  TableRow,
  TableCell,
  TableBody
} from "@mui/material";

import { bucketByRole, zNormalizePersonas, classifyRole, PersonaMetric } from "../utils/rcs_metrics";
import { pickPreferredSource } from "../utils/rcs_normalize";



type PersonaRow = { id: string; label?: string; involvement?: number; activation?: number };

function roleOf(p: PersonaRow) {
  const inv = p.involvement ?? 0;
  const act = p.activation ?? 0;
  if (inv >= 0.5 && act >= 0.5) return "Potential Champion";
  if (inv >= 0.5 && act < 0.5)  return "Blocker";
  if (inv < 0.5 && act >= 0.5)  return "Operator";
  return "Passive Influencer";
}

function pct(n?: number) { return `${Math.round(100 * (n ?? 0))}%`; }

export default function RCSOverview({ rcs }: { rcs: any }) {
  if (!rcs) return null;

  const preferred = pickPreferredSource(rcs) || {};
  const personasSrc =
    preferred?.top_personas?.by_strength ||
    preferred?.top_personas ||
    preferred?.frozen_persona_pool ||
    preferred?.all_personas ||
    // also accept payload wrapped under report or legacy keys
    rcs?.report?.top_personas?.by_strength ||
    rcs?.report?.top_personas ||
    rcs?.top_personas ||
    rcs?.all_personas ||
    rcs?.top_N_personas ||
    [];
  const personas: PersonaMetric[] = (Array.isArray(personasSrc) ? personasSrc.map((p: any) => ({
    id: p?.id || p?.persona || p?.label || String(p),
    label: p?.persona_label || p?.label || p?.id || String(p),
    involvement: Number(p?.involvement ?? p?.I ?? 0),
    activation: Number(p?.activation ?? p?.A ?? 0),
  })) : []);
 
  const byRole = bucketByRole(personas, { mode: "z", zxCut: 0, zyCut: 0 });
   personas.forEach(p => byRole[roleOf(p)].push(p));
 

  const win =
    Number(preferred?.graph_win_likelihood ?? preferred?.baseline?.win_likelihood) ||
    Number(rcs?.graph_win_likelihood) ||
    Number(rcs?.baseline?.win_likelihood) ||
    Number(rcs?.concern_sequences?.[0]?.final_win) ||
    0;
 
  const rawCoalitionCandidates = [
    preferred?.concern_coalitions,
    preferred?.coalitions,
    preferred?.coalition_candidates,
    rcs?.coalitions,
    rcs?.concern_coalitions,
    rcs?.top_coalitions,
    rcs?.report?.coalitions,
    rcs?.report?.concern_coalitions,
    rcs?.report?.output?.coalitions,
    rcs?.output?.coalitions,
    (rcs?.report && rcs.report.report && rcs.report.report.coalitions) // nested wrapper
  ];

  const rawCoalitions = rawCoalitionCandidates.find((c) => Array.isArray(c) && c.length) || [];
  // Debug: if UI still shows none, check console for where we looked and a sample
  if (!rawCoalitions.length) {
    // eslint-disable-next-line no-console
    console.debug("RCSOverview: no coalitions found. checked candidates:", rawCoalitionCandidates.map((c) => (Array.isArray(c) ? `arr(${c.length})` : typeof c)));
  } else {
    // eslint-disable-next-line no-console
    console.debug("RCSOverview: using coalitions from source, sample:", rawCoalitions[0]);
  }

  const coalitions: any[] = Array.isArray(rawCoalitions)
    ? rawCoalitions.map((c: any) => {
        // robust lift/synergy resolution
        const lift = Number(
          c?.lift ??
            c?.compatibility ??
            c?.score ??
            c?.compatibility_score ??
            c?.lift_score ??
            0
        );
        const synergy = Number(
          c?.synergy ?? c?.synergy_score ?? c?.overlap_size ?? c?.overlap ?? 0
        );
        // Accept many shapes: concerns, shared_concerns, members, pair, plain strings
        const rawConcerns =
          c?.concerns ??
          c?.shared_concerns ??
          c?.members ??
          c?.pair ??
          c?.members_list ??
          [];

        const concerns = Array.isArray(rawConcerns)
          ? rawConcerns.map((cc: any) => {
              // string element -> persona id only
              if (typeof cc === "string") {
                return { persona: cc, concern_id: "", concern_label: "" };
              }
              // object element -> be permissive about fields
              return {
                persona:
                  cc?.persona ||
                  cc?.persona_id ||
                  cc?.member_persona ||
                  // fallback to coalition pair if present
                  (Array.isArray(c?.pair) && c.pair[0]) ||
                  cc?.id ||
                  "",
                concern_id: cc?.concern_id || cc?.cid || cc?.id || cc?.concern || "",
                concern_label:
                  cc?.concern_label || cc?.label || cc?.name || cc?.concern_text || ""
              };
            })
          : [];

        // Debug: show one sample mapping in console to verify shapes during dev
        // eslint-disable-next-line no-console
        console.debug("RCSOverview: mapped coalition", { raw: c, mapped: concerns.slice(0, 6) });

        return { ...c, lift, synergy, concerns };
      })
    : [];

  return (
    <Stack spacing={2}>
      {/* Meta / Stats */}
      <Card variant="outlined">
        <CardContent>
          <Typography variant="h6">Graph Snapshot</Typography>
          <Divider sx={{ my: 1.5 }} />
          <Stack direction="row" spacing={4} flexWrap="wrap">
            <Stack>
              <Typography variant="body2" color="text.secondary">Graph win likelihood</Typography>
              <Typography variant="h5">{(win as number).toFixed(3)}</Typography>
            </Stack>
            <Stack>
              <Typography variant="body2" color="text.secondary">Personas analyzed</Typography>
              <Typography variant="h5">{personas.length}</Typography>
            </Stack>
            <Stack>
              <Typography variant="body2" color="text.secondary">Coalition ideas</Typography>
              <Typography variant="h5">{coalitions.length}</Typography>
            </Stack>
          </Stack>
        </CardContent>
      </Card>

      {/* Persona roles */}
      <Card variant="outlined">
        <CardContent>
          <Typography variant="h6">Persona Roles (Involvement × Activation)</Typography>
          <Divider sx={{ my: 1.5 }} />
          <Grid container spacing={2}>
            {Object.entries(byRole).map(([role, list]) => (
              <Grid size={{ xs: 12, md: 6, lg: 6 }} key={role}>
                <Card variant="outlined">
                  <CardContent>
                    <Stack spacing={1}>
                      <Stack direction="row" spacing={1} alignItems="center">
                        <Chip
                          color={
                            role === "Potential Champion" ? "success" :
                            role === "Blocker" ? "error" :
                            role === "Operator" ? "info" : "default"
                          }
                          label={role}
                          size="small"
                        />
                        <Typography variant="body2" color="text.secondary">
                          {list.length} persona{list.length !== 1 ? "s" : ""}
                        </Typography>
                      </Stack>

                      <Stack spacing={0.5}>
                        {list.slice(0, 6).map(p => (
                          <Tooltip
                            key={p.id}
                            title={`Involvement ${pct(p.involvement)} · Activation ${pct(p.activation)}`}
                          >
                            <Typography variant="body2">• {p.label || p.id}</Typography>
                          </Tooltip>
                        ))}
                        {list.length > 6 && (
                          <Typography variant="caption" color="text.secondary">
                            +{list.length - 6} more…
                          </Typography>
                        )}
                      </Stack>
                    </Stack>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </CardContent>
      </Card>

      {/* Coalitions */}
      <Card variant="outlined">
        <CardContent>
          <Typography variant="h6">Potential Persona Coalitions</Typography>
          <Divider sx={{ my: 1.5 }} />
          {!coalitions.length ? (
            <Typography color="text.secondary">No coalitions available.</Typography>
          ) : (
            <Stack spacing={1.5}>
              {coalitions.slice(0, 8).map((c: any, i: number) => {
                // build persona id -> label map from normalized personas (prefer human label)
                const personaMap = new Map((personas || []).map((p: any) => [String(p.id), p.label]));

                // derive member ids + labels (preserve ids so extractor can probe persona-keyed lifts/stages)
                let memberIds: string[] = [];
                let memberLabels: string[] = [];
                if (Array.isArray(c.labels) && Array.isArray(c.pair) && c.labels.length && c.pair.length) {
                  // both labels and pair present -> pair contains ids matching labels
                  memberIds = c.pair.map((p: any) => String(p));
                  memberLabels = c.labels.map((l: any) => String(l));
                } else if (Array.isArray(c.pair) && c.pair.length) {
                  memberIds = c.pair.map((p: any) => String(p));
                  memberLabels = memberIds.map((id) => {
                    const short = id.split(":").slice(-1)[0];
                    return personaMap.get(id) ?? personaMap.get(short) ?? short;
                  });
                } else {
                  // fallback: collect persona ids from shared_concerns / concerns
                  const collector = new Set<string>();
                  const sharedArr = Array.isArray(c.shared_concerns) ? c.shared_concerns : (Array.isArray(c.concerns) ? c.concerns : []);
                  if (Array.isArray(sharedArr) && sharedArr.length) {
                    sharedArr.forEach((sc: any) => {
                      if (typeof sc === "string") collector.add(sc);
                      else {
                        if (sc?.persona) collector.add(String(sc.persona));
                        if (sc?.persona_id) collector.add(String(sc.persona_id));
                        if (sc?.member_persona) collector.add(String(sc.member_persona));
                        if (Array.isArray(sc?.pair)) sc.pair.forEach((pid: any) => collector.add(String(pid)));
                      }
                    });
                  }
                  if (!collector.size && Array.isArray(c.pair)) c.pair.forEach((p: any) => collector.add(String(p)));
                  memberIds = Array.from(collector);
                  memberLabels = memberIds.map((id) => {
                    const short = String(id).split(":").slice(-1)[0];
                    return personaMap.get(id) ?? personaMap.get(short) ?? short;
                  });
                }
                // dev log to verify we resolved meaningful ids+labels
                // eslint-disable-next-line no-console
                console.debug("RCSOverview: memberIds resolved:", memberIds, "memberLabels:", memberLabels);

                const shared = Array.isArray(c.shared_concerns) ? c.shared_concerns : (Array.isArray(c.concerns) ? c.concerns : []);

                // build member rows: persona_label | stage | lift_proxy (avg over shared concerns)
                function extractStageAndLiftFromShared(sc: any, memberIdx: number, memberId?: string) {
                  // stage candidates
                  const stageCandidates = [
                    sc?.stage,
                    sc?.stage_a,
                    sc?.stage_b,
                    sc?.stg,
                    sc?.stage0,
                    sc?.stage1,
                    sc?.phase,
                    sc?.phase_a,
                    sc?.phase_b,
                  ].filter(Boolean);

                  // lift candidates: explicit per-side keys
                  const explicit =
                    memberIdx === 0
                      ? (sc?.lift_a ?? sc?.liftA ?? sc?.lift_0 ?? sc?.lift_1)
                      : (sc?.lift_b ?? sc?.liftB ?? sc?.lift_1 ?? sc?.lift_2);

                  // nested lifts objects/arrays
                  let nested: any = undefined;
                  if (!explicit && sc?.lifts && typeof sc.lifts === "object") {
                    if (memberId && sc.lifts[memberId] != null) nested = sc.lifts[memberId];
                    else if (Array.isArray(sc.lifts) && sc.lifts[memberIdx] != null) nested = sc.lifts[memberIdx];
                    else if (sc.lifts.a != null || sc.lifts.b != null) nested = memberIdx === 0 ? sc.lifts.a : sc.lifts.b;
                  }

                  // fallback generic numeric fields (lift, lift_proxy, lift_score, score, compatibility)
                  const fallbackCandidates = [
                    sc?.lift,
                    sc?.lift_proxy,
                    sc?.lift_score,
                    sc?.score,
                    sc?.compatibility,
                    sc?.compatibility_score,
                    nested,
                    explicit,
                  ];

                  // pick first numeric candidate
                  const pickNumeric = (vals: any[]) => {
                    for (const v of vals) {
                      if (v == null) continue;
                      if (typeof v === "number") return v;
                      if (typeof v === "string" && v.trim()) {
                        const n = Number(v);
                        if (!Number.isNaN(n)) return n;
                      }
                    }
                    // as last resort, search object properties for numeric-looking values
                    for (const v of vals) {
                      if (v && typeof v === "object") {
                        for (const k of Object.keys(v)) {
                          const val = v[k];
                          if (typeof val === "number") return val;
                          if (typeof val === "string" && val.trim()) {
                            const n = Number(val);
                            if (!Number.isNaN(n)) return n;
                          }
                        }
                      }
                    }
                    return undefined;
                  };

                  if (!stageCandidates.length && memberId) {
                    const short = String(memberId).split(":").slice(-1)[0];
                    if (sc?.stages && typeof sc.stages === "object") {
                      const s = sc.stages[memberId] ?? sc.stages[short];
                      if (s) stageCandidates.push(s);
                    }
                    if (sc?.stage_by_persona && typeof sc.stage_by_persona === "object") {
                      const s2 = sc.stage_by_persona[memberId] ?? sc.stage_by_persona[short];
                      if (s2) stageCandidates.push(s2);
                    }
                  }
                  const stage = stageCandidates.length ? String(stageCandidates[0]) : undefined;
                  const lift = pickNumeric(fallbackCandidates);
                  return { stage, lift: typeof lift === "number" ? lift : undefined };
                }

                const memberRows = memberLabels.map((lab: string, idx: number) => {
                  const lifts: number[] = [];
                  const stages = new Set<string>();
                  for (const sc of shared) {
                    const memberIdGuess = memberIds[idx]; // pass the resolved persona id so extractor can find persona-specific fields
                    const { stage, lift } = extractStageAndLiftFromShared(sc, idx, memberIdGuess);
                    console.debug("RCSOverview: shared concern pick", { sc, memberIdx: idx, memberId: memberIdGuess, stage, lift });
                    if (stage) stages.add(stage);
                    if (typeof lift === "number" && !Number.isNaN(lift)) lifts.push(lift);
                  }
                  const avgLift = lifts.length ? lifts.reduce((s, n) => s + n, 0) / lifts.length : 0;
                  const stageText = Array.from(stages).join(", ") || (c?.stage || "");
                  return { persona_label: lab, stage: stageText, lift_proxy: avgLift };
                });

                return (
                  <Card key={i} variant="outlined">
                    <CardContent>
                      <Stack spacing={1}>
                        <Stack direction="row" spacing={2} alignItems="center" justifyContent="space-between">
                          <Typography variant="subtitle1">{c?.label || c?.coalition_id || `Coalition ${i+1}`}</Typography>
                          <Stack direction="row" spacing={1} alignItems="center">
                            <Chip color="primary" size="small" label={`Lift ${(c.lift ?? 0).toFixed(3)}`} />
                            <Chip size="small" label={`Synergy ${(c.synergy ?? 0).toFixed(3)}`} />
                          </Stack>
                        </Stack>

                        <Table size="small">
                          <TableHead>
                            <TableRow>
                              <TableCell>Persona</TableCell>
                              <TableCell>Stage</TableCell>
                              <TableCell>Lift</TableCell>
                            </TableRow>
                          </TableHead>
                          <TableBody>
                            {memberRows.map((mr, j) => (
                              <TableRow key={j}>
                                <TableCell>{mr.persona_label}</TableCell>
                                <TableCell>{mr.stage || "—"}</TableCell>
                                <TableCell>{(mr.lift_proxy ?? 0).toFixed(6)}</TableCell>
                              </TableRow>
                            ))}
                          </TableBody>
                        </Table>
                      </Stack>
                    </CardContent>
                  </Card>
                );
              })}
            </Stack>
          )}
        </CardContent>
      </Card>
    </Stack>
  );
}
