import React from "react";
import {
  Box,
  Chip,
  Paper,
  Stack,
  Tooltip,
  Typography,
  type ChipProps,
} from "@mui/material";
import InfoOutlinedIcon from "@mui/icons-material/InfoOutlined";

export type PersonaEngagementCadenceAsset = {
  asset?: string | null;
  channel?: string | null;
  mode?: string | null;
  expected_delta_bp?: number | null;
  confidence?: number | null;
  funnel_phase?: string | null;
};

export type PersonaEngagementCadenceRow = {
  phase: string;
  duration_days?: number | null;
  frequency_per_week?: number | null;
  assets?: PersonaEngagementCadenceAsset[];
  policy_note?: string | null;
};

export type PersonaEngagementSnapshot = {
  total_touches?: number | null;
  positive_signals?: number | null;
  positive_rate?: number | null;
  last_touch_days?: number | null;
  last_positive_days?: number | null;
  touch_density_per_week?: number | null;
  top_channels?: Array<{ label?: string | null; count?: number | null }>;
  top_sources?: Array<{ label?: string | null; count?: number | null }>;
  fatigue?: number | null;
  fatigue_reason?: string | null;
};

export type PersonaEngagementPerson = {
  person_id?: string | null;
  display_name?: string | null;
  person_involvement_score?: number | null;
  person_belief_level?: number | null;
  committee_probability?: number | null;
  role_band?: string | null;
  belief_phase_probs?: Record<string, number> | null;
  dominant_phase?: string | null;
};

export type PersonaEngagementPlan = {
  persona_id: string;
  persona_label?: string | null;
  journey_phase?: string | null;
  belief_level?: number | null;
  belief_band?: string | null;
  committee_probability?: number | null;
  priority_score?: number | null;
  phase_probs?: Record<string, number> | null;
  dominant_phase?: string | null;
  belief_metrics?: {
    perceptibility?: number | null;
    proximity?: number | null;
    involvement?: number | null;
    activation?: number | null;
  } | null;
  fatigue?: number | null;
  fatigue_reason?: string | null;
  diversify?: boolean;
  diversification_reason?: string | null;
  engagement_state?: string | null;
  state_reason?: string | null;
  base_frequency_per_week?: number | null;
  park_until_signal?: boolean;
  engagement_snapshot?: PersonaEngagementSnapshot | null;
  target_people?: PersonaEngagementPerson[];
  cadence: PersonaEngagementCadenceRow[];
};

type Props = {
  engagements?: PersonaEngagementPlan[];
  personaMatches?: Record<string, string[]>;
  maxAssetsPerPhase?: number;
  emptyCopy?: string;
};

type FunnelBand = "Early" | "Mid" | "Late";

const FUNNEL_MAP: Record<FunnelBand, { color: ChipProps["color"] }> = {
  Early: { color: "info" },
  Mid: { color: "success" },
  Late: { color: "warning" },
};

const STATE_META: Record<
  string,
  { label: string; color: ChipProps["color"]; description: string }
> = {
  probe: {
    label: "Probe cadence",
    color: "default",
    description: "Low-pressure discovery touches to surface signals.",
  },
  ramp: {
    label: "Ramp cadence",
    color: "primary",
    description: "Belief rising—lean in with focused outreach.",
  },
  sustain: {
    label: "Sustain cadence",
    color: "success",
    description: "Belief is strong—maintain steady pressure.",
  },
  decay: {
    label: "Decay cadence",
    color: "warning",
    description: "Dial down touches and monitor for new triggers.",
  },
  park: {
    label: "Parked",
    color: "default",
    description: "On hold until fatigue drops or belief changes.",
  },
};

const PHASE_TOOLTIPS: Record<string, string> = {
  probe: "1 touch every couple of weeks to test for interest.",
  ramp: "2–3 touches per week to accelerate belief change.",
  sustain: "Hold steady with weekly, higher-value touches.",
  decay: "Reduce cadence while monitoring signals.",
  park: "Monitoring only—reactivate once a new trigger fires.",
};

const cadencePhaseToBand = (phase?: string | null): FunnelBand => {
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

const fmtDays = (value?: number | null) => {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  if (value < 1) return "<1 day";
  if (value === 1) return "1 day";
  return `${Math.round(value)} days`;
};

const fmtBasisPoints = (value?: number | null) => {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return `${Number(value).toFixed(1)} bp`;
};

const formatFrequency = (value?: number | null) => {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  const precision = value >= 1 ? 1 : 2;
  return `${value.toFixed(precision)} / wk`;
};

const isSentinelPersona = (personaId?: string | null) =>
  Boolean(personaId && personaId.includes("__STOP__"));

const PersonaEngagementCadence: React.FC<Props> = ({
  engagements,
  personaMatches,
  maxAssetsPerPhase = 3,
  emptyCopy = "Cadence will populate once we map personas to this journey.",
}) => {
  const filteredEngagements = (engagements ?? []).filter(
    (entry) => !isSentinelPersona(entry.persona_id)
  );

  if (!filteredEngagements.length) {
    return (
      <Typography variant="body2" color="text.secondary">
        {emptyCopy}
      </Typography>
    );
  }

  const formatDaysAgo = (value?: number | null) => {
    if (value === null || value === undefined || Number.isNaN(value)) return null;
    if (value < 0.5) return "today";
    if (value < 1.5) return "1 day ago";
    return `${Math.round(value)} days ago`;
  };

  return (
    <Stack spacing={2}>
      {filteredEngagements.map((persona, idx) => {
        const personaKey =
          persona.persona_id || persona.persona_label || `persona-${idx}`;
        const matches = persona.persona_id
          ? personaMatches?.[persona.persona_id]
          : undefined;
        const snapshot = persona.engagement_snapshot || undefined;
        const targetPeople = (persona.target_people && persona.target_people.length
          ? persona.target_people
          : matches?.map((name) => ({ display_name: name })) || []) as PersonaEngagementPerson[];
        const snapshotSummary: string[] = [];
        if (typeof snapshot?.total_touches === "number") {
          snapshotSummary.push(`${snapshot.total_touches} touches`);
        }
        if (typeof snapshot?.positive_rate === "number") {
          snapshotSummary.push(`${fmtPercent(snapshot.positive_rate)} positive`);
        }
        if (typeof snapshot?.last_touch_days === "number") {
          const label = formatDaysAgo(snapshot.last_touch_days);
          if (label) snapshotSummary.push(`Last touch ${label}`);
        }
        if (typeof snapshot?.last_positive_days === "number") {
          const label = formatDaysAgo(snapshot.last_positive_days);
          if (label) snapshotSummary.push(`Positive ${label}`);
        }
        if (typeof snapshot?.touch_density_per_week === "number") {
          snapshotSummary.push(`${snapshot.touch_density_per_week.toFixed(1)} / wk recent`);
        }
        const stateKey = (persona.engagement_state || "").toLowerCase();
        const stateMeta = STATE_META[stateKey];
        const isParked = Boolean(persona.park_until_signal || stateKey === "park");
        const beliefLabel =
          typeof persona.belief_level === "number"
            ? `${titleize(persona.belief_band) || "Belief"} ${fmtPercent(persona.belief_level)}`
            : null;
        const committeeLabel =
          typeof persona.committee_probability === "number"
            ? fmtPercent(persona.committee_probability)
            : null;
        const phaseTooltip = persona.phase_probs
          ? Object.entries(persona.phase_probs)
              .map(([phase, prob]) => `${phase}: ${fmtPercent(prob)}`)
              .join(" · ")
          : null;
        return (
          <Paper
            key={personaKey}
            variant="outlined"
            sx={{
              p: 2,
              opacity: isParked ? 0.75 : 1,
              borderStyle: isParked ? "dashed" : "solid",
            }}
          >
            <Stack
              direction={{ xs: "column", md: "row" }}
              spacing={1}
              justifyContent="space-between"
              alignItems={{ xs: "flex-start", md: "center" }}
            >
              <Typography variant="subtitle1">
                {persona.persona_label || persona.persona_id || "Priority persona"}
              </Typography>
              <Stack direction="row" spacing={1}>
                {persona.journey_phase ? (
                  <Chip
                    size="small"
                    color="default"
                    label={persona.journey_phase}
                  />
                ) : null}
                {stateMeta ? (
                  <Tooltip title={persona.state_reason || stateMeta.description}>
                    <Chip
                      size="small"
                      color={stateMeta.color}
                      variant={stateKey === "park" ? "outlined" : "filled"}
                      label={stateMeta.label}
                    />
                  </Tooltip>
                ) : null}
                {beliefLabel ? (
                  <Chip size="small" color="primary" variant="outlined" label={beliefLabel} />
                ) : null}
                {committeeLabel ? (
                  <Chip
                    size="small"
                    variant="outlined"
                    label={`Committee ${committeeLabel}`}
                  />
                ) : null}
                {persona.dominant_phase ? (
                  <Tooltip title={phaseTooltip || "Dominant belief phase"}>
                    <Chip
                      size="small"
                      variant="outlined"
                      label={`Phase ${persona.dominant_phase}`}
                    />
                  </Tooltip>
                ) : null}
                <Chip
                  size="small"
                  variant="outlined"
                  label={`Fatigue ${fmtPercent(persona.fatigue ?? 0)}`}
                />
                {persona.diversify ? (
                  <Tooltip title={persona.diversification_reason || "Diversify this persona"}>
                    <Chip
                      size="small"
                      color="warning"
                      variant="outlined"
                      label="Diversify"
                    />
                  </Tooltip>
                ) : null}
              </Stack>
            </Stack>

            {persona.fatigue_reason ? (
              <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5 }}>
                {persona.fatigue_reason}
              </Typography>
            ) : null}
            {persona.base_frequency_per_week ? (
              <Typography variant="caption" color="text.secondary" sx={{ display: "block", mt: 0.25 }}>
                Base cadence ~{formatFrequency(persona.base_frequency_per_week)} before phase adjustments.
              </Typography>
            ) : null}
            {persona.state_reason && stateMeta ? (
              <Typography variant="caption" color="text.secondary" sx={{ display: "block", mt: 0.25 }}>
                {persona.state_reason}
              </Typography>
            ) : null}
            {targetPeople.length ? (
              <Stack direction="row" spacing={0.75} sx={{ flexWrap: "wrap", mt: 1 }}>
                {targetPeople.map((person, index) => {
                  const probabilityLabel =
                    typeof person.committee_probability === "number"
                      ? fmtPercent(person.committee_probability)
                      : null;
                  const involvement =
                    typeof person.person_involvement_score === "number"
                      ? ` · ${fmtPercent(person.person_involvement_score)}`
                      : "";
                  const belief =
                    typeof person.person_belief_level === "number"
                      ? ` · belief ${fmtPercent(person.person_belief_level)}`
                      : "";
                  const rolePrefix = person.role_band ? `${titleize(person.role_band)} · ` : "";
                  const personLabel = `${rolePrefix}${person.display_name || "Target"}${
                    probabilityLabel ? ` · ${probabilityLabel}` : ""
                  }${involvement}${belief}`;
                  const personPhaseTooltip = person.belief_phase_probs
                    ? Object.entries(person.belief_phase_probs)
                        .map(([phase, prob]) => `${phase}: ${fmtPercent(prob)}`)
                        .join(" · ")
                    : person.dominant_phase
                    ? `Dominant phase: ${person.dominant_phase}`
                    : undefined;
                  return (
                    <Tooltip
                      key={`${persona.persona_id || "persona"}-${
                        person.person_id || person.display_name || index
                      }`}
                      title={personPhaseTooltip || ""}
                      disableHoverListener={!personPhaseTooltip}
                    >
                      <Chip
                        label={personLabel}
                        size="small"
                        variant="outlined"
                      />
                    </Tooltip>
                  );
                })}
              </Stack>
            ) : null}

            {(snapshotSummary.length > 0 || (snapshot?.top_channels?.length ?? 0) > 0) ? (
              <Stack spacing={0.5} sx={{ mt: 1 }}>
                {snapshotSummary.length ? (
                  <Typography variant="caption" color="text.secondary">
                    {snapshotSummary.join(" · ")}
                  </Typography>
                ) : null}
                {snapshot?.top_channels?.length ? (
                  <Stack direction="row" spacing={0.5} sx={{ flexWrap: "wrap" }}>
                    {snapshot.top_channels.slice(0, 3).map((channel, channelIdx) => (
                      <Chip
                        key={`${persona.persona_id}-channel-${channelIdx}`}
                        size="small"
                        variant="outlined"
                        label={`${channel.label || "Channel"} (${channel.count ?? 0})`}
                      />
                    ))}
                  </Stack>
                ) : null}
              </Stack>
            ) : null}

            <Box
              sx={{
                mt: 1.5,
                display: "grid",
                gap: 1.5,
                gridTemplateColumns: {
                  xs: "1fr",
                  md: "repeat(2, minmax(0, 1fr))",
                  lg: "repeat(3, minmax(0, 1fr))",
                },
              }}
            >
              {(persona.cadence || []).map((phase) => {
                const band = cadencePhaseToBand(phase.phase);
                const chipColor = FUNNEL_MAP[band]?.color ?? "default";
                const assets = (phase.assets || []).slice(0, maxAssetsPerPhase);
                const phaseKey = (phase.phase || "").toLowerCase();
                const phaseTooltip = phase.policy_note || PHASE_TOOLTIPS[phaseKey];
                return (
                  <Paper
                    variant="outlined"
                    sx={{ p: 1.5 }}
                    key={`${persona.persona_id}-${phase.phase}`}
                  >
                    <Stack
                      direction="row"
                      spacing={1}
                      justifyContent="space-between"
                      alignItems="center"
                    >
                      <Stack direction="row" spacing={0.5} alignItems="center">
                        <Typography variant="subtitle2">{titleize(phase.phase)}</Typography>
                        {phaseTooltip ? (
                          <Tooltip title={phaseTooltip}>
                            <InfoOutlinedIcon fontSize="small" color="action" />
                          </Tooltip>
                        ) : null}
                      </Stack>
                      <Chip size="small" color={chipColor} label={`${band} funnel`} />
                    </Stack>
                    <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>
                      {formatFrequency(phase.frequency_per_week)} · {fmtDays(phase.duration_days)}
                    </Typography>
                    {assets.length ? (
                      <Stack spacing={0.5} sx={{ mt: 1 }}>
                        {assets.map((asset, idx) => (
                          <Box key={`${persona.persona_id}-${phase.phase}-asset-${idx}`}>
                              <Typography variant="body2">
                                {asset.asset || "Asset"}{" "}
                                <Typography
                                  component="span"
                                  variant="body2"
                                  color="text.secondary"
                                >
                                  via {asset.channel || "Channel"}
                                  {asset.mode ? ` · ${titleize(asset.mode)}` : ""}
                                </Typography>
                              </Typography>
                              <Typography variant="caption" color="text.secondary">
                                {asset.funnel_phase || `${band} funnel`} · Δ{" "}
                                {fmtBasisPoints(asset.expected_delta_bp)} · Confidence{" "}
                                {fmtPercent(asset.confidence ?? null)}
                              </Typography>
                            </Box>
                          ))}
                        </Stack>
                      ) : (
                        <Typography variant="caption" color="text.secondary" sx={{ display: "block", mt: 1 }}>
                          No asset recommendations yet — keep touches lightweight.
                        </Typography>
                      )}
                    </Paper>
                );
              })}
            </Box>
          </Paper>
        );
      })}
    </Stack>
  );
};

export default PersonaEngagementCadence;
