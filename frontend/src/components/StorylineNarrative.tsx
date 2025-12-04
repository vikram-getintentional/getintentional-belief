import { Box, Button, Chip, Paper, Stack, TextField, Typography } from "@mui/material";

export type StorylineSummary = {
  text?: string | null;
  segments?: string[];
  deal_status?: string | null;
  zmot?: string | null;
  meta_line?: string | null;
  needs_zmot?: boolean;
};

export type StorylineStakeholders = {
  early?: string[];
  mid?: string[];
  late?: string[];
};

export type StorylineNode = {
  id: string;
  text: string;
  phase?: string | null;
  source?: string | null;
  persona?: string | null;
  editable?: boolean;
};

export type StorylinePayload = {
  summary?: StorylineSummary;
  stakeholders?: StorylineStakeholders;
  nodes: StorylineNode[];
  narrative?: string[];
};

const PHASE_COLORS: Record<string, string> = {
  early: "success",
  mid: "warning",
  late: "info",
};

const DEFAULT_FALLBACK = "Model is still learning this portion of the journey.";

function sentenceFromNode(
  node?: StorylineNode,
  fallback: string = DEFAULT_FALLBACK
): string {
  if (!node) return fallback;
  const personaPrefix = node.persona ? `${node.persona}: ` : "";
  return `${personaPrefix}${node.text}`;
}

function buildConversionFlow(
  summary: StorylineSummary,
  nodes: StorylineNode[],
  narrative: string[]
) {
  if (!nodes || nodes.length === 0) {
    return [];
  }
  const [first, second, third] = nodes;
  const last = nodes[nodes.length - 1];
  return [
    {
      heading: "Org faced ZMOT",
      text:
        summary.zmot ||
        narrative[0] ||
        sentenceFromNode(first, "The journey kicked off when Finance signaled urgency."),
    },
    {
      heading: "Pain intensified",
      text:
        narrative[0] ||
        sentenceFromNode(first, "Early personas surfaced the pain internally."),
    },
    {
      heading: "Persona's job had to respond",
      text: sentenceFromNode(first, "The first mover took responsibility for the pain."),
    },
    {
      heading: "Persona delegated to the next team",
      text: sentenceFromNode(
        second,
        "They pulled an adjacent persona in to evaluate solutions."
      ),
    },
    {
      heading: "Persona engaged with enablement assets",
      text: sentenceFromNode(
        third,
        "A mid-stage persona consumed key assets to keep belief moving."
      ),
    },
    {
      heading: "Persona cleared or blocked the deal",
      text: sentenceFromNode(
        last,
        "A late-stage persona handled compliance, legal, or procurement."
      ),
    },
  ].filter((entry) => entry.text && entry.text.trim().length > 0);
}

type StorylineNarrativeProps = {
  storyline?: StorylinePayload | null;
  allowEdits?: boolean;
  edits?: Record<string, string>;
  onEdit?: (nodeId: string, value: string) => void;
  loading?: boolean;
  emptyMessage?: string;
};

const StorylineNarrative = ({
  storyline,
  allowEdits = false,
  edits = {},
  onEdit,
  loading = false,
  emptyMessage = "No storyline available yet.",
}: StorylineNarrativeProps) => {
  if (loading) {
    return (
      <Typography variant="body2" color="text.secondary">
        Generating storyline…
      </Typography>
    );
  }

  if (!storyline || !storyline.nodes || storyline.nodes.length === 0) {
    return (
      <Typography variant="body2" color="text.secondary">
        {emptyMessage}
      </Typography>
    );
  }

  const summary = storyline.summary || {};
  const narrative = storyline.narrative || [];
  const stakeholders = storyline.stakeholders || {};
  const conversionFlow = buildConversionFlow(summary, storyline.nodes, narrative);

  const phaseBuckets: Array<{ key: "early" | "mid" | "late"; title: string }> = [
    { key: "early", title: "Phase 1 – Exploration" },
    { key: "mid", title: "Phase 2 – Diagnosis" },
    { key: "late", title: "Phase 3 – Commitment" },
  ];
  const champion = stakeholders.early?.[0] || storyline.nodes[0]?.persona || null;
  const coalition = stakeholders.mid?.slice(0, 2) || [];
  const gatekeeper = stakeholders.late?.[0] || null;

  const hindsightBullets: string[] = [];
  if (gatekeeper) {
    hindsightBullets.push(
      `${gatekeeper} arrived late — preload compliance & legal proof earlier next time.`
    );
  }
  if (storyline.nodes.some((node) => node.source === "inferred")) {
    hindsightBullets.push(
      "Some steps are inferred between observed engagements. Validate these latent personas with the team."
    );
  }
  if (narrative.length > 0) {
    narrative.slice(-2).forEach((sentence) => hindsightBullets.push(sentence));
  }

  return (
    <Stack spacing={2}>
      <Paper variant="outlined" sx={{ p: 2 }}>
        {summary.text && (
          <Typography variant="body2" sx={{ mb: 1 }}>
            {summary.text}
          </Typography>
        )}
        {summary.meta_line && (
          <Typography variant="body2" color="text.secondary">
            {summary.meta_line}
          </Typography>
        )}
        {summary.zmot && (
          <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
            ZMOT: {summary.zmot}
          </Typography>
        )}
        {!summary.zmot && summary.needs_zmot && (
          <Button size="small" sx={{ mt: 1 }}>
            Add a ZMOT for this account
          </Button>
        )}
      </Paper>

      <Paper variant="outlined" sx={{ p: 2 }}>
        <Typography variant="subtitle2" gutterBottom>
          Stakeholders by stage
        </Typography>
        <Stack
          direction={{ xs: "column", md: "row" }}
          spacing={1}
          alignItems={{ xs: "flex-start", md: "center" }}
        >
          {phaseBuckets.map(({ key, title }) => {
            const list = stakeholders[key] || [];
            if (!list.length) return null;
            return (
              <Stack key={key} spacing={0.5}>
                <Typography variant="caption" color="text.secondary">
                  {title}
                </Typography>
                <Stack direction="row" spacing={0.5} sx={{ flexWrap: "wrap" }}>
                  {list.map((persona) => (
                    <Chip key={`${key}-${persona}`} label={persona} size="small" />
                  ))}
                </Stack>
              </Stack>
            );
          })}
        </Stack>
      </Paper>

      {conversionFlow.length > 0 && (
        <Paper variant="outlined" sx={{ p: 2 }}>
          <Typography variant="subtitle2" gutterBottom>
            Conversion story
          </Typography>
          <Stack spacing={0.5}>
            {conversionFlow.map((entry, idx) => (
              <Typography variant="body2" key={`flow-${idx}`}>
                <strong>{entry.heading}:</strong> {entry.text}
              </Typography>
            ))}
          </Stack>
        </Paper>
      )}

      <Paper variant="outlined" sx={{ p: 2 }}>
        <Typography variant="subtitle2" gutterBottom>
          Roles & coalitions
        </Typography>
        <Stack spacing={0.5}>
          {champion && (
            <Typography variant="body2">Champion: {champion}</Typography>
          )}
          {coalition.length > 0 && (
            <Typography variant="body2">
              Critical coalition: {coalition.join(" + ")}
            </Typography>
          )}
          {gatekeeper && (
            <Typography variant="body2">Gatekeeper: {gatekeeper}</Typography>
          )}
        </Stack>
      </Paper>

      {hindsightBullets.length > 0 && (
        <Paper variant="outlined" sx={{ p: 2 }}>
          <Typography variant="subtitle2" gutterBottom>
            Hindsight / next time
          </Typography>
          <Stack spacing={0.5}>
            {hindsightBullets.slice(0, 3).map((line, idx) => (
              <Typography variant="body2" key={`hindsight-${idx}`}>
                • {line}
              </Typography>
            ))}
          </Stack>
        </Paper>
      )}

      {allowEdits && (
        <Paper variant="outlined" sx={{ p: 2 }}>
          <Typography variant="subtitle2" gutterBottom>
            Detailed timeline (editable)
          </Typography>
          <Stack spacing={1.5}>
            {storyline.nodes.map((node) => {
              const textValue = edits[node.id] ?? node.text;
              const canEdit = node.editable && Boolean(onEdit);
              const phaseChip =
                node.phase && PHASE_COLORS[node.phase.toLowerCase()] ? (
                  <Chip
                    size="small"
                    color={PHASE_COLORS[node.phase.toLowerCase()] as any}
                    label={`${node.phase.charAt(0).toUpperCase() + node.phase.slice(1)} phase`}
                  />
                ) : null;

              const personaChip = node.persona ? (
                <Chip size="small" variant="outlined" label={node.persona} />
              ) : null;

              const sourceChip = node.source ? (
                <Chip
                  size="small"
                  variant="outlined"
                  label={node.source === "observed" ? "Observed" : "Predicted"}
                />
              ) : null;

              return (
                <Paper key={node.id} variant="outlined" sx={{ p: 2 }}>
                  <Stack direction="row" spacing={1} sx={{ flexWrap: "wrap", mb: 1 }}>
                    {phaseChip}
                    {personaChip}
                    {sourceChip}
                  </Stack>
                  {canEdit ? (
                    <TextField
                      fullWidth
                      multiline
                      minRows={2}
                      value={textValue}
                      onChange={(event) => onEdit?.(node.id, event.target.value)}
                    />
                  ) : (
                    <Typography variant="body2">{textValue}</Typography>
                  )}
                </Paper>
              );
            })}
          </Stack>
        </Paper>
      )}
    </Stack>
  );
};

export default StorylineNarrative;
