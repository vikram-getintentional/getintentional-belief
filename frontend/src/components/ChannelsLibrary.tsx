import { useState } from "react";
import {
  Box,
  Button,
  Card,
  CardActions,
  CardContent,
  CardHeader,
  Chip,
  Grid,
  Stack,
  Typography,
} from "@mui/material";

type Channel = {
  id: string;
  name: string;
  slug?: string;
  channel_type?: string | null;
  delivery_mode?: string | null;
  reach_score_estimate?: number | null;
  metadata_complete?: boolean;
  notes?: string | null;
  impacts?: Array<{
    asset_id: string;
    belief_transition_id?: string | null;
    impact_strength?: number | null;
    evidence_count?: number | null;
  }>;
  target_personas?: string[] | null;
  target_account_segments?: string[] | null;
  target_belief_stages?: string[] | null;
  target_concerns?: string[] | null;
  usage?: {
    total_engagements: number;
    personas?: Array<{ id: string; label?: string | null; count: number }>;
    industries?: Array<{ label: string; count: number }>;
    stages?: Array<{ code?: string | null; label: string; count: number }>;
    concerns?: Array<{ label: string; count: number }>;
    segments?: Array<{ label: string; count: number }>;
    accounts?: Array<{
      id: string;
      name?: string | null;
      industry?: string | null;
      revenue_range?: string | null;
      employee_range?: string | null;
      geography?: string | null;
      funding_stage?: string | null;
    }>;
  };
  typical_assets?: Array<{ id: string; label?: string | null }> | null;
  approval_status?: string | null;
  derived_metadata?: Record<string, any> | null;
  auto_classification_confidence?: number | null;
};

interface ChannelsLibraryProps {
  channels: Channel[];
  onEdit: (channel: Channel) => void;
  onApprove: (channel: Channel) => void;
}

const toTitleCase = (value: string) =>
  value
    .replace(/[_\s]+/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .split(" ")
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");

const formatChannelLabel = (value?: string | null) => {
  if (!value) return "";
  return toTitleCase(value);
};

const summarizeAccounts = (
  accounts?: Array<{ name?: string | null; industry?: string | null; id: string }>,
  limit = 3
) => {
  if (!accounts || accounts.length === 0) return null;
  const names = accounts
    .map((acc) => acc.name || acc.industry || acc.id)
    .filter((value): value is string => Boolean(value));
  if (!names.length) return null;
  const preview = names.slice(0, limit);
  const remaining = names.length - preview.length;
  return remaining > 0 ? `${preview.join(", ")} +${remaining}` : preview.join(", ");
};

const ChannelsLibrary = ({ channels, onEdit, onApprove }: ChannelsLibraryProps) => {
  const [expandedChannelId, setExpandedChannelId] = useState<string | null>(null);
  if (!channels.length) {
    return (
      <Grid container spacing={2}>
        <Grid size={12}>
          <Typography color="text.secondary">No channels found.</Typography>
        </Grid>
      </Grid>
    );
  }

  const renderChannelCard = (channel: Channel) => {
    const metadataComplete = Boolean(channel.metadata_complete);
    const needsApproval =
      (channel.approval_status || "pending").toLowerCase() !== "approved";
    const impacts = channel.impacts ?? [];
    const evidenceCount = impacts.reduce(
      (sum, impact) => sum + (impact.evidence_count || 0),
      0
    );
    const avgStrength =
      impacts.length > 0
        ? impacts.reduce((sum, impact) => sum + (impact.impact_strength || 0), 0) /
          impacts.length
        : null;
    const topAssets = channel.typical_assets?.slice(0, 3) ?? [];
    const isExpanded = expandedChannelId === channel.id;

    return (
      <Grid key={channel.id} size={{ xs: 12, md: 6 }}>
        <Card
          variant="outlined"
          sx={{ height: "100%", display: "flex", flexDirection: "column", gap: 1 }}
        >
          <CardHeader
            title={
              <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap" useFlexGap>
                <Typography variant="h6">{channel.name}</Typography>
                <Chip
                  size="small"
                  color={metadataComplete ? "success" : "warning"}
                  variant={metadataComplete ? "outlined" : "filled"}
                  label={metadataComplete ? "Metadata complete" : "Needs metadata"}
                />
                {needsApproval ? (
                  <Chip size="small" color="warning" label="Needs approval" />
                ) : (
                  <Chip size="small" color="success" variant="outlined" label="Approved" />
                )}
              </Stack>
            }
            subheader={`Channel ID: ${channel.id}${channel.slug ? ` • Slug: ${channel.slug}` : ""}`}
          />
          <CardContent sx={{ flexGrow: 1, display: "flex", flexDirection: "column", gap: 1.5 }}>
            <Stack direction="row" spacing={0.75} flexWrap="wrap" useFlexGap>
              {channel.channel_type && (
                <Chip
                  label={formatChannelLabel(channel.channel_type)}
                  size="small"
                  color="primary"
                />
              )}
              {channel.delivery_mode && (
                <Chip
                  label={formatChannelLabel(channel.delivery_mode)}
                  size="small"
                  color="secondary"
                />
              )}
            </Stack>

            {channel.reach_score_estimate !== null &&
              channel.reach_score_estimate !== undefined && (
                <Typography variant="body2">
                  Reach score: {channel.reach_score_estimate.toFixed(2)}
                </Typography>
              )}

            {channel.notes && (
              <Typography variant="body2" color="text.secondary">
                Notes: {channel.notes}
              </Typography>
            )}

            {needsApproval && channel.derived_metadata && (
              <Box sx={{ borderRadius: 1, border: 1, borderColor: "warning.light", p: 1 }}>
                <Typography variant="caption" color="warning.main">
                  Auto classification
                </Typography>
                {channel.derived_metadata.channel?.label && (
                  <Typography variant="body2" color="text.secondary">
                    Suggested channel: {channel.derived_metadata.channel.label}
                  </Typography>
                )}
                {typeof channel.auto_classification_confidence === "number" && (
                  <Typography variant="caption" color="text.secondary">
                    Confidence {(channel.auto_classification_confidence * 100).toFixed(0)}%
                  </Typography>
                )}
              </Box>
            )}

            {topAssets.length > 0 && (
              <Box>
                <Typography variant="caption" color="text.secondary">
                  Top assets
                </Typography>
                <Stack direction="row" spacing={0.75} flexWrap="wrap" useFlexGap sx={{ mt: 0.5 }}>
                  {topAssets.map((asset) => (
                    <Chip
                      key={`${channel.id}-asset-${asset.id}`}
                      size="small"
                      variant="outlined"
                      label={asset.label || asset.id}
                    />
                  ))}
                </Stack>
              </Box>
            )}

            <Stack direction="row" spacing={2} flexWrap="wrap">
              <Typography variant="caption" color="text.secondary">
                Evidence: {evidenceCount}
              </Typography>
              {avgStrength !== null && (
                <Typography variant="caption" color="text.secondary">
                  Avg Δ log p: {avgStrength.toFixed(3)}
                </Typography>
              )}
            </Stack>

            {channel.target_personas && channel.target_personas.length > 0 && (
              <Stack direction="row" spacing={0.75} flexWrap="wrap" useFlexGap>
                {channel.target_personas.map((persona) => (
                  <Chip
                    key={`${channel.id}-persona-${persona}`}
                    size="small"
                    color="info"
                    variant="outlined"
                    label={persona}
                  />
                ))}
              </Stack>
            )}

            {channel.target_account_segments && channel.target_account_segments.length > 0 && (
              <Stack direction="row" spacing={0.75} flexWrap="wrap" useFlexGap>
                {channel.target_account_segments.map((segment) => (
                  <Chip
                    key={`${channel.id}-segment-${segment}`}
                    size="small"
                    variant="outlined"
                    label={segment}
                  />
                ))}
              </Stack>
            )}

            {channel.target_belief_stages && channel.target_belief_stages.length > 0 && (
              <Stack direction="row" spacing={0.75} flexWrap="wrap" useFlexGap>
                {channel.target_belief_stages.map((stage) => (
                  <Chip
                    key={`${channel.id}-stage-${stage}`}
                    size="small"
                    variant="outlined"
                    label={stage}
                  />
                ))}
              </Stack>
            )}

            {channel.typical_assets && channel.typical_assets.length > 0 && (
              <Box>
                <Typography variant="caption" color="text.secondary" sx={{ display: "block", mb: 0.5 }}>
                  Typical assets
                </Typography>
                <Stack direction="row" spacing={0.75} flexWrap="wrap" useFlexGap>
                  {channel.typical_assets.map((asset) => (
                    <Chip
                      key={`${channel.id}-typical-asset-${asset.id}`}
                      size="small"
                      variant="outlined"
                      color="secondary"
                      label={asset.label || asset.id}
                    />
                  ))}
                </Stack>
              </Box>
            )}

            {channel.usage && channel.usage.total_engagements > 0 && (
              <Box sx={{ borderRadius: 1, border: 1, borderColor: "divider", p: 1 }}>
                <Stack spacing={0.75}>
                  <Typography variant="subtitle2">Usage</Typography>
                  <Typography variant="caption" color="text.secondary">
                    Engagements tracked: {channel.usage.total_engagements}
                  </Typography>
                  {channel.usage.personas && channel.usage.personas.length > 0 && (
                    <Stack direction="row" spacing={0.75} flexWrap="wrap" useFlexGap>
                      {channel.usage.personas.slice(0, 4).map((persona) => (
                        <Chip
                          key={`${channel.id}-usage-persona-${persona.id}`}
                          size="small"
                          variant="outlined"
                          label={`${persona.label || persona.id} (${persona.count})`}
                        />
                      ))}
                    </Stack>
                  )}
                  {channel.usage.industries && channel.usage.industries.length > 0 && (
                    <Typography variant="caption" color="text.secondary">
                      Account types:{" "}
                      {channel.usage.industries
                        .slice(0, 3)
                        .map((entry) => `${entry.label} (${entry.count})`)
                        .join(", ")}
                      {channel.usage.industries.length > 3
                        ? ` +${channel.usage.industries.length - 3}`
                        : ""}
                    </Typography>
                  )}
                  {channel.usage.stages && channel.usage.stages.length > 0 && (
                    <Stack direction="row" spacing={0.75} flexWrap="wrap" useFlexGap>
                      {channel.usage.stages.slice(0, 4).map((stage) => (
                        <Chip
                          key={`${channel.id}-usage-stage-${stage.code || stage.label}`}
                          size="small"
                          variant="outlined"
                          color="success"
                          label={`${stage.label} (${stage.count})`}
                        />
                      ))}
                    </Stack>
                  )}
                  {channel.usage.concerns && channel.usage.concerns.length > 0 && (
                    <Stack direction="row" spacing={0.75} flexWrap="wrap" useFlexGap>
                      {channel.usage.concerns.slice(0, 6).map((concern) => (
                        <Chip
                          key={`${channel.id}-usage-concern-${concern.label}`}
                          size="small"
                          variant="outlined"
                          color="warning"
                          label={`${concern.label} (${concern.count})`}
                        />
                      ))}
                    </Stack>
                  )}
                  {summarizeAccounts(channel.usage.accounts) && (
                    <Typography variant="caption" color="text.secondary">
                      Accounts: {summarizeAccounts(channel.usage.accounts)}
                    </Typography>
                  )}
                  {channel.usage.segments && channel.usage.segments.length > 0 && (
                    <Stack direction="row" spacing={0.75} flexWrap="wrap" useFlexGap>
                      {channel.usage.segments.slice(0, 6).map((segment) => (
                        <Chip
                          key={`${channel.id}-usage-segment-${segment.label}`}
                          size="small"
                          variant="outlined"
                          label={`${segment.label} (${segment.count})`}
                        />
                      ))}
                    </Stack>
                  )}
                </Stack>
              </Box>
            )}
          </CardContent>
          <CardActions sx={{ justifyContent: "space-between", px: 2, pb: 2 }}>
            <Stack direction="row" spacing={1}>
              <Button size="small" variant="outlined" onClick={() => onEdit(channel)}>
                Edit metadata
              </Button>
              {needsApproval && (
                <Button size="small" color="success" onClick={() => onApprove(channel)}>
                  Approve
                </Button>
              )}
            </Stack>
            <Button
              size="small"
              onClick={() => setExpandedChannelId(isExpanded ? null : channel.id)}
            >
              {isExpanded ? "Hide reasoning" : "View reasoning"}
            </Button>
          </CardActions>
          {isExpanded && (
            <Box sx={{ px: 3, pb: 2 }}>
              <Stack spacing={1}>
                {channel.target_personas && channel.target_personas.length > 0 && (
                  <Typography variant="body2" color="text.secondary">
                    Personas: {channel.target_personas.slice(0, 4).join(", ")}
                  </Typography>
                )}
                {channel.target_belief_stages && channel.target_belief_stages.length > 0 && (
                  <Typography variant="body2" color="text.secondary">
                    Belief focus: {channel.target_belief_stages.join(", ")}
                  </Typography>
                )}
              </Stack>
            </Box>
          )}
        </Card>
      </Grid>
    );
  };

  const sortByName = (list: Channel[]) =>
    [...list].sort((a, b) => a.name.localeCompare(b.name, undefined, { sensitivity: "base" }));

  const groups: Array<{ key: string; title: string; items: Channel[] }> = [];
  const seen = new Set<string>();

  const needsMetadata = channels.filter((channel) => !channel.metadata_complete);
  if (needsMetadata.length) {
    groups.push({
      key: "needs-metadata",
      title: "Needs metadata",
      items: sortByName(needsMetadata),
    });
    needsMetadata.forEach((channel) => seen.add(channel.id));
  }

  const uncategorized = channels.filter(
    (channel) => !seen.has(channel.id) && !(channel.channel_type && channel.channel_type.trim())
  );
  if (uncategorized.length) {
    groups.push({
      key: "uncategorized",
      title: "Uncategorized",
      items: sortByName(uncategorized),
    });
    uncategorized.forEach((channel) => seen.add(channel.id));
  }

  const categoryBuckets = new Map<string, Channel[]>();
  channels.forEach((channel) => {
    if (seen.has(channel.id)) return;
    const key = channel.channel_type || "";
    const bucket = categoryBuckets.get(key) || [];
    bucket.push(channel);
    categoryBuckets.set(key, bucket);
  });

  const sortedCategories = Array.from(categoryBuckets.entries()).sort((a, b) =>
    formatChannelLabel(a[0]).localeCompare(formatChannelLabel(b[0]))
  );
  sortedCategories.forEach(([key, items]) => {
    if (items.length === 0) return;
    groups.push({
      key: `category-${key || "other"}`,
      title: formatChannelLabel(key) || "Other",
      items: sortByName(items),
    });
  });

  if (groups.length === 0) {
    groups.push({ key: "all", title: "Channels", items: sortByName(channels) });
  }

  return (
    <Stack spacing={3}>
      {groups.map((group) => (
        <Box key={group.key}>
          <Typography variant="subtitle1" sx={{ mb: 1.5 }}>
            {group.title}
          </Typography>
          <Grid container spacing={2}>
            {group.items.map((channel) => renderChannelCard(channel))}
          </Grid>
        </Box>
      ))}
    </Stack>
  );
};

export default ChannelsLibrary;
