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

type UsageCount = {
  label: string;
  count: number;
};

type UsagePersona = {
  id: string;
  label?: string | null;
  count: number;
};

type UsageAccount = {
  id: string;
  name?: string | null;
  industry?: string | null;
  deal_status?: string | null;
  revenue_range?: string | null;
  employee_range?: string | null;
  geography?: string | null;
  funding_stage?: string | null;
};

type AssetUsage = {
  total_engagements: number;
  accounts?: UsageAccount[];
  industries?: UsageCount[];
  regions?: UsageCount[];
  sizes?: UsageCount[];
  personas?: UsagePersona[];
  concerns?: UsageCount[];
  stages?: Array<{ code?: string | null; label: string; count: number }>;
  segments?: UsageCount[];
};

type Asset = {
  id: string;
  name: string;
  slug?: string;
  category?: string | null;
  content_type?: string | null;
  time_to_consume?: string | null;
  depth?: string | null;
  metadata_complete?: boolean;
  description?: string | null;
  notes?: string | null;
  call_stage?: string | null;
  impacts?: Array<{
    persona_id: string;
    persona_label?: string | null;
    belief_transition_id?: string | null;
    impact_strength?: number | null;
    evidence_count?: number | null;
  }>;
  target_personas?: string[] | null;
  target_account_segments?: string[] | null;
  target_belief_stages?: string[] | null;
  target_concerns?: string[] | null;
  usage?: AssetUsage;
  typical_channels?: Array<{ id: string; label?: string | null }> | null;
  approval_status?: string | null;
  derived_metadata?: Record<string, any> | null;
  auto_classification_confidence?: number | null;
};

interface AssetsLibraryProps {
  assets: Asset[];
  onEdit: (asset: Asset) => void;
  onApprove: (asset: Asset) => void;
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

const formatMetadataLabel = (value?: string | null) => {
  if (!value) return "";
  return toTitleCase(value);
};

function buildMetadataChips(asset: Asset) {
  const chips: Array<{ key: string; label: string; color: "primary" | "secondary" | "default" }> =
    [];
  if (asset.category)
    chips.push({ key: "category", label: formatMetadataLabel(asset.category), color: "primary" });
  if (asset.content_type)
    chips.push({
      key: "content_type",
      label: formatMetadataLabel(asset.content_type),
      color: "secondary",
    });
  if (asset.time_to_consume)
    chips.push({
      key: "time_to_consume",
      label: formatMetadataLabel(asset.time_to_consume),
      color: "default",
    });
  if (asset.depth)
    chips.push({ key: "depth", label: formatMetadataLabel(asset.depth), color: "default" });
  if (asset.call_stage)
    chips.push({
      key: "call_stage",
      label: `Stage: ${formatMetadataLabel(asset.call_stage)}`,
      color: "default",
    });
  return chips;
}

function summarizeImpacts(asset: Asset) {
  const impacts = asset.impacts ?? [];
  if (!impacts.length) {
    return {
      personaLabels: [] as string[],
      evidenceCount: 0,
      avgStrength: null as number | null,
    };
  }
  const personaLabels = Array.from(
    new Set(
      impacts
        .map((impact) => impact.persona_label || impact.persona_id)
        .filter((label): label is string => Boolean(label))
    )
  );
  const evidenceCount = impacts.reduce(
    (sum, impact) => sum + (impact.evidence_count || 0),
    0
  );
  const avgStrength =
    impacts.reduce((sum, impact) => sum + (impact.impact_strength || 0), 0) / impacts.length;
  return { personaLabels, evidenceCount, avgStrength };
}

function summarizeAccounts(accounts?: UsageAccount[], limit = 3): string | null {
  if (!accounts || accounts.length === 0) return null;
  const names = accounts
    .map((account) => account.name || account.industry || account.id)
    .filter(Boolean) as string[];
  if (names.length === 0) return null;
  const visible = names.slice(0, limit);
  const remaining = names.length - visible.length;
  return remaining > 0 ? `${visible.join(", ")} +${remaining}` : visible.join(", ");
}

const AssetsLibrary = ({ assets, onEdit, onApprove }: AssetsLibraryProps) => {
  const [expandedAssetId, setExpandedAssetId] = useState<string | null>(null);
  if (!assets.length) {
    return (
      <Grid container spacing={2}>
        <Grid size={12}>
          <Typography color="text.secondary">No assets found.</Typography>
        </Grid>
      </Grid>
    );
  }

  const renderAssetCard = (asset: Asset) => {
    const chips = buildMetadataChips(asset);
    const { personaLabels, evidenceCount, avgStrength } = summarizeImpacts(asset);
    const metadataComplete = Boolean(asset.metadata_complete);
    const needsApproval = (asset.approval_status || "pending") !== "approved";
    const bestChannels = asset.typical_channels?.slice(0, 3) ?? [];
    const accountsSummary = summarizeAccounts(asset.usage?.accounts);
    const isExpanded = expandedAssetId === asset.id;

    return (
      <Grid key={asset.id} size={{ xs: 12, md: 6 }}>
        <Card
          variant="outlined"
          sx={{ height: "100%", display: "flex", flexDirection: "column", gap: 1 }}
        >
          <CardHeader
            title={
              <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap" useFlexGap>
                <Typography variant="h6">{asset.name}</Typography>
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
            subheader={asset.slug ? `id: ${asset.slug}` : undefined}
          />
          <CardContent sx={{ flexGrow: 1, display: "flex", flexDirection: "column", gap: 1.5 }}>
            {chips.length > 0 && (
              <Stack direction="row" spacing={0.75} flexWrap="wrap" useFlexGap>
                {chips.map((chip) => (
                  <Chip
                    key={`${asset.id}-${chip.key}-${chip.label}`}
                    label={chip.label}
                    size="small"
                    color={chip.color}
                  />
                ))}
              </Stack>
            )}

            {asset.description && <Typography variant="body2">{asset.description}</Typography>}

            {asset.notes && (
              <Typography variant="body2" color="text.secondary">
                Notes: {asset.notes}
              </Typography>
            )}

            {needsApproval && asset.derived_metadata && (
              <Box sx={{ borderRadius: 1, border: 1, borderColor: "warning.light", p: 1 }}>
                <Typography variant="caption" color="warning.main">
                  Auto-generated
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Suggested asset: {asset.derived_metadata.asset_name?.label || "—"}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Suggested channel: {asset.derived_metadata.channel?.label || "—"}
                </Typography>
                {typeof asset.auto_classification_confidence === "number" && (
                  <Typography variant="caption" color="text.secondary">
                    Confidence {(asset.auto_classification_confidence * 100).toFixed(0)}%
                  </Typography>
                )}
              </Box>
            )}

            {bestChannels.length > 0 && (
              <Box>
                <Typography variant="caption" color="text.secondary">
                  Best channels
                </Typography>
                <Stack direction="row" spacing={0.75} flexWrap="wrap" useFlexGap sx={{ mt: 0.5 }}>
                  {bestChannels.map((channel) => (
                    <Chip
                      key={`${asset.id}-channel-${channel.id}`}
                      size="small"
                      variant="outlined"
                      label={channel.label || channel.id}
                    />
                  ))}
                </Stack>
              </Box>
            )}

            {personaLabels.length > 0 && (
              <Stack direction="row" spacing={0.75} flexWrap="wrap" useFlexGap>
                {personaLabels.map((label) => (
                  <Chip
                    key={`${asset.id}-persona-${label}`}
                    size="small"
                    variant="outlined"
                    label={label}
                  />
                ))}
              </Stack>
            )}

            <Box sx={{ display: "flex", gap: 2, flexWrap: "wrap" }}>
              <Typography variant="caption" color="text.secondary">
                Evidence: {evidenceCount}
              </Typography>
              {avgStrength !== null && (
                <Typography variant="caption" color="text.secondary">
                  Avg Δ log p: {avgStrength.toFixed(3)}
                </Typography>
              )}
              {accountsSummary && (
                <Typography variant="caption" color="text.secondary">
                  Seen in {accountsSummary}
                </Typography>
              )}
            </Box>

            {asset.target_personas && asset.target_personas.length > 0 && (
              <Stack direction="row" spacing={0.75} flexWrap="wrap" useFlexGap>
                {asset.target_personas.map((persona) => (
                  <Chip
                    key={`${asset.id}-target-persona-${persona}`}
                    size="small"
                    color="info"
                    variant="outlined"
                    label={persona}
                  />
                ))}
              </Stack>
            )}

            {asset.target_account_segments && asset.target_account_segments.length > 0 && (
              <Stack direction="row" spacing={0.75} flexWrap="wrap" useFlexGap>
                {asset.target_account_segments.map((segment) => (
                  <Chip
                    key={`${asset.id}-segment-${segment}`}
                    size="small"
                    variant="outlined"
                    label={segment}
                  />
                ))}
              </Stack>
            )}

            {asset.target_belief_stages && asset.target_belief_stages.length > 0 && (
              <Stack direction="row" spacing={0.75} flexWrap="wrap" useFlexGap>
                {asset.target_belief_stages.map((stage) => (
                  <Chip
                    key={`${asset.id}-target-stage-${stage}`}
                    size="small"
                    variant="outlined"
                    label={stage}
                  />
                ))}
              </Stack>
            )}

            {asset.target_concerns && asset.target_concerns.length > 0 && (
              <Stack direction="row" spacing={0.75} flexWrap="wrap" useFlexGap>
                {asset.target_concerns.map((concern) => (
                  <Chip
                    key={`${asset.id}-target-concern-${concern}`}
                    size="small"
                    variant="outlined"
                    label={concern}
                  />
                ))}
              </Stack>
            )}

            {asset.typical_channels && asset.typical_channels.length > 0 && (
              <Box>
                <Typography variant="caption" color="text.secondary" sx={{ display: "block", mb: 0.5 }}>
                  Typical channels
                </Typography>
                <Stack direction="row" spacing={0.75} flexWrap="wrap" useFlexGap>
                  {asset.typical_channels.map((channel) => (
                    <Chip
                      key={`${asset.id}-typical-channel-${channel.id}`}
                      size="small"
                      variant="outlined"
                      color="secondary"
                      label={channel.label || channel.id}
                    />
                  ))}
                </Stack>
              </Box>
            )}

            {asset.usage && asset.usage.total_engagements > 0 && (
              <Box sx={{ borderRadius: 1, border: 1, borderColor: "divider", p: 1 }}>
                <Stack spacing={0.75}>
                  <Typography variant="subtitle2">Usage</Typography>
                  <Typography variant="caption" color="text.secondary">
                    Engagements tracked: {asset.usage.total_engagements}
                  </Typography>
                  {asset.usage.personas && asset.usage.personas.length > 0 && (
                    <Stack direction="row" spacing={0.75} flexWrap="wrap" useFlexGap>
                      {asset.usage.personas.slice(0, 4).map((persona) => (
                        <Chip
                          key={`${asset.id}-usage-persona-${persona.id}`}
                          size="small"
                          label={`${persona.label || persona.id} (${persona.count})`}
                          variant="outlined"
                        />
                      ))}
                    </Stack>
                  )}
                  {asset.usage.stages && asset.usage.stages.length > 0 && (
                    <Stack direction="row" spacing={0.75} flexWrap="wrap" useFlexGap>
                      {asset.usage.stages.slice(0, 4).map((stage) => (
                        <Chip
                          key={`${asset.id}-usage-stage-${stage.code || stage.label}`}
                          size="small"
                          label={`${stage.label} (${stage.count})`}
                          variant="outlined"
                          color="success"
                        />
                      ))}
                    </Stack>
                  )}
                  {asset.usage.industries && asset.usage.industries.length > 0 && (
                    <Typography variant="caption" color="text.secondary">
                      Account types:{" "}
                      {asset.usage.industries
                        .slice(0, 3)
                        .map((entry) => `${entry.label} (${entry.count})`)
                        .join(", ")}
                      {asset.usage.industries.length > 3
                        ? ` +${asset.usage.industries.length - 3}`
                        : ""}
                    </Typography>
                  )}
                  {asset.usage.stages && asset.usage.stages.length > 0 && (
                    <Typography variant="caption" color="text.secondary">
                      Stages:{" "}
                      {asset.usage.stages
                        .slice(0, 3)
                        .map((entry) => `${entry.label} (${entry.count})`)
                        .join(", ")}
                      {asset.usage.stages.length > 3
                        ? ` +${asset.usage.stages.length - 3}`
                        : ""}
                    </Typography>
                  )}
                  {summarizeAccounts(asset.usage.accounts) && (
                    <Typography variant="caption" color="text.secondary">
                      Accounts: {summarizeAccounts(asset.usage.accounts)}
                    </Typography>
                  )}
                  {asset.usage.concerns && asset.usage.concerns.length > 0 && (
                    <Stack direction="row" spacing={0.75} flexWrap="wrap" useFlexGap>
                      {asset.usage.concerns.slice(0, 6).map((concern) => (
                        <Chip
                          key={`${asset.id}-usage-concern-${concern.label}`}
                          size="small"
                          variant="outlined"
                          color="warning"
                          label={`${concern.label} (${concern.count})`}
                        />
                      ))}
                    </Stack>
                  )}
                  {asset.usage.segments && asset.usage.segments.length > 0 && (
                    <Stack direction="row" spacing={0.75} flexWrap="wrap" useFlexGap>
                      {asset.usage.segments.slice(0, 6).map((segment) => (
                        <Chip
                          key={`${asset.id}-usage-segment-${segment.label}`}
                          size="small"
                          variant="outlined"
                          color="default"
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
              <Button size="small" variant="outlined" onClick={() => onEdit(asset)}>
                Edit metadata
              </Button>
              {needsApproval && (
                <Button size="small" color="success" onClick={() => onApprove(asset)}>
                  Approve
                </Button>
              )}
            </Stack>
            <Button size="small" onClick={() => setExpandedAssetId(isExpanded ? null : asset.id)}>
              {isExpanded ? "Hide reasoning" : "View reasoning"}
            </Button>
          </CardActions>
          {isExpanded && (
            <Box sx={{ px: 3, pb: 2 }}>
              <Stack spacing={1}>
                {personaLabels.length > 0 && (
                  <Typography variant="body2" color="text.secondary">
                    Personas: {personaLabels.join(", ")}
                  </Typography>
                )}
                {asset.target_concerns && asset.target_concerns.length > 0 && (
                  <Typography variant="body2" color="text.secondary">
                    Key concerns: {asset.target_concerns.slice(0, 3).join(", ")}
                  </Typography>
                )}
                {asset.target_belief_stages && asset.target_belief_stages.length > 0 && (
                  <Typography variant="body2" color="text.secondary">
                    Belief focus: {asset.target_belief_stages.join(", ")}
                  </Typography>
                )}
              </Stack>
            </Box>
          )}
        </Card>
      </Grid>
    );
  };

  const sortByName = (list: Asset[]) =>
    [...list].sort((a, b) => a.name.localeCompare(b.name, undefined, { sensitivity: "base" }));

  const groups: Array<{ key: string; title: string; items: Asset[] }> = [];
  const seen = new Set<string>();

  const needsMetadata = assets.filter((asset) => !asset.metadata_complete);
  if (needsMetadata.length) {
    groups.push({
      key: "needs-metadata",
      title: "Needs metadata",
      items: sortByName(needsMetadata),
    });
    needsMetadata.forEach((asset) => seen.add(asset.id));
  }

  const uncategorized = assets.filter(
    (asset) => !seen.has(asset.id) && !(asset.category && asset.category.trim())
  );
  if (uncategorized.length) {
    groups.push({
      key: "uncategorized",
      title: "Uncategorized",
      items: sortByName(uncategorized),
    });
    uncategorized.forEach((asset) => seen.add(asset.id));
  }

  const categoryBuckets = new Map<string, Asset[]>();
  assets.forEach((asset) => {
    if (seen.has(asset.id)) return;
    const key = asset.category || "";
    const bucket = categoryBuckets.get(key) || [];
    bucket.push(asset);
    categoryBuckets.set(key, bucket);
  });

  const sortedCategories = Array.from(categoryBuckets.entries()).sort((a, b) =>
    formatMetadataLabel(a[0]).localeCompare(formatMetadataLabel(b[0]))
  );
  sortedCategories.forEach(([key, items]) => {
    if (items.length === 0) return;
    groups.push({
      key: `category-${key || "other"}`,
      title: formatMetadataLabel(key) || "Other",
      items: sortByName(items),
    });
  });

  if (groups.length === 0) {
    groups.push({ key: "all", title: "Assets", items: sortByName(assets) });
  }

  return (
    <Stack spacing={3}>
      {groups.map((group) => (
        <Box key={group.key}>
          <Typography variant="subtitle1" sx={{ mb: 1.5 }}>
            {group.title}
          </Typography>
          <Grid container spacing={2}>
            {group.items.map((asset) => renderAssetCard(asset))}
          </Grid>
        </Box>
      ))}
    </Stack>
  );
};

export default AssetsLibrary;
