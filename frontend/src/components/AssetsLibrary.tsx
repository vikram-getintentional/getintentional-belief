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
};

interface AssetsLibraryProps {
  assets: Asset[];
  onEdit: (asset: Asset) => void;
}

function buildMetadataChips(asset: Asset) {
  const chips: Array<{ key: string; label: string; color: "primary" | "secondary" | "default" }> = [];
  if (asset.category) chips.push({ key: "category", label: asset.category, color: "primary" });
  if (asset.content_type)
    chips.push({ key: "content_type", label: asset.content_type, color: "secondary" });
  if (asset.time_to_consume)
    chips.push({ key: "time_to_consume", label: asset.time_to_consume, color: "default" });
  if (asset.depth) chips.push({ key: "depth", label: asset.depth, color: "default" });
  if (asset.call_stage) chips.push({ key: "call_stage", label: `Stage: ${asset.call_stage}`, color: "default" });
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

const AssetsLibrary = ({ assets, onEdit }: AssetsLibraryProps) => {
  if (!assets.length) {
    return (
      <Grid container spacing={2}>
        <Grid size={12}>
          <Typography color="text.secondary">No assets found.</Typography>
        </Grid>
      </Grid>
    );
  }

  return (
    <Grid container spacing={2}>
      {assets.map((asset) => {
        const chips = buildMetadataChips(asset);
        const { personaLabels, evidenceCount, avgStrength } = summarizeImpacts(asset);
        const metadataComplete = Boolean(asset.metadata_complete);
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

                {asset.description && (
                  <Typography variant="body2">{asset.description}</Typography>
                )}

                {asset.notes && (
                  <Typography variant="body2" color="text.secondary">
                    Notes: {asset.notes}
                  </Typography>
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
                </Box>
              </CardContent>
              <CardActions sx={{ justifyContent: "flex-end", px: 2, pb: 2 }}>
                <Button size="small" variant="outlined" onClick={() => onEdit(asset)}>
                  Edit metadata
                </Button>
              </CardActions>
            </Card>
          </Grid>
        );
      })}
    </Grid>
  );
};

export default AssetsLibrary;
