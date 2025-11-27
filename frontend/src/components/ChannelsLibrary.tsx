import {
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
};

interface ChannelsLibraryProps {
  channels: Channel[];
  onEdit: (channel: Channel) => void;
}

const ChannelsLibrary = ({ channels, onEdit }: ChannelsLibraryProps) => {
  if (!channels.length) {
    return (
      <Grid container spacing={2}>
        <Grid size={12}>
          <Typography color="text.secondary">No channels found.</Typography>
        </Grid>
      </Grid>
    );
  }

  return (
    <Grid container spacing={2}>
      {channels.map((channel) => {
        const metadataComplete = Boolean(channel.metadata_complete);
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

        return (
          <Grid key={channel.id} size={{ xs: 12, md: 6 }}>
            <Card variant="outlined" sx={{ height: "100%", display: "flex", flexDirection: "column", gap: 1 }}>
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
                  </Stack>
                }
                subheader={channel.slug ? `id: ${channel.slug}` : undefined}
              />
              <CardContent sx={{ flexGrow: 1, display: "flex", flexDirection: "column", gap: 1.5 }}>
                <Stack direction="row" spacing={0.75} flexWrap="wrap" useFlexGap>
                  {channel.channel_type && (
                    <Chip label={channel.channel_type} size="small" color="primary" />
                  )}
                  {channel.delivery_mode && (
                    <Chip label={channel.delivery_mode} size="small" color="secondary" />
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
              </CardContent>
              <CardActions sx={{ justifyContent: "flex-end", px: 2, pb: 2 }}>
                <Button size="small" variant="outlined" onClick={() => onEdit(channel)}>
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

export default ChannelsLibrary;
