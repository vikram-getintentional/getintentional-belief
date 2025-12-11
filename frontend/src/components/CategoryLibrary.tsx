import { Box, Card, CardContent, CardHeader, Chip, Stack, Typography } from "@mui/material";
import type {
  ArsenalAssetCategoryMeta,
  ArsenalChannelCategoryMeta,
} from "../types/apiContracts";

const formatLabel = (value?: string | null) => {
  if (!value) return "";
  return value
    .replace(/[_\s]+/g, " ")
    .trim()
    .split(" ")
    .filter(Boolean)
    .map((segment) => segment.charAt(0).toUpperCase() + segment.slice(1))
    .join(" ");
};

type CategoryLibraryProps = {
  categories: (ArsenalAssetCategoryMeta | ArsenalChannelCategoryMeta)[];
  type: "asset" | "channel";
  assetLabelMap?: Record<string, string>;
  channelCategoryAssets?: Record<string, Array<{ assetId: string; score: number }>>;
};

const CategoryLibrary = ({
  categories,
  type,
  assetLabelMap,
  channelCategoryAssets,
}: CategoryLibraryProps) => {
  if (!categories.length) {
    return (
      <Typography color="text.secondary">
        No {type === "asset" ? "asset categories" : "channel categories"} found.
      </Typography>
    );
  }

  return (
    <Stack spacing={2}>
      {categories.map((category) => {
        const label = category.label || formatLabel(category.name);
        const stageList =
          type === "asset"
            ? (category as ArsenalAssetCategoryMeta).activity_stages
            : (category as ArsenalChannelCategoryMeta).channel_activity_stages;
        const entryIds = type === "asset"
          ? (category as ArsenalAssetCategoryMeta).asset_ids
          : (category as ArsenalChannelCategoryMeta).channel_ids;
        const delivery =
          type === "channel" ? (category as ArsenalChannelCategoryMeta).delivery_mode : null;
        const reach =
          type === "channel" ? (category as ArsenalChannelCategoryMeta).reach_score_estimate : null;
        const linkedAssets =
          type === "channel"
            ? (channelCategoryAssets?.[category.key] || [])
                .slice()
                .sort((a, b) => b.score - a.score)
                .slice(0, 3)
                .map((entry) => assetLabelMap?.[entry.assetId] || entry.assetId)
            : [];

        return (
          <Card key={category.key} variant="outlined">
            <CardHeader
              title={label}
              subheader={`Key: ${category.key}`}
              sx={{ pb: 0 }}
            />
            <CardContent>
              {stageList && stageList.length > 0 && (
                <Box mb={1}>
                  <Typography variant="subtitle2">Activity stages</Typography>
                  <Stack direction="row" spacing={0.5} useFlexGap flexWrap="wrap" sx={{ mt: 0.5 }}>
                    {stageList.map((stage) => (
                      <Chip key={stage} size="small" label={stage} />
                    ))}
                  </Stack>
                </Box>
              )}
              {type === "channel" && delivery && (
                <Typography variant="body2">
                  Delivery mode: {formatLabel(delivery)}
                </Typography>
              )}
              {type === "channel" && reach !== null && reach !== undefined && (
                <Typography variant="body2">
                  Reach score: {reach.toFixed(2)}
                </Typography>
              )}
              {type === "asset" && entryIds && entryIds.length > 0 && (
                <Box mt={1}>
                  <Typography variant="subtitle2">Assets using this category</Typography>
                  <Stack direction="row" spacing={0.5} useFlexGap flexWrap="wrap" sx={{ mt: 0.5 }}>
                    {entryIds.map((entry) => (
                      <Chip
                        key={entry}
                        size="small"
                        variant="outlined"
                        label={assetLabelMap?.[entry] || entry}
                      />
                    ))}
                  </Stack>
                </Box>
              )}
              {type === "channel" && linkedAssets.length > 0 && (
                <Box mt={1}>
                  <Typography variant="subtitle2">Top arsenals using this channel</Typography>
                  <Stack direction="row" spacing={0.5} useFlexGap flexWrap="wrap" sx={{ mt: 0.5 }}>
                    {linkedAssets.map((assetLabel) => (
                      <Chip key={assetLabel} size="small" variant="outlined" label={assetLabel} />
                    ))}
                  </Stack>
                </Box>
              )}
            </CardContent>
          </Card>
        );
      })}
    </Stack>
  );
};

export default CategoryLibrary;
