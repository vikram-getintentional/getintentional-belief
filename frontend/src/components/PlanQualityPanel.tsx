import React from "react";
import {
  Alert,
  Box,
  Button,
  LinearProgress,
  Rating,
  Stack,
  Typography,
  Paper,
  Tooltip,
  SxProps,
  Theme,
} from "@mui/material";
import CheckCircleOutlineIcon from "@mui/icons-material/CheckCircleOutline";
import WarningAmberIcon from "@mui/icons-material/WarningAmber";
import ErrorOutlineIcon from "@mui/icons-material/ErrorOutline";
import { Link as RouterLink } from "react-router-dom";
import type { PlanQuality, QualityReadinessDimension } from "../types/apiContracts";

const statusIcon = (status: QualityReadinessDimension["status"]) => {
  switch (status) {
    case "ready":
      return <CheckCircleOutlineIcon color="success" fontSize="small" />;
    case "partial":
      return <WarningAmberIcon color="warning" fontSize="small" />;
    default:
      return <ErrorOutlineIcon color="error" fontSize="small" />;
  }
};

type PlanQualityPanelProps = {
  quality?: PlanQuality | null;
  sx?: SxProps<Theme>;
};

const READINESS_BANNER_THRESHOLD = 40;

const missingDimensions = (quality: PlanQuality) =>
  quality.readiness.dimensions
    .filter((dim) => dim.score < 50)
    .map((dim) => dim.label);

const readinessStatusLabel = (score: number) => {
  if (score >= 90) return "Strong";
  if (score >= 70) return "Emerging";
  if (score >= 40) return "Weak";
  return "Draft";
};

const PlanQualityPanel: React.FC<PlanQualityPanelProps> = ({ quality, sx }) => {
  if (!quality) {
    return null;
  }

  const readiness = quality.readiness;
  const confidence = quality.predictiveConfidence;
  const bannerVisible = readiness.score < READINESS_BANNER_THRESHOLD;
  const missing = missingDimensions(quality);

  return (
    <Paper variant="outlined" sx={{ p: 2, ...sx }}>
      <Stack spacing={2}>
        <Stack
          direction={{ xs: "column", sm: "row" }}
          alignItems="center"
          justifyContent="space-between"
          spacing={1}
        >
          <Box>
            <Typography variant="subtitle1">Plan Readiness</Typography>
            <Typography variant="body2" color="text.secondary">
              {readinessStatusLabel(readiness.score)} • Score {readiness.score}/100
            </Typography>
          </Box>
          <Box sx={{ width: { xs: "100%", sm: 220 } }}>
            <LinearProgress
              variant="determinate"
              value={Math.min(Math.max(readiness.score, 0), 100)}
              sx={{ height: 6, borderRadius: 3 }}
            />
            <Typography variant="caption" color="text.secondary" sx={{ display: "block", mt: 0.5 }}>
              {readiness.score}/100 readiness
            </Typography>
          </Box>
        </Stack>

        <Stack spacing={1}>
          {readiness.dimensions.map((dimension) => {
            const action = dimension.recommended_actions?.[0];
            return (
              <Stack
                key={dimension.id}
                direction="row"
                alignItems="flex-start"
                spacing={1}
                sx={{ flexWrap: "wrap" }}
              >
                <Box sx={{ mt: 0.25 }}>{statusIcon(dimension.status)}</Box>
                <Box sx={{ flex: 1, minWidth: 0 }}>
                  <Stack direction="row" alignItems="baseline" spacing={1}>
                    <Typography variant="subtitle2">{dimension.label}</Typography>
                    <Typography variant="caption" color="text.secondary">
                      {dimension.score}/100
                    </Typography>
                  </Stack>
                  <Typography variant="body2" color="text.secondary">
                    {dimension.reason}
                  </Typography>
                </Box>
                {action?.route && (
                  <Button
                    size="small"
                    component={RouterLink}
                    to={action.route}
                    variant="outlined"
                  >
                    {action.label}
                  </Button>
                )}
              </Stack>
            );
          })}
        </Stack>

        <Stack
          direction={{ xs: "column", sm: "row" }}
          alignItems="center"
          justifyContent="space-between"
          spacing={2}
        >
          <Stack direction="row" alignItems="center" spacing={1}>
            <Typography variant="subtitle1">Predictive Confidence</Typography>
            <Tooltip title={confidence.explanation}>
              <Rating value={confidence.stars} readOnly max={5} size="small" />
            </Tooltip>
          </Stack>
          <Typography variant="body2" color="text.secondary">
            {confidence.score.toFixed(2)} confidence • {confidence.explanation}
          </Typography>
        </Stack>

        {bannerVisible && missing.length > 0 ? (
          <Alert severity="info" variant="outlined">
            This plan is draft-quality because we’re missing inputs. GI will produce a much
            sharper plan once you add: {missing.join(", ")}.
          </Alert>
        ) : null}
      </Stack>
    </Paper>
  );
};

export default PlanQualityPanel;
