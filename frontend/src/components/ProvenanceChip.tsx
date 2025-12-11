import { Chip, Tooltip } from "@mui/material";
import type { ChipProps } from "@mui/material";
import type { SxProps, Theme } from "@mui/material/styles";

export type InsightSource = "data" | "graph" | "default" | "mixed";

export type ValueProvenance = {
  source?: InsightSource;
  confidence?: number | null;
  [key: string]: any;
};

export type TextInsight = ValueProvenance & {
  text: string;
};

const SOURCE_LABELS: Record<InsightSource, string> = {
  data: "Data",
  graph: "Graph",
  default: "Default",
  mixed: "Mixed",
};

const SOURCE_COLORS: Record<InsightSource, ChipProps["color"]> = {
  data: "success",
  graph: "info",
  default: "warning",
  mixed: "secondary",
};

type ProvenanceChipProps = {
  provenance?: ValueProvenance;
  size?: ChipProps["size"];
  sx?: SxProps<Theme>;
};

const ProvenanceChip = ({ provenance, size = "small", sx }: ProvenanceChipProps) => {
  if (!provenance?.source) {
    return null;
  }
  const color = SOURCE_COLORS[provenance.source] ?? "default";
  const label = SOURCE_LABELS[provenance.source] ?? provenance.source;
  const chip = (
    <Chip
      size={size}
      label={label}
      color={color}
      variant="outlined"
      sx={sx}
      data-testid="provenance-chip"
    />
  );
  if (typeof provenance.confidence === "number") {
    const pct = Math.round(provenance.confidence * 100);
    return <Tooltip title={`Confidence ${pct}%`}>{chip}</Tooltip>;
  }
  return chip;
};

export default ProvenanceChip;
