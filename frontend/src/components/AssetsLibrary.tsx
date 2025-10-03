import React from "react";
import { Card, CardContent, CardHeader, Typography, Grid } from "@mui/material";

type Persona = {
  title: string;
};

type Asset = {
  id: string | number;
  name: string;
  format: string;
  stages?: string[];
  personas?: Persona[];
  industry_tags?: string[];
  engagement_rate?: number;
  evergreen?: boolean;
  evidence_strength?: string | null;
  notes?: string;
  purpose?: string;
};

interface AssetsLibraryProps {
  assets: Asset[];
}

const AssetsLibrary: React.FC<AssetsLibraryProps> = ({ assets }) => (
  <Grid container spacing={2}>
    {assets.length === 0 && (
      <Grid size={12}>
        <Typography color="textSecondary">No assets found.</Typography>
      </Grid>
    )}
    {assets.map(asset => (
      <Grid size={12} key={asset.id}>
        <Card variant="outlined">
          <CardHeader
            title={
              <Typography variant="h6" gutterBottom>
                {asset.name}
              </Typography>
            }
            subheader={
              <Typography variant="body1" gutterBottom>
                {asset.format}
              </Typography>
            }
            sx={{ color: "indigo" }}
          />
          <CardContent>
            {asset.stages && (
              <Typography variant="body2">
                <strong>Stages:</strong> {asset.stages.join(", ")}
              </Typography>
            )}
            {asset.personas && (
              <Typography variant="body2">
                <strong>Personas:</strong> {asset.personas.map(p => p.title).join(", ")}
              </Typography>
            )}
            {asset.industry_tags && (
              <Typography variant="body2">
                <strong>Industries:</strong> {asset.industry_tags.join(", ")}
              </Typography>
            )}
            {asset.purpose && (
              <Typography variant="body2">
                <strong>Purpose:</strong> {asset.purpose}
              </Typography>
            )}
            {asset.engagement_rate !== undefined && (
              <Typography variant="body2">
                <strong>Engagement Rate:</strong> {(asset.engagement_rate * 100).toFixed(2)}%
              </Typography>
            )}
            {asset.evergreen !== undefined && (
              <Typography variant="body2">
                <strong>Evergreen:</strong> {asset.evergreen ? "Yes" : "No"}
              </Typography>
            )}
            {asset.evidence_strength && (
              <Typography variant="body2">
                <strong>Evidence Strength:</strong> {asset.evidence_strength}
              </Typography>
            )}
            {asset.notes && (
              <Typography variant="body2">
                <strong>Notes:</strong> {asset.notes}
              </Typography>
            )}
          </CardContent>
        </Card>
      </Grid>
    ))}
  </Grid>
);

export default AssetsLibrary;