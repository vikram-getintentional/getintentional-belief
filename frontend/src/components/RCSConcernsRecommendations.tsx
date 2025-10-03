import React from "react";
import { Box, Card, CardContent, Typography, Grid, Chip } from "@mui/material";

type Concern = {
  concern_label: string;
  persona_label: string;
  asset_name: string;
  channel: string;
  fitness: number; // 0-1
  reach: number;   // 0-1
  potential_lift: number; // as bips (basis points)
};

export default function RCSConcernsRecommendations({
  concerns,
}: {
  concerns: any[]; // Accept raw backend shape
}) {
    console.log("RCSConcernsRecommendations input:", concerns);
  return (
    <Grid container spacing={2}>
      {concerns.map((c, idx) => {
        const asset = c.asset_reco || {};
        const channelObj = asset.channel || {};
        const fitness = asset.fitness ?? 0;
        const reach = channelObj.reach_score ?? 1;
        const channel = channelObj.name ?? "";
        return (
          <Grid size={{xs: 12}} key={idx}>
            <Card variant="outlined" sx={{ height: "100%", display: "flex", flexDirection: "column", justifyContent: "space-between" }}>
              <CardContent>
                <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                    For: {c.persona_label}
                </Typography>
                <Typography variant="subtitle2" gutterBottom>
                  Concerned with: {c.concern_label}
                </Typography>
                
                <Typography variant="body2" color="text.secondary">
                  Asset: {asset.asset_name}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Channel: {channel}
                </Typography>
              </CardContent>
              <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center", px: 2, pb: 2 }}>
                <Typography variant="body2" color="success.main">
                  Success Likelihood: {Math.round((fitness) * 100)}%
                </Typography>
                <Typography variant="body2" color="success.main">
                  Reach: {Math.round((reach) * 100)}%
                </Typography>
                <Typography variant="body2" color="info.main">
                  Potential Lift: {Math.round((c.potential_lift ?? 0) * 10000)} bips
                </Typography>
              </Box>
            </Card>
          </Grid>
        );
      })}
    </Grid>
  );
}