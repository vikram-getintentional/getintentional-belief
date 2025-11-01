import React from "react";
import { Box, Typography, Divider, Table, TableHead, TableBody, TableRow, TableCell } from "@mui/material";

export default function RCSConcernsRecommendations({ tactical }: { tactical: any }) {
  const nextCampaigns = tactical?.next_campaigns || [];
  const nextSequences = tactical?.next_sequences || [];

  return (
    <Box sx={{ p: 2 }}>
      <Typography variant="h5" gutterBottom>Recommended Plays</Typography>

      <Typography variant="h6" sx={{ mt: 2 }}>Next Campaigns</Typography>
      {nextCampaigns.length ? (
        <Table size="small" sx={{ mb: 3 }}>
          <TableHead>
            <TableRow>
              <TableCell>Campaign</TableCell>
              <TableCell>Persona</TableCell>
              <TableCell>Objective</TableCell>
              <TableCell>Expected Lift</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {nextCampaigns.map((c: any, i: number) => (
              <TableRow key={i}>
                <TableCell>{c.campaign || c.description}</TableCell>
                <TableCell>{c.persona || c.target_persona}</TableCell>
                <TableCell>{c.objective || c.goal}</TableCell>
                <TableCell>
                  {typeof c.expected_lift === "number" ? `${Math.round(c.expected_lift * 100)}%` : (c.lift || "—")}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      ) : (
        <Typography color="text.secondary">No next campaigns available.</Typography>
      )}

      <Divider sx={{ my: 3 }} />

      <Typography variant="h6">Next Sequences</Typography>
      {nextSequences.length ? (
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>Stage</TableCell>
              <TableCell>Concern</TableCell>
              <TableCell>Recommended Asset</TableCell>
              <TableCell>Lift</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {nextSequences.map((s: any, i: number) => (
              <TableRow key={i}>
                <TableCell>{s.stage || s.phase}</TableCell>
                <TableCell>{s.concern}</TableCell>
                <TableCell>{s.asset || s.recommended_asset}</TableCell>
                <TableCell>
                  {typeof s.lift === "number" ? `${Math.round(s.lift * 100)}%` : (s.expected_lift || "—")}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      ) : (
        <Typography color="text.secondary">No next sequences available.</Typography>
      )}
    </Box>
  );
}
