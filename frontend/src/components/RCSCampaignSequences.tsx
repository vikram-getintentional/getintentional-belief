import React, { useMemo, useState } from "react";
import {
  Box,
  Card,
  CardContent,
  Chip,
  Stack,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  Typography,
  Button,
  Tooltip,
} from "@mui/material";

function pickSequences(rcs: any): any[] {
  // canonical today
  const seq = rcs?.frozen_strategy?.report?.concern_sequences;
  if (Array.isArray(seq)) return seq;
  // fallbacks
  if (Array.isArray(rcs?.report?.concern_sequences)) return rcs.report.concern_sequences;
  if (Array.isArray(rcs?.frozen_strategy?.report?.sequences)) return rcs.frozen_strategy.report.sequences;
  return [];
}

export default function RCSCampaignSequences({ rcs }: { rcs: any }) {
  const all = useMemo(() => pickSequences(rcs), [rcs]);
  const [limit, setLimit] = useState(10);

  const rows = all.slice(0, limit).map((s: any, i: number) => {
    // each entry typically has: persona, stage, sequence: [{concern_label, asset_label, expected_lift, ...}]
    const persona = s?.persona_label || s?.persona || s?.pid || "—";
    const stage = s?.stage || s?.stage_label || "—";
    const step = (s?.sequence || [])[0] || {};
    const concern = step?.concern_label || step?.concern || "—";
    const asset = step?.asset_label || step?.asset || step?.recommended_asset || "—";
    const lift = step?.expected_lift ?? step?.lift ?? null;

    return { key: `${i}-${persona}`, persona, stage, concern, asset, lift };
  });

  return (
    <Card variant="outlined" sx={{ borderRadius: 2 }}>
      <CardContent>
        <Typography variant="subtitle1" sx={{ mb: 2, fontWeight: 600 }}>
          Campaign Sequences
        </Typography>

        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>Persona</TableCell>
              <TableCell>Stage</TableCell>
              <TableCell>Concern</TableCell>
              <TableCell>Recommended Asset</TableCell>
              <TableCell align="right">Expected Lift</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {rows.map((r) => (
              <TableRow key={r.key}>
                <TableCell sx={{ maxWidth: 260 }}>
                  <Tooltip title={r.persona}>
                    <Chip size="small" label={r.persona} />
                  </Tooltip>
                </TableCell>
                <TableCell>{r.stage}</TableCell>
                <TableCell sx={{ maxWidth: 360 }}>{r.concern}</TableCell>
                <TableCell sx={{ maxWidth: 360 }}>{r.asset}</TableCell>
                <TableCell align="right">
                  {typeof r.lift === "number" ? `${Math.round(r.lift * 100)}%` : "—"}
                </TableCell>
              </TableRow>
            ))}
            {rows.length === 0 && (
              <TableRow>
                <TableCell colSpan={5} sx={{ color: "text.disabled" }}>
                  No sequences returned.
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>

        {limit < all.length && (
          <Box sx={{ display: "flex", justifyContent: "center", mt: 2 }}>
            <Button variant="outlined" onClick={() => setLimit((n) => Math.min(n + 10, all.length))}>
              Load more
            </Button>
          </Box>
        )}
      </CardContent>
    </Card>
  );
}
