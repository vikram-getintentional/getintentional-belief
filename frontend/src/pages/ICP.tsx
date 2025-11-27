// pages/ZmotIcp.tsx
import { useEffect, useState } from "react";
import {
  Box, Grid, Card, CardContent, CardHeader, Typography, Chip, Stack,
  Select, MenuItem, FormControl, InputLabel, Divider, List, ListItem, ListItemText,
  LinearProgress, Toolbar
} from "@mui/material";
import type { IcpUpliftsPayload, IcpCombo } from "../types/index";



export default function ZmotIcp() {
  const token = localStorage.getItem("token");
  const [products, setProducts] = useState<{ id: string; name: string }[]>([]);
  const [selectedProductId, setSelectedProductId] = useState<string>("");
  const [icp, setIcp] = useState<IcpUpliftsPayload | null>(null);
  const [statusMsg, setStatusMsg] = useState("");
  const [loading, setLoading] = useState(false);

  // Company + products
  useEffect(() => {
    (async () => {
      try {
        const me = await fetch("http://localhost:8000/me", {
          headers: { Authorization: `Bearer ${token}` },
        }).then(r => r.json());
        const prod = await fetch(`http://localhost:8000/get-products/${me.company_id}`, {
          headers: { Authorization: `Bearer ${token}` },
        }).then(r => r.json());
        if (!prod.products?.length) {
          setStatusMsg("No products found. Please run Value Prop first.");
          return;
        }
        setProducts(prod.products);
        if (prod.products.length === 1) setSelectedProductId(prod.products[0].id);
      } catch {
        setStatusMsg("Error fetching company or products.");
      }
    })();
  }, [token]);

  // Fetch ICP uplifts
  useEffect(() => {
    if (!selectedProductId) return;
    setLoading(true);
    (async () => {
      try {
        const data: IcpUpliftsPayload = await fetch(
          `http://localhost:8000/get-zmot-icp/${selectedProductId}`,
          { headers: { Authorization: `Bearer ${token}` } }
        ).then(r => r.json());

        // Normalize: sort combos by win_rate desc
        console.log("Fetched ICP data:", data);
        data.combos = (data.combos || []).slice().sort((a, b) => b.win_rate - a.win_rate);
        setIcp(data);
        setStatusMsg("");
      } catch {
        setStatusMsg("Error fetching ICPs.");
        setIcp(null);
      } finally {
        setLoading(false);
      }
    })();
  }, [selectedProductId, token]);

  return (
    <Box sx={{ display: "flex" }}>

      <Box component="main" sx={{ flexGrow: 1, p: 3 }}>
        <Toolbar />
        <Typography variant="h5" fontWeight={700} gutterBottom>ICP Archetypes</Typography>

        {products.length > 1 && (
          <Box sx={{ mb: 2, maxWidth: 360 }}>
            <FormControl fullWidth>
              <InputLabel id="prod-label">Select Product</InputLabel>
              <Select
                labelId="prod-label"
                label="Select Product"
                value={selectedProductId}
                onChange={(e) => setSelectedProductId(e.target.value)}
              >
                {products.map(p => <MenuItem key={p.id} value={p.id}>{p.name}</MenuItem>)}
              </Select>
            </FormControl>
          </Box>
        )}

        {statusMsg && <Typography color="error" sx={{ mb: 2 }}>{statusMsg}</Typography>}
        {loading && <LinearProgress sx={{ my: 2 }} />}

        {icp && (
          <Box sx={{ mb: 3 }}>
            <Typography variant="body2">
              Baseline conversion likelihood: <b>{(icp.baseline?.win_rate ?? 0).toFixed(3)}</b>
            </Typography>
            <Typography variant="caption" color="text.secondary">
              Graph fingerprint: {icp.graph_fingerprint?.slice(0, 10)}…
            </Typography>
          </Box>
        )}

        <Grid container spacing={2}>
          {(icp?.combos || []).map((combo, idx) => (
            <Grid size={{ xs: 12, sm: 6, md: 4, lg: 3 }} key={idx}>
              <IcpComboCard combo={combo} baseline={icp?.baseline?.win_rate ?? 0} />
            </Grid>
          ))}
        </Grid>
      </Box>
    </Box>
  );
}

function IcpComboCard({ combo, baseline }: { combo: IcpCombo; baseline: number }) {
  return (
    <Card variant="outlined" sx={{ height: "100%", display: "flex", flexDirection: "column" }}>
      <CardHeader
        title={
          <Stack direction="row" spacing={1} flexWrap="wrap">
            {combo.chips.map((c, i) => (
              <Chip key={i} label={`${c.family}: ${c.label}`} size="small" />
            ))}
          </Stack>
        }
      />
      <CardContent sx={{ pt: 0 }}>
        <Stack spacing={1} sx={{ mb: 1 }}>
          <Typography variant="body2">
            Win rate: <b>{(combo.win_rate).toFixed(3)}</b> &nbsp; (baseline {(baseline).toFixed(3)})
          </Typography>
          <Typography variant="body2">Lift: <b>{(combo.lift_abs).toFixed(3)}</b> ({(combo.lift_rel).toFixed(2)}×)</Typography>
        </Stack>

        <Divider sx={{ my: 1 }} />

        {/* ZMOTs */}
        {combo.zmots?.length ? (
          <>
            <Typography variant="subtitle2" sx={{ mb: 1 }}>Top ZMOTs</Typography>
            <List dense disablePadding>
              {combo.zmots.slice(0, 5).map(z => (
                <ListItem key={z.zmot_event_id} disablePadding sx={{ mb: 1 }}>
                  <ListItemText
                    primary={`${z.zmot_label} · score ${(z.score).toFixed(2)}`}
                    secondary={
                      <>
                        <Typography variant="caption" display="block" sx={{ mt: 0.5 }}>
                          Observable moments:
                          {" "}
                          {z.observable_moments.slice(0, 3).map(m => m.label).join(", ") || "—"}
                        </Typography>
                        <Typography variant="caption" display="block">
                          Keywords:
                          {" "}
                          {z.trigger_keywords.slice(0, 6).map(k => k.label).join(", ") || "—"}
                        </Typography>
                      </>
                    }
                  />
                </ListItem>
              ))}
            </List>
          </>
        ) : (
          <Typography variant="caption" color="text.secondary">No ZMOTs found for this combo.</Typography>
        )}
      </CardContent>
    </Card>
  );
}
