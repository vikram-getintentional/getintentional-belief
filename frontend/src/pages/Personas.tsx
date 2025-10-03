// pages/Personas.tsx
import { useEffect, useState } from "react";
import {
  Box, Grid, Card, CardHeader, CardContent, Typography, Chip, Stack,
  Select, MenuItem, FormControl, InputLabel, LinearProgress, Toolbar, Drawer, Button
} from "@mui/material";
import type { PersonaCardRCSPayload } from "../types/index";

<<<<<<< Updated upstream
const Personas = () => {
  const [_, setCompanyId] = useState<string | null>(null);
  const [products, setProducts] = useState<{ id: string; name: string }[]>([]);
  const [selectedProductId, setSelectedProductId] = useState<string | null>(null);
  const [personas, setPersonas] = useState([]);
  const [showBuilder, setShowBuilder] = useState(false);
  const [statusMsg, setStatusMsg] = useState("");
=======
const drawerWidth = 240;

export default function Personas() {
>>>>>>> Stashed changes
  const token = localStorage.getItem("token");
  const [products, setProducts] = useState<{ id: string; name: string }[]>([]);
  const [selectedProductId, setSelectedProductId] = useState<string>("");
  const [cards, setCards] = useState<PersonaCardRCSPayload[]>([]);
  const [statusMsg, setStatusMsg] = useState("");
  const [loading, setLoading] = useState(false);

  // company + products
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

  // fetch personas (RCS-based)
  useEffect(() => {
<<<<<<< Updated upstream
    console.log("🔄 Working with product:", selectedProductId);
    if (!selectedProductId) {
      console.log("❌ No product selected, skipping persona fetch.");
      return;
    }
    const fetchPersonas = async () => {
      setPersonas([]); // Clear old data immediately
      setStatusMsg("Fetching personas...");
=======
    if (!selectedProductId) return;
    setLoading(true);
    (async () => {
>>>>>>> Stashed changes
      try {
        const data: PersonaCardRCSPayload[] = await fetch(
          `http://localhost:8000/get-personas/${selectedProductId}`,
<<<<<<< Updated upstream
          {
            headers: { Authorization: `Bearer ${token}` },
          }
        );
        const personaData = await personaRes.json();

        // 5. Handle backend messages
        if (personaData?.detail === "No Summaries or Capabilities Mapped") {
          setStatusMsg("No summaries or capabilities mapped. Please run Value Prop first.");
          setPersonas([]);
          return;
        }
=======
          { headers: { Authorization: `Bearer ${token}` } }
        ).then(r => r.json());
>>>>>>> Stashed changes

        // Expecting the new per-node payload from get_personas_rcs_priority
        console.log("Fetched personas data:", data);
        const mapped = (data || []).map(card => ({
          ...card,
          persona: {
            title: card.persona_title,
            department: Array.isArray(card.persona_departments) ? card.persona_departments[0] : card.persona_departments,
            seniority: Array.isArray(card.persona_seniority) ? card.persona_seniority[0] : card.persona_seniority,
          },
          jobs: (card.jobs || []).map(j => typeof j === "string" ? JSON.parse(j) : j),
          pains: (card.pains || []).map(p => typeof p === "string" ? JSON.parse(p) : p),
        }));
        setCards(mapped.slice().sort((a, b) => b.priority_score - a.priority_score));
        setStatusMsg("");
<<<<<<< Updated upstream
        setPersonas(personaData);

        // 6. If no personas, prompt to infer
        if (!personaData || personaData.length === 0) {
          setStatusMsg("No personas found. Click 'Infer Personas' to generate.");
        }
      } catch (err) {
=======
      } catch {
>>>>>>> Stashed changes
        setStatusMsg("Error fetching personas.");
        setCards([]);
      } finally {
        setLoading(false);
      }
    })();
  }, [selectedProductId, token]);

  return (
<<<<<<< Updated upstream
    <div className="p-8">
      <h2 className="text-2xl font-bold mb-6">Saved Personas</h2>

      {/* 3. If multiple products, let user select */}
      {products.length > 1 && (
        <div className="mb-4">
          <label className="mr-2 font-semibold">Select Product:</label>
          <select
            value={selectedProductId || ""}
            onChange={e => setSelectedProductId(e.target.value)}
            className="border rounded px-2 py-1"
          >
            <option value="" disabled>Select a product</option>
            {products.map(p => (
              <option key={p.id} value={p.id}>{p.name}</option>
            ))}
          </select>
        </div>
      )}

      {/* 4. If no products, show message */}
      {statusMsg && <div className="mb-4 text-red-600">{statusMsg}</div>}


      {/* 5. Show personas if they exist */}
      {personas.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {personas.map((p, i) => (
            <PersonaCard
              key={i}
              persona={p}
              isSelected={true}
              onToggle={() => {}}
            />
=======
    <Box sx={{ display: "flex" }}>

      <Box component="main" sx={{ flexGrow: 1, p: 3 }}>
        <Toolbar />
        <Typography variant="h5" fontWeight={700} gutterBottom>Personas</Typography>

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

        <Grid container spacing={2}>
          {cards.map((card) => (
            <Grid size={{xs:12, md:6, lg:4}} key={`${card.persona_title}-${card.persona_departments?.[0] || ""}-${card.persona_seniority?.[0] || ""}`}>
              <PersonaMuiCard data={card} />
            </Grid>
>>>>>>> Stashed changes
          ))}
        </Grid>
      </Box>
    </Box>
  );
}

<<<<<<< Updated upstream
export default Personas;
=======
function PersonaMuiCard({ data }: { data: PersonaCardRCSPayload }) {
  const { persona, importance, activation, care, marginal_lift, priority_score, jobs, pains } = data;

  return (
    <Card variant="outlined" sx={{ height: "100%", display: "flex", flexDirection: "column" }}>
      <CardHeader
        title={
          <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap">
            <Typography variant="subtitle1" fontWeight={700}>{persona.title}</Typography>
            {persona.department && <Chip size="small" label={persona.department} />}
            {persona.seniority && <Chip size="small" label={persona.seniority} />}
          </Stack>
        }
        subheader={<Typography variant="caption">Priority: {(priority_score).toFixed(3)}</Typography>}
      />
      <CardContent sx={{ pt: 0 }}>
        <Stack spacing={0.5} sx={{ mb: 1 }}>
          <Typography variant="body2">Importance (involvement): <b>{importance.toFixed(3)}</b></Typography>
          <Typography variant="body2">Activation: <b>{activation.toFixed(3)}</b></Typography>
          <Typography variant="body2">Care: <b>{care.toFixed(3)}</b></Typography>
          <Typography variant="body2">Marginal lift: <b>{marginal_lift.toFixed(3)}</b></Typography>
        </Stack>

        <Typography variant="subtitle2" sx={{ mt: 1 }}>Top Jobs</Typography>
        {jobs.slice(0, 4).map((j, i) => (
          <Typography key={i} variant="body2" color="text.secondary">• {j.description} (rel {j.relevance.toFixed(2)})</Typography>
        ))}

        <Typography variant="subtitle2" sx={{ mt: 1 }}>Top Pains</Typography>
        {pains.slice(0, 4).map((p, i) => (
          <Typography key={i} variant="body2" color="text.secondary">• {p.description} (rel {p.relevance.toFixed(2)})</Typography>
        ))}
      </CardContent>
    </Card>
  );
}
>>>>>>> Stashed changes
