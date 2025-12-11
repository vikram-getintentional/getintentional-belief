import React, { useEffect, useMemo, useState } from "react";
import {
  Box,
  Button,
  Card,
  CardActionArea,
  CardContent,
  Chip,
  Grid,
  LinearProgress,
  Stack,
  Typography,
  FormControl,
  InputLabel,
  MenuItem,
  Select,
  Toolbar,
  Divider,
  Paper,
} from "@mui/material";

const SHOW_BATCH = 6;

type PersonaFrequency = {
  label: string;
  frequency: number;
};

type PersonaSamplePerson = {
  person_name?: string | null;
  title?: string | null;
  department?: string | null;
  seniority?: string | null;
  account_name?: string | null;
};

type PersonaSampleAccount = {
  account_id?: string;
  account_name?: string | null;
  meta?: Record<string, string | null | undefined>;
};

type PersonaInsight = {
  persona_id: string;
  label: string;
  title?: string | null;
  department?: string | null;
  seniority?: string | null;
  scores: {
    perceptibility?: number;
    proximity?: number;
    involvement?: number;
    wolves_score?: number;
    wolves_delta_bp?: number;
    wolves_involvement_rate?: number;
    wolves_blocker_rate?: number;
  };
  concerns: Array<{
    label?: string | null;
    stage?: string | null;
    phase?: string | null;
    score?: number | null;
  }>;
  coalitions: Array<{
    persona_id: string;
    persona_label: string;
    frequency: number;
  }>;
  seen_in: {
    overall: PersonaFrequency;
    segments: PersonaFrequency[];
  };
  people_samples: PersonaSamplePerson[];
  account_samples: PersonaSampleAccount[];
};

type PersonaInsightsResponse = {
  product_id: string;
  wolves_metrics_updated_at?: string | null;
  personas: PersonaInsight[];
};

const formatPercent = (value?: number | null) => {
  if (value === null || value === undefined) return "—";
  return `${Math.round(value * 100)}%`;
};

const scoreLabel = (value?: number | null) =>
  value === null || value === undefined ? "—" : value.toFixed(2);

const PersonasPage: React.FC = () => {
  const token =
    typeof window !== "undefined" ? localStorage.getItem("token") : null;
  const [products, setProducts] = useState<Array<{ id: string; name: string }>>([]);
  const [selectedProductId, setSelectedProductId] = useState<string>("");
  const [personas, setPersonas] = useState<PersonaInsight[]>([]);
  const [loading, setLoading] = useState(false);
  const [statusMsg, setStatusMsg] = useState<string>("");
  const [visibleCount, setVisibleCount] = useState(SHOW_BATCH);
  const [selectedPersona, setSelectedPersona] = useState<PersonaInsight | null>(null);

  useEffect(() => {
    if (!token) return;
    (async () => {
      try {
        const me = await fetch("http://localhost:8000/me", {
          headers: { Authorization: `Bearer ${token}` },
        }).then((r) => r.json());
        const prod = await fetch(
          `http://localhost:8000/get-products/${me.company_id}`,
          { headers: { Authorization: `Bearer ${token}` } }
        ).then((r) => r.json());
        const list = prod.products ?? [];
        setProducts(list);
        if (list.length === 1) {
          setSelectedProductId(list[0].id);
        }
      } catch (err) {
        console.error(err);
        setStatusMsg("Unable to load products. Please refresh.");
      }
    })();
  }, [token]);

  useEffect(() => {
    if (!selectedProductId || !token) return;
    setLoading(true);
    setStatusMsg("");
    setSelectedPersona(null);
    setVisibleCount(SHOW_BATCH);
    (async () => {
      try {
        const data: PersonaInsightsResponse = await fetch(
          `http://localhost:8000/products/${selectedProductId}/persona_insights`,
          { headers: { Authorization: `Bearer ${token}` } }
        ).then((r) => {
          if (!r.ok) throw new Error(r.statusText);
          return r.json();
        });
        const sorted = (data.personas || []).slice().sort((a, b) => {
          const aw = a.scores?.wolves_score ?? 0;
          const bw = b.scores?.wolves_score ?? 0;
          if (bw !== aw) return bw - aw;
          const ai = a.scores?.involvement ?? 0;
          const bi = b.scores?.involvement ?? 0;
          return bi - ai;
        });
        setPersonas(sorted);
        if (!sorted.length) {
          setStatusMsg("No personas found for this product yet.");
        }
      } catch (err) {
        console.error(err);
        setStatusMsg("Unable to load personas for this product.");
        setPersonas([]);
      } finally {
        setLoading(false);
      }
    })();
  }, [selectedProductId, token]);

  const visiblePersonas = useMemo(
    () => personas.slice(0, visibleCount),
    [personas, visibleCount]
  );

  const handleShowMore = () => {
    setVisibleCount((prev) => Math.min(personas.length, prev + SHOW_BATCH));
  };

  const handleShowLess = () => {
    setVisibleCount(SHOW_BATCH);
  };

  return (
    <Box sx={{ display: "flex" }}>
      <Box component="main" sx={{ flexGrow: 1, p: 3 }}>
        <Toolbar />
        <Stack direction="row" justifyContent="space-between" alignItems="center">
          <Typography variant="h5" fontWeight={700} gutterBottom>
            Personas
          </Typography>
          {products.length > 1 && (
            <FormControl size="small" sx={{ minWidth: 240 }}>
              <InputLabel id="persona-prod-label">Select Product</InputLabel>
              <Select
                labelId="persona-prod-label"
                value={selectedProductId}
                label="Select Product"
                onChange={(e) => setSelectedProductId(e.target.value)}
              >
                {products.map((product) => (
                  <MenuItem key={product.id} value={product.id}>
                    {product.name}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          )}
        </Stack>

        {statusMsg && (
          <Typography color="error" sx={{ mb: 2 }}>
            {statusMsg}
          </Typography>
        )}
        {loading && <LinearProgress sx={{ my: 2 }} />}

        {selectedPersona ? (
          <PersonaDetail persona={selectedPersona} onBack={() => setSelectedPersona(null)} />
        ) : (
          <>
            <Grid container spacing={2}>
              {visiblePersonas.map((persona) => (
                <Grid item xs={12} md={6} lg={4} key={persona.persona_id}>
                  <PersonaCard persona={persona} onSelect={setSelectedPersona} />
                </Grid>
              ))}
            </Grid>
            {!loading && personas.length > visibleCount && (
              <Box sx={{ mt: 2, textAlign: "center" }}>
                <Button variant="outlined" onClick={handleShowMore}>
                  Show more personas
                </Button>
              </Box>
            )}
            {!loading && personas.length > SHOW_BATCH && visibleCount > SHOW_BATCH && (
              <Box sx={{ mt: 1, textAlign: "center" }}>
                <Button variant="text" onClick={handleShowLess}>
                  Show fewer
                </Button>
              </Box>
            )}
          </>
        )}
      </Box>
    </Box>
  );
};

const PersonaCard: React.FC<{
  persona: PersonaInsight;
  onSelect: (persona: PersonaInsight) => void;
}> = ({ persona, onSelect }) => {
  const { scores, seen_in } = persona;
  return (
    <Card variant="outlined">
      <CardActionArea onClick={() => onSelect(persona)} sx={{ p: 2 }}>
        <Stack spacing={1}>
          <Stack spacing={0.5}>
            <Typography variant="subtitle1" fontWeight={700}>
              {persona.title || persona.label}
            </Typography>
            <Stack direction="row" spacing={0.5} flexWrap="wrap">
              {persona.department && <Chip size="small" label={persona.department} />}
              {persona.seniority && <Chip size="small" label={persona.seniority} />}
            </Stack>
          </Stack>
          <Stack direction="row" spacing={1} flexWrap="wrap">
            <Chip
              size="small"
              color="primary"
              label={`Wolves ${scoreLabel(scores.wolves_score)}`}
            />
            <Chip size="small" label={`Perceptibility ${scoreLabel(scores.perceptibility)}`} />
            <Chip size="small" label={`Proximity ${scoreLabel(scores.proximity)}`} />
            <Chip size="small" label={`Involvement ${scoreLabel(scores.involvement)}`} />
          </Stack>
          <Box>
            <Typography variant="caption" color="text.secondary">
              Seen in
            </Typography>
            <Typography variant="body2">
              {seen_in.overall.label}: {formatPercent(seen_in.overall.frequency)}
            </Typography>
            {seen_in.segments.slice(0, 1).map((segment) => (
              <Typography key={segment.label} variant="body2" color="text.secondary">
                {segment.label}: {formatPercent(segment.frequency)}
              </Typography>
            ))}
          </Box>
        </Stack>
      </CardActionArea>
    </Card>
  );
};

const PersonaDetail: React.FC<{ persona: PersonaInsight; onBack: () => void }> = ({
  persona,
  onBack,
}) => {
  const { scores, seen_in, people_samples, account_samples, concerns, coalitions } = persona;
  return (
    <Box>
      <Button variant="text" size="small" sx={{ mb: 2 }} onClick={onBack}>
        ← Back to personas
      </Button>
      <Paper variant="outlined" sx={{ p: 3 }}>
        <Stack spacing={2}>
          <Stack spacing={1}>
            <Typography variant="h5" fontWeight={700}>
              {persona.label}
            </Typography>
            <Stack direction="row" spacing={1} flexWrap="wrap">
              {persona.title && <Chip label={persona.title} size="small" />}
              {persona.department && <Chip label={persona.department} size="small" />}
              {persona.seniority && <Chip label={persona.seniority} size="small" />}
            </Stack>
          </Stack>

          <Box>
            <Typography variant="subtitle2">Scores</Typography>
            <Stack direction="row" spacing={1} flexWrap="wrap" sx={{ mt: 0.5 }}>
              <Chip label={`Perceptibility ${scoreLabel(scores.perceptibility)}`} />
              <Chip label={`Proximity ${scoreLabel(scores.proximity)}`} />
              <Chip label={`Involvement ${scoreLabel(scores.involvement)}`} />
              <Chip label={`Wolves ${scoreLabel(scores.wolves_score)}`} color="primary" />
              {scores.wolves_delta_bp !== undefined && scores.wolves_delta_bp !== null && (
                <Chip label={`Δ Win ${scoreLabel(scores.wolves_delta_bp)}`} />
              )}
            </Stack>
          </Box>

          <Box>
            <Typography variant="subtitle2">Seen in</Typography>
            <Typography variant="body2">
              {seen_in.overall.label}: {formatPercent(seen_in.overall.frequency)}
            </Typography>
            {seen_in.segments.map((segment) => (
              <Typography key={segment.label} variant="body2" color="text.secondary">
                {segment.label}: {formatPercent(segment.frequency)}
              </Typography>
            ))}
          </Box>

          {!!coalitions.length && (
            <Box>
              <Typography variant="subtitle2">Coalitions</Typography>
              <Stack direction="row" spacing={0.5} flexWrap="wrap" sx={{ mt: 0.5 }}>
                {coalitions.map((coal) => (
                  <Chip
                    key={`${persona.persona_id}-coal-${coal.persona_id}`}
                    label={`${coal.persona_label} (${formatPercent(coal.frequency)})`}
                    variant="outlined"
                  />
                ))}
              </Stack>
            </Box>
          )}

          {!!concerns.length && (
            <Box>
              <Typography variant="subtitle2">Typical Concerns</Typography>
              <Stack spacing={0.5} sx={{ mt: 0.5 }}>
                {concerns.map((concern, idx) => (
                  <Typography key={`${persona.persona_id}-concern-${idx}`} variant="body2">
                    • {concern.label || "Concern"}
                    {concern.stage ? ` (${concern.stage})` : ""}
                  </Typography>
                ))}
              </Stack>
            </Box>
          )}

          <Grid container spacing={2}>
            <Grid item xs={12} md={6}>
              <Typography variant="subtitle2">Sample People</Typography>
              {people_samples.length ? (
                <Stack spacing={0.5} sx={{ mt: 1 }}>
                  {people_samples.map((person, idx) => (
                    <Paper key={`${persona.persona_id}-person-${idx}`} variant="outlined" sx={{ p: 1 }}>
                      <Typography variant="body2" fontWeight={600}>
                        {person.person_name || "Unknown"}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        {person.title || "—"}
                        {person.account_name ? ` · ${person.account_name}` : ""}
                      </Typography>
                    </Paper>
                  ))}
                </Stack>
              ) : (
                <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                  No mapped people yet.
                </Typography>
              )}
            </Grid>
            <Grid item xs={12} md={6}>
              <Typography variant="subtitle2">Sample Accounts</Typography>
              {account_samples.length ? (
                <Stack spacing={0.5} sx={{ mt: 1 }}>
                  {account_samples.map((account, idx) => (
                    <Paper key={`${persona.persona_id}-acct-${idx}`} variant="outlined" sx={{ p: 1 }}>
                      <Typography variant="body2" fontWeight={600}>
                        {account.account_name || "Account"}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        {META_KEYS.map((key) => account.meta?.[key]).filter(Boolean).join(" · ") || "—"}
                      </Typography>
                    </Paper>
                  ))}
                </Stack>
              ) : (
                <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                  No account examples yet.
                </Typography>
              )}
            </Grid>
          </Grid>
        </Stack>
      </Paper>
    </Box>
  );
};

const META_KEYS = ["industry", "geography", "revenue_range", "employee_range", "funding_stage"];

export default PersonasPage;
