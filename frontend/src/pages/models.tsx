import { ReactNode, useEffect, useState } from "react";
import {
  Box,
  Tabs,
  Tab,
  Stack,
  Paper,
  Typography,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
} from "@mui/material";
import CRMWinModels from "../components/model_ui/crm_win_models";

type ParameterEstimate = {
  name: string;
  description: string;
};

type SampleExample = {
  title: string;
  scenario: string;
  insight: string;
};

type ModelNarrativeContent = {
  id: string;
  label: string;
  model: string;
  math: string[];
  usedFor: string[];
  description: string;
  explains: string[];
  parameters: ParameterEstimate[];
  extraNotes?: string[];
  example?: SampleExample;
};

const STATIC_MODELS: ModelNarrativeContent[] = [
  {
    id: "bayesian-hmm",
    label: "Bayesian Graph Network + HMM",
    model:
      "Bayesian updater for belief-graph edges combined with a Hidden Markov Model (HMM) for persona belief transitions.",
    math: [
      "Bayesian edge update:\nP(E_new) = [P(Data | E_old) · P(E_old)] / P(Data)",
      "HMM state transition:\nAᵢⱼ = P(S(t+1) = j | S(t) = i)\nBelief states: {Unaware, ZMOT, Discovery, Evaluation, Pilot, Pre-Close, Customer}",
    ],
    usedFor: [
      "Re-weighting persona influence and pain criticality",
      "Updating asset → belief influence factors",
      "Tracking belief evolution over time and correcting survivorship bias",
    ],
    description:
      "This keeps the belief graph grounded in observed reality. Persona importance, pain strength, and belief transitions update continuously so the system reacts whenever engagement evidence disagrees with the original assumptions.",
    explains: [
      "Who matters (posterior persona importance by segment)",
      "What belief state each persona currently occupies",
      "Which transitions actually drive conversion and in what sequence",
      "Which coalition of personas reliably closes deals",
    ],
    parameters: [
      { name: "Prior edge probabilities", description: "Graph priors seeded from strategy + canon." },
      {
        name: "Likelihood weights",
        description: "Evidence strength derived from parsed engagements (channel × verb × asset).",
      },
      { name: "Posterior edge probabilities", description: "Updated persona/pain influence weights." },
      { name: "HMM transition matrix", description: "State-to-state probabilities between belief stages." },
      { name: "State persistence weights", description: "How long each state tends to hold without new proof." },
      { name: "Observation confidence scores", description: "Penalty for noisy or inferred engagements." },
      { name: "Decay rate", description: "Forgets stale evidence so new signals dominate." },
      { name: "Local vs global gain", description: "Controls whether updates stay account-specific or generalize." },
    ],
    extraNotes: ["Updated nightly after ingestion batches finish.", "Supports per-product hyperparameters."],
    example: {
      title: "Ops persona jumps ahead of RevOps",
      scenario:
        "Several enterprise FinServ accounts showed Operators pushing evaluation before RevOps weighed in.",
      insight:
        "Posterior weights increased the Operator persona importance +12% while decreasing RevOps weight, so the planner now opens Operator-led interventions earlier.",
    },
  },
  {
    id: "knn",
    label: "kNN Asset → Belief Matching",
    model:
      "Modified k-Nearest Neighbors over embedding space to match assets to belief transitions and persona-pain contexts.",
    math: [
      "Cosine similarity:\nsim(a, b) = (a · b) / (||a|| · ||b||)",
      "Asset score:\nScore(asset) = Σ(k=1→K) [wₖ · sim(asset, beliefₖ)]",
    ],
    usedFor: [
      "Selecting the best asset types, formats, and messages for a belief transition",
      "Ranking assets by expected impact on a persona’s belief state",
      "Detecting arsenal gaps plus cold-start recommendations",
    ],
    description:
      "Treats assets as behavioral instruments. New content inherits behavior from the nearest historical assets, so GI can recommend something immediately even when little engagement history exists.",
    explains: [
      "Which assets to deploy for each persona × belief pairing",
      "What formats work best per stage (events, webinars, briefs, etc.)",
      "Where coverage gaps exist and when to create net-new assets",
    ],
    parameters: [
      { name: "Embedding backbone", description: "Sentence-T5 fine-tuned on arsenal × belief corpora." },
      { name: "Vector dimension", description: "768-d embeds normalized per asset channel." },
      { name: "k (neighbors)", description: "Typically 12 with distance-weighted voting." },
      { name: "Similarity cutoff", description: "0.68 default; assets below this require human review." },
      { name: "Weighting scheme", description: "Max-of-top-3 × mean-of-rest hybrid for stability." },
      { name: "Asset confidence", description: "Blend of similarity, engagement proof, and recency." },
      { name: "Cold-start fallback", description: "Defaults to canonical exemplars per asset type + persona." },
    ],
    extraNotes: ["Re-scores assets as soon as new items enter the arsenal or metadata changes."],
    example: {
      title: "Matching a new ‘Data Trust Webinar’",
      scenario:
        "Fresh webinar with little history needs a home. kNN finds proximity to high-performing ‘Data Quality Summit’ content for compliance personas.",
      insight:
        "Planner surfaces it inside compliance concerns with 0.82 fitness even before direct engagement proof lands.",
    },
  },
  {
    id: "decay",
    label: "Exponential Decay & Engagement Fatigue",
    model:
      "Time-based exponential decay applied to belief reinforcement, engagement effectiveness, and persona responsiveness.",
    math: ["Impact(t) = Impact₀ · e^(−λ·t)"],
    usedFor: [
      "Engagement cadence optimization",
      "Preventing over-messaging and fatigue",
      "Deciding when to intensify, pause, or diversify persona paths",
    ],
    description:
      "Adds temporal realism. Beliefs cool down if not reinforced, repeated exposure loses punch, and GI learns when to persist versus pivoting toward alternate personas.",
    explains: [
      "How often to engage each persona before diminishing returns",
      "When to stop pressing a specific coalition member",
      "When to open alternate paths to progress the deal",
    ],
    parameters: [
      { name: "λ (decay constant)", description: "Per belief transition, estimated from historical reopen rates." },
      { name: "Saturation threshold", description: "Upper bound before the channel is considered fatigued." },
      { name: "Minimum effective exposure", description: "Touches required before the belief meaningfully shifts." },
      { name: "Cooldown duration", description: "Time to wait before retrying the same persona." },
      { name: "Reactivation multiplier", description: "Boost applied after cooldown if engagement restarts." },
      { name: "Cross-persona spillover", description: "How one persona’s saturation affects adjacent personas." },
    ],
    extraNotes: ["λ updates weekly; cooldown logic reacts instantly to bursty engagement."],
    example: {
      title: "Preventing fatigue for Security leadership",
      scenario:
        "Security persona ignored three touches in eight days. Decay model flagged λ=0.42 and triggered a 14-day cooldown.",
      insight:
        "Planner automatically shifted spend toward adjacent Data Ops personas until security interest recovered.",
    },
  },
  {
    id: "coverage",
    label: "Model Coverage Summary",
    model:
      "Explains how the modeling stack answers strategist questions so people know which signal to trust for each decision.",
    math: [],
    usedFor: [
      "Diagnosing which statistical layer is informing any recommendation",
      "Describing the evidence chain behind planner and arsenal guidance",
    ],
    description:
      "Summarizes how the models combine: GraphWin + Bayesian network explain who matters; the HMM reveals state/order; kNN dictates what to run; decay governs how often to engage.",
    explains: [
      "Who matters → GraphWin + Bayesian Graph Network",
      "In what order / coalition → Bayesian Graph Network + HMM",
      "Belief state progress → HMM",
      "What to use → kNN asset matching",
      "How often to engage → Exponential Decay",
    ],
    parameters: [
      { name: "Dependency graph", description: "Wiring between planner surfaces and underlying models." },
      { name: "Refresh cadence", description: "GraphWin (wins), Bayesian (nightly), kNN (continuous), Decay (weekly)." },
      { name: "Data coverage", description: "Episodes, engagements, and arsenal counts fueling each model." },
    ],
    example: {
      title: "Tracing an intervention recommendation",
      scenario:
        "Planner card cites ‘Data Analyst • Concern: fragmented lineage’. Coverage doc shows GraphWin + Bayesian for persona, kNN for asset, Decay for cadence.",
      insight:
        "Gives RevOps reviewer confidence about which telemetry would change the recommendation if challenged.",
    },
  },
];

const graphWinNarrative: ModelNarrativeContent = {
  id: "graphwin",
  label: "GraphWin Calibration",
  model:
    "Lightweight regression calibration layer comparing belief-graph predicted win probability vs actual outcomes, conditioned on account metadata.",
  math: [
    "Baseline linear form:\nP̂(win) = β₀ + β₁·P(graph) + β₂·X₁ + β₃·X₂ + … + βₙ·Xₙ",
    "Residual learning:\nε = P(actual) − P̂(win)\nwhere P(graph) is the belief-graph win probability\nand Xₙ are metadata features (industry, ARR, geo, segment, maturity).",
  ],
  usedFor: [
    "Calibrating belief-driven predictions against real outcomes",
    "Detecting systematic bias by segment",
    "Improving cohort-level win lift, persona prioritization, and execution confidence",
  ],
  description:
    "Keeps the belief graph anchored to reality. GraphWin flags when the graph over/under-estimates win likelihood for a specific ICP and applies a regression correction so forecasts stay grounded.",
  explains: [
    "Who matters (persona weight scaling by ICP)",
    "Expected win lift by account cluster",
    "Confidence bounds for execution planning",
  ],
  parameters: [
    { name: "β₀ (intercept)", description: "Baseline log-odds of a win when every feature is zeroed." },
    { name: "β₁ · P(graph)", description: "Slope aligning belief graph probability with observed close rate." },
    { name: "β_industry dummies", description: "Offsets per industry (SaaS, FinServ, Public Sector, etc.)." },
    { name: "β_revenue_range", description: "Adjustments for ARR bands (50–100M, 100–500M, >1B)." },
    { name: "β_geography", description: "Regional corrections (NA, EMEA, APAC, LATAM)." },
    { name: "β_employee_band", description: "Scaling for org size (0–500, 500–1k, 1k–5k, 5k+)." },
    { name: "Fit metrics", description: "R², RMSE, MAE stored per refresh to monitor drift." },
    { name: "Sample size", description: "Wins/loss count plus last rebuild timestamp." },
  ],
  extraNotes: ["Refreshes whenever new wins/losses arrive or the belief graph changes materially."],
  example: {
    title: "North America SaaS 1–5K cluster",
    scenario:
      "Belief graph predicted 0.62 win probability but CRM outcomes showed 0.48 across 73 opportunities.",
    insight:
      "GraphWin learned β₁=0.74 and a −0.11 industry offset, so planner timelines now down-rank these accounts until corroborating proof appears.",
  },
};

const SectionCard = ({ title, children }: { title: string; children: ReactNode }) => (
  <Paper variant="outlined" sx={{ p: 2 }}>
    <Typography variant="subtitle2" sx={{ mb: 1, textTransform: "uppercase", letterSpacing: 1 }}>
      {title}
    </Typography>
    {children}
  </Paper>
);

const BulletList = ({ items }: { items: string[] }) => (
  <Stack
    component="ul"
    spacing={0.75}
    sx={{ pl: 2, listStyle: "disc", "& li": { fontSize: 14, color: "text.secondary" } }}
  >
    {items.map((item) => (
      <Typography component="li" key={item}>
        {item}
      </Typography>
    ))}
  </Stack>
);

const ParameterTable = ({ estimates }: { estimates: ParameterEstimate[] }) => (
  <Paper variant="outlined" sx={{ overflowX: "auto" }}>
    <Table size="small">
      <TableHead>
        <TableRow>
          <TableCell sx={{ fontWeight: "bold" }}>Parameter</TableCell>
          <TableCell sx={{ fontWeight: "bold" }}>What it captures</TableCell>
        </TableRow>
      </TableHead>
      <TableBody>
        {estimates.map((estimate) => (
          <TableRow key={estimate.name}>
            <TableCell sx={{ whiteSpace: "nowrap" }}>{estimate.name}</TableCell>
            <TableCell>{estimate.description}</TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  </Paper>
);

const ExampleCard = ({ example }: { example: SampleExample }) => (
  <Paper variant="outlined" sx={{ p: 2 }}>
    <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
      {example.title}
    </Typography>
    <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
      <strong>Scenario:</strong> {example.scenario}
    </Typography>
    <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
      <strong>Insight:</strong> {example.insight}
    </Typography>
  </Paper>
);

const ModelNarrative = ({ content }: { content: ModelNarrativeContent }) => (
  <Stack spacing={2}>
    <SectionCard title="Model">{content.model}</SectionCard>
    {content.math.length > 0 && (
      <SectionCard title="Math / Equations">
        <Stack spacing={1.5}>
          {content.math.map((snippet, idx) => (
            <Box
              key={`${content.id}-math-${idx}`}
              component="pre"
              sx={{
                backgroundColor: "#0f172a",
                color: "#bbf7d0",
                borderRadius: 1,
                p: 2,
                fontFamily: "JetBrains Mono, SFMono-Regular, Consolas, monospace",
                fontSize: 13,
                whiteSpace: "pre-wrap",
              }}
            >
              {snippet}
            </Box>
          ))}
        </Stack>
      </SectionCard>
    )}
    <SectionCard title="Used For">
      <BulletList items={content.usedFor} />
    </SectionCard>
    <SectionCard title="Description / GI Impact">
      <Typography variant="body2" color="text.secondary">
        {content.description}
      </Typography>
    </SectionCard>
    <SectionCard title="It Explains">
      <BulletList items={content.explains} />
    </SectionCard>
    <SectionCard title="Estimated Parameters">
      <ParameterTable estimates={content.parameters} />
    </SectionCard>
    {content.extraNotes && content.extraNotes.length > 0 && (
      <SectionCard title="Operational Notes">
        <BulletList items={content.extraNotes} />
      </SectionCard>
    )}
    {content.example && (
      <SectionCard title="Example">
        <ExampleCard example={content.example} />
      </SectionCard>
    )}
  </Stack>
);

type ModelTab = {
  id: string;
  label: string;
  render: (args: { productId: string | null; token: string | null }) => ReactNode;
};

const MODEL_TABS: ModelTab[] = [
  {
    id: "graphwin",
    label: "GraphWin",
    render: ({ productId, token }) => (
      <Stack spacing={3}>
        <ModelNarrative content={graphWinNarrative} />
        <SectionCard title="Live Regression Diagnostics">
          <CRMWinModels productId={productId ?? ""} token={token ?? ""} />
        </SectionCard>
      </Stack>
    ),
  },
  ...STATIC_MODELS.map<ModelTab>((content) => ({
    id: content.id,
    label: content.label,
    render: () => <ModelNarrative content={content} />,
  })),
];

const Models = () => {
  const [products, setProducts] = useState<{ id: string; name: string }[]>([]);
  const [selectedProductId, setSelectedProductId] = useState<string | null>(null);
  const [statusMsg, setStatusMsg] = useState("");
  const [rebuildStatus, setRebuildStatus] = useState<string>("");
  const [activeModelId, setActiveModelId] = useState<string>(MODEL_TABS[0].id);
  const token = localStorage.getItem("token");

  useEffect(() => {
    const fetchCompanyAndProducts = async () => {
      try {
        const meRes = await fetch("http://localhost:8000/me", {
          headers: { Authorization: `Bearer ${token}` },
        });
        const meData = await meRes.json();

        const prodRes = await fetch(`http://localhost:8000/get-products/${meData.company_id}`, {
          headers: { Authorization: `Bearer ${token}` },
        });
        const prodData = await prodRes.json();
        if (!prodData.products || prodData.products.length === 0) {
          setStatusMsg("No products found. Please run Value Prop first.");
          return;
        }
        setProducts(prodData.products);

        if (prodData.products.length > 0) {
          setSelectedProductId((prev) => prev ?? prodData.products[0].id);
        }
      } catch (err) {
        setStatusMsg("Error fetching company or products.");
      }
    };
    fetchCompanyAndProducts();
  }, [token]);

  const triggerGlobalThesisRebuild = async () => {
    if (!token) {
      setRebuildStatus("Missing auth token.");
      return;
    }
    if (!selectedProductId) {
      setRebuildStatus("Select a product first.");
      return;
    }
    try {
      setRebuildStatus("Rebuilding global thesis…");
      const res = await fetch(
        `http://localhost:8000/journey/rebuild-global-thesis/${selectedProductId}`,
        {
          method: "POST",
          headers: {
            Authorization: `Bearer ${token}`,
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ window: 1000 }),
        }
      );
      if (!res.ok) {
        const txt = await res.text();
        throw new Error(txt || "Failed to rebuild thesis");
      }
      const data = await res.json();
      const episodes = data?.thesis?.meta?.num_episodes ?? "?";
      setRebuildStatus(`Global thesis rebuilt (episodes analyzed: ${episodes}).`);
    } catch (err: any) {
      setRebuildStatus(err?.message || "Global thesis rebuild failed.");
    }
  };

  const handleTabChange = (_: React.SyntheticEvent, newValue: string) => {
    setActiveModelId(newValue);
  };

  const activeTab = MODEL_TABS.find((tab) => tab.id === activeModelId) ?? MODEL_TABS[0];

  return (
    <Box sx={{ p: 4 }}>
      {products.length > 1 && (
        <Box sx={{ mb: 2, display: "flex", alignItems: "center", gap: 2 }}>
          <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
            Select Product:
          </Typography>
          <select
            value={selectedProductId || ""}
            onChange={(e) => setSelectedProductId(e.target.value)}
            className="border rounded px-2 py-1"
          >
            <option value="" disabled>
              Select a product
            </option>
            {products.map((p) => (
              <option key={p.id} value={p.id}>
                {p.name}
              </option>
            ))}
          </select>
        </Box>
      )}
      {statusMsg && (
        <Typography sx={{ mb: 2 }} color="error">
          {statusMsg}
        </Typography>
      )}
      <Box sx={{ mb: 3, display: "flex", alignItems: "center", gap: 2 }}>
        <button
          className="bg-indigo-600 text-white px-3 py-2 rounded disabled:opacity-50"
          onClick={triggerGlobalThesisRebuild}
          disabled={!selectedProductId}
        >
          Rebuild Journey Thesis
        </button>
        {rebuildStatus && (
          <Typography variant="body2" color="text.secondary">
            {rebuildStatus}
          </Typography>
        )}
      </Box>

      <Box sx={{ borderBottom: 1, borderColor: "divider", mb: 3 }}>
        <Tabs
          value={activeModelId}
          onChange={handleTabChange}
          textColor="primary"
          indicatorColor="primary"
          variant="scrollable"
          scrollButtons="auto"
        >
          {MODEL_TABS.map((tab) => (
            <Tab key={tab.id} label={tab.label} value={tab.id} />
          ))}
        </Tabs>
      </Box>

      <Box sx={{ mt: 2 }}>{activeTab.render({ productId: selectedProductId, token })}</Box>
    </Box>
  );
};

export default Models;
