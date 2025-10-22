import React, { useEffect, useMemo, useState } from "react";

import { Loader2, Plus, X } from "lucide-react";
import { Card, CardContent, CardHeader, Input, Select, FormControl, MenuItem, Button, Slider } from "@mui/material";
import { Label } from "@mui/icons-material";

// -----------------------------
// Types matching /simulate-rcs
// -----------------------------

type Timeframe = { startDate?: string; endDate?: string };

type ArsenalRow = {
  asset?: { id?: string; name?: string; format?: string; evergreen?: boolean };
  channel?: { id?: string; name?: string; type?: string; reach_score?: number };
  fitment?: string; // this is the concern stage label coming through today
  engagement?: string;
  expectedLift?: string;
  concernsAddressed?: string[];
  why?: string;
  breadthScore?: number;
};

type Campaign = {
  id: string;
  description: string;
  timeframe?: Timeframe;
  personas?: string[];
  arsenalTable?: ArsenalRow[];
};

type StagePlan = {
  stage: string;
  stageLabel: string;
  policy?: Record<string, number>;
  campaigns?: Campaign[];
  debug?: any;
};

type SimulationResponse = {
  generatedAt: string;
  keyStats: {
    personasTotal?: number;
    expectedWinsPct?: number;
    avgBelief?: string;
  };
  rcsReport: any;
  frozenStrategy: any;
  stagePlan: StagePlan;
  diffs?: {
    winLikelihoodDelta?: number;
    personasAdded?: string[];
    personasRemoved?: string[];
    campaignCountDelta?: number;
  };
};

// -----------------------------
// Small UI helpers
// -----------------------------

function Chip({ text, onRemove }: { text: string; onRemove?: () => void }) {
  return (
    <span className="inline-flex items-center gap-2 rounded-full border px-3 py-1 text-sm">
      {text}
      {onRemove && (
        <button className="rounded-full p-1 hover:bg-muted" onClick={onRemove}>
          <X className="h-3 w-3" />
        </button>
      )}
    </span>
  );
}

function Kpi({ label, value, helper }: { label: string; value: React.ReactNode; helper?: string }) {
  return (
    <Card className="shadow-sm">
      <CardHeader className="pb-2">
        <div className="text-sm font-medium text-muted-foreground">{label}</div>
      </CardHeader>
      <CardContent className="pt-0">
        <div className="text-2xl font-semibold">{value}</div>
        {helper && <div className="text-xs text-muted-foreground mt-1">{helper}</div>}
      </CardContent>
    </Card>
  );
}

// -----------------------------
// Main component
// -----------------------------

export default function RcsSimulator() {
  // Inputs
  const [companyId, setCompanyId] = useState<string | null>(null);
  const [products, setProducts] = useState<{ id: string; name: string }[]>([]);
  const [selectedProductId, setSelectedProductId] = useState<string | null>(null);
  const [statusMsg, setStatusMsg] = useState("");
  const token = localStorage.getItem("token");
  const [accountId, setAccountId] = useState("");
  const [productId, setProductId] = useState("");

  // 1. Get company ID on mount
    useEffect(() => {
      const fetchCompanyAndProducts = async () => {
        try {
          const meRes = await fetch("http://localhost:8000/me", {
            headers: { Authorization: `Bearer ${token}` },
          });
          const meData = await meRes.json();
          setCompanyId(meData.company_id);
  
          // 2. Get product IDs for this company
          const prodRes = await fetch(
            `http://localhost:8000/get-products/${meData.company_id}`,
            {
              headers: { Authorization: `Bearer ${token}` },
            }
          );
          const prodData = await prodRes.json();
          if (!prodData.products || prodData.products.length === 0) {
            setStatusMsg("No products found. Please run Value Prop first.");
            return;
          }
          console.log("✅ Product IDs set:", prodData.products.map((p: any) => p.id));
          setProducts(prodData.products);
  
          // 3. If only one product, select it automatically
          if (prodData.products.length === 1) {
            console.log("✅ Automatically selecting single product:", prodData.products[0].id);
            setSelectedProductId(prodData.products[0].id);
          }
        } catch (err) {
          setStatusMsg("Error fetching company or products.");
        }
      };
      fetchCompanyAndProducts();
    }, [token]);

  const [attributes, setAttributes] = useState<string[]>([]);
  const [zmots, setZmots] = useState<string[]>([]);
  const [personas, setPersonas] = useState<string[]>([]);

  const [stage, setStage] = useState("auto");
  const [playsPerStep, setPlaysPerStep] = useState<number>(2);
  const [boostFactor, setBoostFactor] = useState<number>(2.0);

  const [archetype, setArchetype] = useState({
    industry: "",
    geography: "",
    revenue_range: "",
    employee_range: "",
    funding_stage: "",
    competitors_used: "",
    tech_stack: "",
  });

  // Editing chips
  const [newAttr, setNewAttr] = useState("");
  const [newZmot, setNewZmot] = useState("");
  const [newPersona, setNewPersona] = useState("");

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<SimulationResponse | null>(null);

  // handlers for chips
  const addChip = (kind: "attr" | "zmot" | "persona") => {
    if (kind === "attr" && newAttr.trim()) {
      setAttributes((x) => Array.from(new Set([...x, newAttr.trim()])));
      setNewAttr("");
    }
    if (kind === "zmot" && newZmot.trim()) {
      const v = newZmot.trim().startsWith("zmot:") ? newZmot.trim() : `zmot:${newZmot.trim()}`;
      setZmots((x) => Array.from(new Set([...x, v])));
      setNewZmot("");
    }
    if (kind === "persona" && newPersona.trim()) {
      setPersonas((x) => Array.from(new Set([...x, newPersona.trim()])));
      setNewPersona("");
    }
  };

  const removeChip = (kind: "attr" | "zmot" | "persona", val: string) => {
    if (kind === "attr") setAttributes((x) => x.filter((a) => a !== val));
    if (kind === "zmot") setZmots((x) => x.filter((a) => a !== val));
    if (kind === "persona") setPersonas((x) => x.filter((a) => a !== val));
  };

  const totalConcerns = useMemo(() => {
    const seqs: any[] = data?.rcsReport?.concern_sequences || [];
    let count = 0;
    seqs.forEach((s) => (count += (s.sequence || []).length));
    return count;
  }, [data]);

  
  

  async function runSimulation() {
    setLoading(true);
    setError(null);
    setData(null);
    try {
      const resp = await fetch(`http://localhost:8000/simulate-rcs?product_id=${productId}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          account_id: accountId,
          product_id: productId,
          attributes,
          zmots,
          persona_engagements: personas,
          stage,
          plays_per_step: playsPerStep,
          boost_factor: boostFactor,
          archetype,
        }),
      });
      if (!resp.ok) throw new Error(`Simulation failed (${resp.status})`);
      const json: SimulationResponse = await resp.json();
      setData(json);
    } catch (e: any) {
      setError(e?.message || "Something went wrong");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="mx-auto max-w-7xl p-6 space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-semibold">RCS Simulator</h1>
        <div className="flex items-center gap-3">
          <Button onClick={runSimulation} disabled={loading}>
            {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Plus className="h-4 w-4" />} Run Simulation
          </Button>
        </div>
      </div>

      {/* Inputs */}
      <Card>
        <CardHeader>
          <div className="text-base">Scenario Inputs</div>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <Label>Account ID</Label>
              <Input placeholder="account GUID" value={accountId} onChange={(e) => setAccountId(e.target.value)} />
            </div>
            <div>
              <Label>Product ID</Label>
              <Input placeholder="product:*" value={productId} onChange={(e) => setProductId(e.target.value)} />
            </div>
            <div>
              <Label>Stage</Label>
              <FormControl fullWidth size="small">
                <Select value={stage} onChange={(e) => setStage((e.target as HTMLSelectElement).value as string)}>
                  <MenuItem value="auto">Auto</MenuItem>
                  <MenuItem value="pre_zmot">Pre‑ZMOT</MenuItem>
                  <MenuItem value="zmot">ZMOT</MenuItem>
                  <MenuItem value="problem_realization">Problem Realization</MenuItem>
                  <MenuItem value="discovery">Discovery</MenuItem>
                  <MenuItem value="barriers">Barriers</MenuItem>
                  <MenuItem value="implementation">Implementation</MenuItem>
                </Select>
              </FormControl>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="space-y-2">
              <Label>Attributes</Label>
              <div className="flex gap-2">
                <Input placeholder="industry:SaaS" value={newAttr} onChange={(e) => setNewAttr(e.target.value)}
                       onKeyDown={(e) => e.key === "Enter" && addChip("attr")} />
                <Button variant="secondary" onClick={() => addChip("attr")}>Add</Button>
              </div>
              <div className="mt-2 flex flex-wrap gap-2">
                {attributes.map((a) => (
                  <Chip key={a} text={a} onRemove={() => removeChip("attr", a)} />
                ))}
              </div>
            </div>

            <div className="space-y-2">
              <Label>ZMOT Signals</Label>
              <div className="flex gap-2">
                <Input placeholder="zmot:integration_rfp" value={newZmot} onChange={(e) => setNewZmot(e.target.value)}
                       onKeyDown={(e) => e.key === "Enter" && addChip("zmot")} />
                <Button variant="secondary" onClick={() => addChip("zmot")}>Add</Button>
              </div>
              <div className="mt-2 flex flex-wrap gap-2">
                {zmots.map((z) => (
                  <Chip key={z} text={z} onRemove={() => removeChip("zmot", z)} />
                ))}
              </div>
            </div>

            <div className="space-y-2">
              <Label>Engaged Personas (ids)</Label>
              <div className="flex gap-2">
                <Input placeholder="persona:..." value={newPersona} onChange={(e) => setNewPersona(e.target.value)}
                       onKeyDown={(e) => e.key === "Enter" && addChip("persona")} />
                <Button variant="secondary" onClick={() => addChip("persona")}>Add</Button>
              </div>
              <div className="mt-2 flex flex-wrap gap-2 max-h-24 overflow-auto">
                {personas.map((p) => (
                  <Chip key={p} text={p} onRemove={() => removeChip("persona", p)} />
                ))}
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div>
              <Label>Plays per step: {playsPerStep}</Label>
              <Slider defaultValue={playsPerStep} min={1} max={5} valueLabelDisplay="auto"
                      onChange={(_, v) => setPlaysPerStep(v as number)} />
            </div>
            <div>
              <Label>Boost factor: {boostFactor.toFixed(2)}</Label>
              <Slider defaultValue={boostFactor} min={1} max={4} step={0.25} valueLabelDisplay="auto"
                      onChange={(_, v) => setBoostFactor(v as number)} />
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <Label>Industry</Label>
              <Input value={archetype.industry} onChange={(e) => setArchetype({ ...archetype, industry: e.target.value })} />
            </div>
            <div>
              <Label>Geography</Label>
              <Input value={archetype.geography} onChange={(e) => setArchetype({ ...archetype, geography: e.target.value })} />
            </div>
            <div>
              <Label>Revenue Range</Label>
              <Input value={archetype.revenue_range} onChange={(e) => setArchetype({ ...archetype, revenue_range: e.target.value })} />
            </div>
            <div>
              <Label>Employee Range</Label>
              <Input value={archetype.employee_range} onChange={(e) => setArchetype({ ...archetype, employee_range: e.target.value })} />
            </div>
            <div>
              <Label>Funding Stage</Label>
              <Input value={archetype.funding_stage} onChange={(e) => setArchetype({ ...archetype, funding_stage: e.target.value })} />
            </div>
            <div>
              <Label>Competitors Used</Label>
              <Input value={archetype.competitors_used} onChange={(e) => setArchetype({ ...archetype, competitors_used: e.target.value })} />
            </div>
            <div className="md:col-span-3">
              <Label>Tech Stack</Label>
              <textarea rows={2} className="w-full rounded-md border p-2" value={archetype.tech_stack} onChange={(e) => setArchetype({ ...archetype, tech_stack: e.target.value })} />
            </div>
          </div>
        </CardContent>
      </Card>

      {error && (
        <Card className="border-destructive/50">
          <CardContent className="text-destructive py-4">{error}</CardContent>
        </Card>
      )}

      {/* Results */}
      {data && (
        <div className="space-y-6">
          {/* KPIs */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <Kpi label="Personas in Scope" value={data.keyStats?.personasTotal ?? 0} />
            <Kpi label="Expected Wins" value={`${Math.round((data.keyStats?.expectedWinsPct || 0) * 100)}%`} />
            <Kpi label="Avg Belief" value={data.keyStats?.avgBelief || "—"} />
            <Kpi label="Total Concerns" value={totalConcerns} />
          </div>

          {/* Diffs */}
          {data.diffs && (
            <Card>
              <CardHeader>
                <div className="text-base">Change vs Baseline</div>
              </CardHeader>
              <CardContent className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <Kpi label="Δ Win Likelihood" value={(data.diffs.winLikelihoodDelta ?? 0).toFixed(3)} />
                <Kpi label="Campaign Δ" value={data.diffs.campaignCountDelta ?? 0} />
                <div className="md:col-span-2">
                  <div className="text-sm font-medium mb-1">Personas Added</div>
                  <div className="flex flex-wrap gap-2">
                    {(data.diffs.personasAdded || []).map((p) => (
                      <span key={p} className="inline-flex items-center rounded-full bg-muted px-3 py-1 text-sm">{p}</span>
                    ))}
                  </div>
                </div>
              </CardContent>
            </Card>
          )}

          {/* Stage plan with campaigns */}
          <Card>
            <CardHeader>
              <div className="text-base">
                {data.stagePlan?.stageLabel || data.stagePlan?.stage || "Stage"} – Recommended Campaigns
              </div>
            </CardHeader>
            <CardContent className="space-y-6">
              {(data.stagePlan?.campaigns || []).map((c) => (
                <div key={c.id} className="rounded-2xl border p-4 shadow-sm">
                  <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-2">
                    <div className="font-medium">{c.description}</div>
                    <div className="text-xs text-muted-foreground">
                      {c.timeframe?.startDate} → {c.timeframe?.endDate}
                    </div>
                  </div>

                  {/* Personas row */}
                  <div className="mt-2 flex flex-wrap gap-2">
                    {(c.personas || []).map((p) => (
                      <span key={p} className="inline-flex items-center rounded-full border px-3 py-1 text-sm">{p}</span>
                    ))}
                  </div>

                  {/* Arsenal table */}
                  <div className="mt-4 overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="text-left text-muted-foreground">
                          <th className="py-2 pr-4">Asset</th>
                          <th className="py-2 pr-4">Channel</th>
                          <th className="py-2 pr-4">Fitment</th>
                          <th className="py-2 pr-4">Engagement</th>
                          <th className="py-2 pr-4">Expected Lift</th>
                          <th className="py-2 pr-4">Concerns</th>
                          <th className="py-2 pr-4">Why / Breadth</th>
                        </tr>
                      </thead>
                      <tbody>
                        {(c.arsenalTable || []).map((r, idx) => (
                          <tr key={`${c.id}-${idx}`} className="border-t">
                            <td className="py-3 pr-4 align-top">
                              <div className="font-medium">{r.asset?.name || "—"}</div>
                              <div className="text-xs text-muted-foreground">{r.asset?.format}</div>
                            </td>
                            <td className="py-3 pr-4 align-top">
                              <div>{r.channel?.name || "—"}</div>
                              <div className="text-xs text-muted-foreground">{r.channel?.type}</div>
                            </td>
                            <td className="py-3 pr-4 align-top capitalize">{r.fitment || "—"}</td>
                            <td className="py-3 pr-4 align-top">{r.engagement || "—"}</td>
                            <td className="py-3 pr-4 align-top">{r.expectedLift || "—"}</td>
                            <td className="py-3 pr-4 align-top">
                              <div className="flex flex-wrap gap-2 max-w-xl">
                                {(r.concernsAddressed || []).map((c) => (
                                  <span key={c} className="inline-flex items-center rounded-full bg-muted px-2 py-1 text-xs whitespace-nowrap">
                                    {c}
                                  </span>
                                ))}
                              </div>
                            </td>
                            <td className="py-3 pr-4 align-top text-xs text-muted-foreground">
                              <div>{r.why || ""}</div>
                              {typeof r.breadthScore === "number" && (
                                <div className="mt-1">Breadth: {(r.breadthScore * 100).toFixed(0)}%</div>
                              )}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              ))}

              {(!data.stagePlan?.campaigns || data.stagePlan.campaigns.length === 0) && (
                <div className="text-sm text-muted-foreground">No campaigns returned for this scenario.</div>
              )}
            </CardContent>
          </Card>

          {/* Raw tabs (optional debugging) */}
          <div>
            <div className="flex gap-2 mb-2">
              <div className="px-3 py-1 rounded bg-muted">RCS Report (raw)</div>
              <div className="px-3 py-1 rounded">Frozen Strategy</div>
              <div className="px-3 py-1 rounded">Stage Plan (raw)</div>
            </div>

            <pre className="mt-3 max-h-[520px] overflow-auto rounded-lg bg-muted p-4 text-xs">
              {JSON.stringify(data.rcsReport, null, 2)}
            </pre>

            <pre className="mt-3 max-h-[520px] overflow-auto rounded-lg bg-muted p-4 text-xs">
              {JSON.stringify(data.frozenStrategy, null, 2)}
            </pre>

            <pre className="mt-3 max-h-[520px] overflow-auto rounded-lg bg-muted p-4 text-xs">
              {JSON.stringify(data.stagePlan, null, 2)}
            </pre>
          </div>
        </div>
      )}
    </div>
  );
}
