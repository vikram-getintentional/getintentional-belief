// ============================
// Component: RCSConfigurator.tsx (replaces archetype picker)
// ============================
import React, { useEffect } from "react";
import {
  Tabs, Tab, Box, Typography, Select, MenuItem, FormControl, InputLabel,
  Autocomplete, TextField, Chip
} from "@mui/material";

import RCSNetworkGraph from "./RCSGraph";
import RCSPersonaInvolvmentVisual from "./RCSPersonaInvolvmentVisual";
import RCSConcernsRecommendations from "./RCSConcernsRecommendations";
import RCSCampaignSequences from "./RCSCampaignSequences";
import RCSOverview from "./RCSOverview";
import { buildAssetsByPersona, buildLabelMaps } from "../utils/label_utils";

type NodeOption = { id: string; type: string; label: string };
type Family = "industry" | "revenue" | "employees" | "funding" | "geography";

type ICPOption = {
  id: string;       // attribute node id
  label: string;    // human label
  family: Family;
};

export default function RCSConfigurator({
  selectedProductId,
  token,
}: {
  selectedProductId: string;
  token: string;
}) {
  const [tab, setTab] = React.useState(0);
  const [loading, setLoading] = React.useState(false);
  const [err, setErr] = React.useState<string | null>(null);

  // ICP selectors
  const [icpOptions, setIcpOptions] = React.useState<Record<Family, ICPOption[]>>({
    industry: [], revenue: [], employees: [], funding: [], geography: []
  });
  const [icpSelected, setIcpSelected] = React.useState<Record<Family, string | "">>({
    industry: "", revenue: "", employees: "", funding: "", geography: ""
  });

  // ZMOT suggestions for selected attributes
  const [zmotSuggestions, setZmotSuggestions] = React.useState<NodeOption[]>([]);

  // Graph + RCS
  const [rcs, setRcs] = React.useState<any>(null);
  const [rcsGraph, setRcsGraph] = React.useState<any>(null);

  // User-engaged nodes
  const [nodeOptions, setNodeOptions] = React.useState<NodeOption[]>([]);
  const [selectedNodes, setSelectedNodes] = React.useState<NodeOption[]>([]);

  // 1) Load available ICP attribute options
  useEffect(() => {
    if (!selectedProductId) return;
    let alive = true;
    (async () => {
      setLoading(true); setErr(null);
      try {
        const res = await fetch(`http://localhost:8000/get-icp-archetype-options/${selectedProductId}`, {
          headers: { Authorization: `Bearer ${token}` }
        });
        if (!res.ok) throw new Error(`Failed to load ICP attributes (${res.status})`);
        const data = await res.json();
        console.log("ICP attributes:", data);

        // Map backend keys to frontend Family keys
        const keyMap: Record<string, Family> = {
          industry: "industry",
          revenue_range: "revenue",
          employee_range: "employees",
          funding_stage: "funding",
          geography: "geography",
        };

        const icpOptions: Record<Family, ICPOption[]> = {
          industry: [],
          revenue: [],
          employees: [],
          funding: [],
          geography: [],
        };

        Object.entries(data).forEach(([backendKey, arr]) => {
          const fam = keyMap[backendKey];
          if (fam && Array.isArray(arr)) {
            icpOptions[fam] = arr.map((opt: any) => ({
              id: opt.id,
              label: opt.label,
              family: fam,
            }));
          }
        });

        setIcpOptions(icpOptions);

        if (!alive) return;
          setIcpOptions(icpOptions);
      } catch (e: any) {
        if (alive) setErr(e?.message || "Unable to load ICP attributes");
      } finally {
        if (alive) setLoading(false);
      }
    })();
    return () => { alive = false; };
  }, [selectedProductId, token]);

  // 2) When ICP picks change, fetch ZMOT suggestions
  useEffect(() => {
    const attrIds = Object.values(icpSelected).filter(Boolean) as string[];
    if (!selectedProductId) return;

    let alive = true;
    (async () => {
      if (attrIds.length === 0) {
        setZmotSuggestions([]);
        return;
      }
      try {
        const res = await fetch(`http://localhost:8000/get-zmots-for-attributes/${selectedProductId}`, {
          method: "POST",
          headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
          body: JSON.stringify({ attribute_ids: attrIds })
        });
        if (!res.ok) throw new Error("Failed to load ZMOT suggestions");
        const data = await res.json();
        // expect data like { zmots: [...], moments: [...], keywords: [...] }
        const zmots = (data.zmots || []).map((z: any) => ({
          id: z.zmot_event_id || z.id,
          type: "zmot_event",
          label: z.zmot_event || z.label || z.id,
        }));
        if (!alive) return;
        setZmotSuggestions(zmots);
      } catch {
        if (alive) setZmotSuggestions([]);
      }
    })();
    return () => { alive = false; };
  }, [icpSelected, selectedProductId, token]);

  // 3) Fetch RCS whenever attributes or engaged nodes change
  useEffect(() => {
    if (!selectedProductId) return;
    let alive = true;
    (async () => {
      setLoading(true); setErr(null);
      try {
        const attribute_ids = Object.values(icpSelected).filter(Boolean) as string[];
        const payload = {
          attribute_ids,
          selected_node_ids: selectedNodes.map(n => n.id)
        };
        console.log("RCS payload:", payload);
        const res = await fetch(`http://localhost:8000/get-reverse-case-study/${selectedProductId}`, {
          method: "POST",
          headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
          body: JSON.stringify(payload)
        });
        if (!res.ok) throw new Error(`Failed to get reverse case study (${res.status})`);
        const data = await res.json();
        if (!alive) return;
        setRcs(data.output);
        setRcsGraph(data.graph);
      } catch (e: any) {
        if (alive) { setErr(e?.message || "Failed to load reverse case study"); setRcs(null); }
      } finally {
        if (alive) setLoading(false);
      }
    })();
    return () => { alive = false; };
  }, [icpSelected, selectedNodes, selectedProductId, token]);

  // 4) Build selectable graph node options (personas/pains/jobs) + add ZMOT suggestions
  useEffect(() => {
    if (!rcsGraph || !rcsGraph.nodes) {
      setNodeOptions([]);
      return;
    }
    const allowed = new Set(["persona", "pain", "job"]);
    const graphNodes = (rcsGraph.nodes || [])
      .filter((n: any) => allowed.has((n.node_type || n.type || "").toLowerCase()))
      .map((n: any) => ({
        id: n.id,
        type: (n.node_type || n.type || "").toLowerCase(),
        label: n.label || n.name || n.id,
      }));
    setNodeOptions([...graphNodes, ...zmotSuggestions]);
  }, [rcsGraph, zmotSuggestions]);

  const familyPicker = (fam: Family, label: string) => (
    <FormControl sx={{ minWidth: 220 }}>
      <InputLabel id={`${fam}-label`}>{label}</InputLabel>
      <Select
        labelId={`${fam}-label`}
        value={icpSelected[fam]}
        label={label}
        onChange={e => setIcpSelected(s => ({ ...s, [fam]: e.target.value as string }))}
      >
        <MenuItem value=""><em>None</em></MenuItem>
        {icpOptions[fam].map(opt => (
          <MenuItem key={opt.id} value={opt.id}>{opt.label}</MenuItem>
        ))}
      </Select>
    </FormControl>
  );

  const engagedPicker = (
    <Autocomplete
      multiple
      options={nodeOptions.filter(opt => !selectedNodes.some(sel => sel.id === opt.id))}
      getOptionLabel={option => `${option.type}: ${option.label}`}
      value={selectedNodes}
      onChange={(_, newValue) => setSelectedNodes(newValue)}
      renderInput={params => (
        <TextField {...params} label="Engaged nodes (personas/pains/jobs/zmots)" variant="outlined" />
      )}
      fullWidth
      disablePortal
      clearOnBlur
    />
  );

  function TabPanel(props: { children?: React.ReactNode; value: number; index: number }) {
    const { children, value, index, ...other } = props;
    return (
      <div role="tabpanel" hidden={value !== index} {...other}>
        {value === index && <Box sx={{ p: 2 }}>{children}</Box>}
      </div>
    );
  }

  return (
    <Box sx={{ height: "100%", width: "100%", maxWidth: "1200px", mx: "auto", p: 3 }}>
      <Typography variant="h5" sx={{ mb: 2 }}>
        Reverse Case Studies (Attributes + Engaged Nodes)
      </Typography>

      <Box sx={{ display: "flex", gap: 2, flexWrap: "wrap", mb: 2 }}>
        {familyPicker("industry", "Industry")}
        {familyPicker("revenue", "Revenue Range")}
        {familyPicker("employees", "Employee Range")}
        {familyPicker("funding", "Funding Stage")}
        {familyPicker("geography", "Geography")}
      </Box>

      <Box sx={{ mb: 2 }}>
        {engagedPicker}
        <Box sx={{ display: "flex", flexWrap: "wrap", gap: 1, mt: 1 }}>
          {selectedNodes.map(node => (
            <Chip
              key={node.id}
              label={`${node.type}: ${node.label}`}
              onDelete={() => setSelectedNodes(selectedNodes.filter(n => n.id !== node.id))}
            />
          ))}
        </Box>
      </Box>

      <Tabs value={tab} onChange={(_, v) => setTab(v)} sx={{ mb: 2 }}>
        <Tab label="Graph" />
        <Tab label="Overview" />
        <Tab label="Campaign Sequences" />
        <Tab label="Persona Involvement Matrix" />
        <Tab label="Recommended Plays" />
      </Tabs>

      <TabPanel value={tab} index={0}>
        {rcsGraph ? (
          <div style={{ width: "100%", position: "relative", overflow: "hidden", padding: 5 }}>
            <RCSNetworkGraph graphData={rcsGraph || { nodes: [], links: [] }} />
          </div>
        ) : (
          <Typography variant="body1" color="textSecondary">
            {loading ? "Loading graph..." : "Pick attributes and/or engaged nodes to generate the scaffold."}
          </Typography>
        )}
      </TabPanel>

      <TabPanel value={tab} index={1}>
        {rcs ? <RCSOverview rcs={rcs} /> : (
          <Typography color="text.secondary">No conversion data available.</Typography>
        )}
      </TabPanel>

      <TabPanel value={tab} index={2}>
        {rcs ? (
          (() => {
            const sequences = rcs.concern_sequences || rcs.sequences_and_campaigns || [];
            const assetsByPersona = buildAssetsByPersona(rcs);
            const labelMap = buildLabelMaps(rcs);
            return (
              <RCSCampaignSequences
                sequences={sequences}
                labelMap={labelMap}
                assetsByPersona={assetsByPersona}
              />
            );
          })()
        ) : (
          <Typography color="text.secondary">No conversion data available.</Typography>
        )}
      </TabPanel>

      <TabPanel value={tab} index={3}>
        {rcs?.all_personas?.length ? (
          (() => {
            const personas = rcs.all_personas.map((p: any) => ({
              label: p.label || p.id,
              involvement: p.involvement,
              activation: p.activation,
            }));
            return <RCSPersonaInvolvmentVisual personas={personas} />;
          })()
        ) : (
          <Typography color="text.secondary">No persona involvement data available.</Typography>
        )}
      </TabPanel>

      <TabPanel value={tab} index={4}>
        <Typography variant="h6" sx={{ mb: 2 }}>Top N Concerns</Typography>
        {rcs?.prioritized_10_concerns?.length ? (
          <RCSConcernsRecommendations concerns={rcs.prioritized_10_concerns} />
        ) : (
          <Typography color="text.secondary">No concerns data available.</Typography>
        )}
      </TabPanel>

      {err && (
        <Box sx={{ mt: 2, bgcolor: "error.light", color: "error.dark", p: 2, borderRadius: 2 }}>
          {err}
        </Box>
      )}
    </Box>
  );
}
