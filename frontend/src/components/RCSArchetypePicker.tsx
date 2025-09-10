import React, { useEffect, useRef } from "react";
import ForceGraph2D from "react-force-graph-2d";
import * as d3 from "d3";



type Archetype = {
  archetype_id: string;
  label?: string;
  industry?: string;
  revenue?: string;
  employees?: string;
  funding_stage?: string;
  geography?: string;
};

type SuggestedEvent = {
  event_id: string;
  trigger_event: string;
  fit: number;
};

type RCSScaffold = {
  rcs_id: string;
  stage: "archetype" | "event" | "engagement";
  archetype: Archetype;
  insights: {
    pains: { pain_id: string; text: string }[];
    jobs: { job_id: string; text: string }[];
    personas: { title: string; department: string; seniority: string }[];
    prevalent_triggers: {
      pain_trigger_id: string;
      attribute: string;
      prevalence: number;
    }[];
  };
  event: {
    suggested_events: SuggestedEvent[];
    suggested_engagements: { engagement_id: string; name: string; fit: number }[];
  };
  engagement: any;
  plan: { sequence: any[] };
  metrics: { leading: any[]; lagging: any[] };
  notes: string[];
};

function clsx(...xs: (string | false | null | undefined)[]) {
  return xs.filter(Boolean).join(" ");
}
type GraphNode = Omit<d3.SimulationNodeDatum, 'fx' | 'fy'> & {
  fx?: number;
  fy?: number;
  node_type?: string;
  label?: string;
  id?: string;
  type?: string;
  temporal_depth?: number;
  candidate_score?: number;
};

function MyNetworkGraph({ graphData }: { graphData: { nodes: GraphNode[]; links: any[] } }) {
  const fgRef = useRef<any>(null);
  const [highlightedNodes, setHighlightedNodes] = React.useState<string[]>([]);
  const [highlightedLinks, setHighlightedLinks] = React.useState<string[]>([]);

  useEffect(() => {
    if (fgRef.current) {
      // Map temporal_depth to y position (higher depth = lower on screen)
      fgRef.current.d3Force('y', d3.forceY().strength((node: any) => 2)
        .y((node: any) => {
          // Adjust these numbers as needed for your layout
          const baseY = 0;
          const scale = -20; // pixels per depth unit
          // If you want higher depth lower, use +scale; if higher depth higher, use -scale
          return baseY + (node.temporal_depth || 0) * scale;
        })
      );
    }
  }, [graphData]);

  function getCausalChain(nodeId: string, nodes: GraphNode[], links: any[]) {
    const nodeIds = new Set<string>();
    const linkIds = new Set<string>();

    // BFS for successors
    let queue = [nodeId];
    while (queue.length) {
      const current = queue.pop()!;
      nodeIds.add(current);
      links.forEach(link => {
        if (link.source === current && !nodeIds.has(link.target)) {
          queue.push(link.target);
          linkIds.add(`${link.source}->${link.target}`);
        }
      });
    }

    // BFS for predecessors
    queue = [nodeId];
    while (queue.length) {
      const current = queue.pop()!;
      nodeIds.add(current);
      links.forEach(link => {
        if (link.target === current && !nodeIds.has(link.source)) {
          queue.push(link.source);
          linkIds.add(`${link.source}->${link.target}`);
        }
      });
    }

    return {
      nodeIds: Array.from(nodeIds),
      linkIds: Array.from(linkIds)
    };
  }

  return (
    <ForceGraph2D
      ref={fgRef}
      graphData={graphData}
      nodeAutoColorBy="type"
      linkDirectionalParticles={0.5}
      nodeLabel={node =>
        `${node.label || node.id}
        ${node.type ? `Type: ${node.type}` : ""}
        ${node.temporal_depth ? `Temporal Depth: ${node.temporal_depth}` : ""}
        ${node.cumulative_likelihood ? `Cumulative Likelihood: ${node.cumulative_likelihood}` : ""}
        ${node.candidate_score ? `Score: ${node.candidate_score}` : ""}`
      }
      linkWidth={link => link.candidate_score ? Math.max(1, link.candidate_score * 100) : 1}
      onNodeClick={node => {
        if (!node.id) return;
        const { nodeIds, linkIds } = getCausalChain(node.id, graphData.nodes, graphData.links);
        setHighlightedNodes(nodeIds);
        setHighlightedLinks(linkIds);
      }}
      nodeCanvasObject={(node, ctx, globalScale) => {
        const isHighlighted = highlightedNodes.includes(node.id ?? "");
        ctx.save();
        ctx.beginPath();
        ctx.arc(node.x ?? 0, node.y ?? 0, isHighlighted ? 12 : 8, 0, 2 * Math.PI, false);
        ctx.fillStyle = isHighlighted ? "orange" : node.color || "#888";
        ctx.fill();
        ctx.font = `${Math.max(12, 4 / globalScale)}px Sans-Serif`;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillStyle = isHighlighted ? "black" : "#333";
        ctx.restore();
      }}
      linkCanvasObject={(link, ctx) => {
        const sourceId = typeof link.source === "object" ? link.source.id : link.source;
        const targetId = typeof link.target === "object" ? link.target.id : link.target;
        const isHighlighted = highlightedLinks.includes(`${sourceId}->${targetId}`);
        const sourceNode = typeof link.source === "object" ? link.source : graphData.nodes.find(n => n.id === link.source);
        const targetNode = typeof link.target === "object" ? link.target : graphData.nodes.find(n => n.id === link.target);
        if (sourceNode && targetNode) {
          ctx.save();
          ctx.beginPath();
          ctx.moveTo(sourceNode.x ?? 0, sourceNode.y ?? 0);
          ctx.lineTo(targetNode.x ?? 0, targetNode.y ?? 0);
          ctx.strokeStyle = isHighlighted ? "orange" : "#999";
          ctx.lineWidth = isHighlighted ? 4 : 1;
          ctx.stroke();
          ctx.restore();
        }
      }}
    />
  );
}

export default function RCSArchetypePicker({
  selectedProductId,
  token,
}: {
  selectedProductId: string;
  token: string;
}) {
  const [archetypes, setArchetypes] = React.useState<Archetype[]>([]);
  const [selectedArch, setSelectedArch] = React.useState<string>("");
  const [rcs, setRcs] = React.useState<RCSScaffold | null>(null);
  const [rcsGraph, setRcsGraph] = React.useState<any>(null);
  const [loadingList, setLoadingList] = React.useState(false);
  const [loadingRCS, setLoadingRCS] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);
  const [zmotEvents, setZmotEvents] = React.useState<{ zmot_event_id: string; zmot_event: string; occurance: number }[]>([]);
  const [selectedZmotEvent, setSelectedZmotEvent] = React.useState<string>("");
  const [selectedEngagementMeta, setSelectedEngagementMeta] = React.useState<any>(null);

  // Load archetypes when product changes
  React.useEffect(() => {
    if (!selectedProductId) return;
    let alive = true;
    setLoadingList(true);
    setError(null);
    setArchetypes([]);
    setSelectedArch("");
    setZmotEvents([]);
    setSelectedZmotEvent("");
    setRcs(null);

    (async () => {
      try {
        const res = await fetch(`http://localhost:8000/get-archetypes/${selectedProductId}`, {
          headers: { Authorization: `Bearer ${token}` }
        });
        if (!res.ok) throw new Error(`Failed to load archetypes (${res.status})`);
        const data = await res.json();
        console.log("Output archetype:", data);
        if (!alive) return;
        setArchetypes(data.archetypes || []);
      } catch (e: any) {
        if (alive) setError(e?.message || "Unable to load archetypes");
      } finally {
        if (alive) setLoadingList(false);
      }
    })();
    return () => { alive = false; };
  }, [selectedProductId, token]);

  // Load ZMOT events when archetype changes
  React.useEffect(() => {
    if (!selectedArch) {
      setZmotEvents([]);
      setSelectedZmotEvent("");
      return;
    }
    let alive = true;
    setZmotEvents([]);
    setSelectedZmotEvent("");
    (async () => {
      
      try {
        const res = await fetch(
          `http://localhost:8000/get-zmots-for-archetype/${selectedProductId}?archetype_id=${selectedArch}`,
          { headers: { Authorization: `Bearer ${token}` } }
        );
        console.log("ZMOT fetch for arch:", selectedArch);
        if (!res.ok) throw new Error("Failed to load ZMOT events");
        const data = await res.json();
        console.log("Zmot response:", data);
        if (!alive) return;
        setZmotEvents(Array.isArray(data) ? data : data.zmots || []);
      } catch (err) {
        if (alive) setZmotEvents([]);
      }
    })();
    return () => { alive = false; };
  }, [selectedArch, selectedProductId, token]);

  // Fetch RCS scaffold on any relevant change
  React.useEffect(() => {
    
    setLoadingRCS(true);
    setError(null);
    (async () => {
      try {
        const payload: any = {};
        if (selectedArch) payload.archetype_id = selectedArch;
        if (selectedZmotEvent) payload.zmot_event_id = selectedZmotEvent;
        if (selectedEngagementMeta) payload.engagement_meta = selectedEngagementMeta;
        const res = await fetch(`http://localhost:8000/get-reverse-case-study/${selectedProductId}`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${token}`,
          },
          body: JSON.stringify(payload),
        });
        if (!res.ok) throw new Error(`Failed to get reverse case study (${res.status})`);
        const data = await res.json();
        console.log("RCS response:", data);
        setRcs(data.output);
        setRcsGraph(data.graph);
      } catch (e: any) {
        setError(e.message || "Failed to load reverse case study");
        setRcs(null);
      } finally {
        setLoadingRCS(false);
      }
    })();
  }, [selectedArch, selectedZmotEvent, selectedEngagementMeta, selectedProductId, token]);

  const current = archetypes.find(a => a.archetype_id === selectedArch);
  const title = current
    ? current.label ||
      [current.industry, current.funding_stage, current.geography, current.revenue, current.employees].filter(Boolean).join(" | ")
    : "";

  return (
    <div className="w-full max-w-6xl mx-auto p-6">
      <h1 className="text-2xl font-semibold mb-4">Reverse Case Studies Generator (RCS)</h1>
      {rcsGraph && (
        <div
          style={{
            height: "60vh",
            maxHeight: "60vh", 
            position: "relative",
            overflow: "hidden",
            padding: 5,
          }}
        >
          <MyNetworkGraph graphData={rcsGraph || {nodes: [], links: []}} />
        </div>
      )}
      {/* Archetype Picker */}
      <div className="bg-white border rounded-2xl p-4 shadow-sm">
        <label className="block text-sm font-medium mb-2">Select archetype</label>
        <div className="flex items-center gap-3">
          <select
            className="w-full border rounded-xl px-3 py-2"
            value={selectedArch}
            onChange={e => setSelectedArch(e.target.value)}
            disabled={loadingList}
          >
            <option value="" disabled>
              {loadingList ? "Loading…" : "Choose an archetype…"}
            </option>
            {archetypes.map(a => (
              <option key={a.archetype_id} value={a.archetype_id}>
                {[
                  a.label,
                  [a.industry, a.funding_stage, a.geography, a.revenue, a.employees].filter(Boolean).join(" | ")
                ].filter(Boolean)[0]}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* ZMOT Picker */}
      {zmotEvents.length > 0 && (
        <div className="mt-4">
          <label className="block text-sm font-medium mb-2">Select ZMOT Event</label>
          <select
            className="w-full border rounded-xl px-3 py-2"
            value={selectedZmotEvent}
            onChange={e => setSelectedZmotEvent(e.target.value)}
          >
            <option value="">No ZMOT filter</option>
            {zmotEvents.map(ev => (
              <option key={ev.zmot_event_id} value={ev.zmot_event_id}>{ev.zmot_event}</option>
            ))}
          </select>
        </div>
      )}

      {/* Errors */}
      {error && (
        <div className="mt-4 bg-red-50 text-red-700 border border-red-200 rounded-xl p-3">
          {error}
        </div>
      )}
      

      {/* RCS Scaffold */}
      {rcs && (
        <div className="mt-6 space-y-6">
            <div className="bg-white border rounded-2xl p-5 shadow-sm">
            <h2 className="text-xl font-semibold mb-2">Backend Output</h2>
            <pre className="text-xs bg-gray-50 p-4 rounded-xl overflow-x-auto">
                {JSON.stringify(rcs, null, 2)}
            </pre>
            </div>
        </div>
        )}

      {/* Empty state */}
      {!rcs && !loadingRCS && !error && (
        <div className="mt-6 text-sm text-gray-500">
          Select an archetype to generate the scaffold.
        </div>
      )}
    </div>
  );
}

function Card({ title, items }: { title: string; items: string[] }) {
  return (
    <div className="bg-white border rounded-2xl p-5 shadow-sm">
      <h3 className="font-medium mb-3">{title}</h3>
      <ul className="space-y-2">
        {items.slice(0, 6).map((t, i) => (
          <li key={i} className="text-sm">{t}</li>
        ))}
        {!items.length && <li className="text-sm text-gray-500">None</li>}
      </ul>
    </div>
  );
}

function PersonasCard({ personas }: { personas: { title: string; department: string; seniority: string }[] }) {
  return (
    <div className="bg-white border rounded-2xl p-5 shadow-sm">
      <h3 className="font-medium mb-3">Personas</h3>
      <ul className="space-y-2">
        {personas.slice(0, 6).map((p, i) => (
          <li key={i} className="text-sm">
            <span className="font-medium">{p.title || "—"}</span>
            <span className="text-gray-500"> • {p.department || "—"} • {p.seniority || "—"}</span>
          </li>
        ))}
        {!personas.length && <li className="text-sm text-gray-500">None</li>}
      </ul>
    </div>
  );
}