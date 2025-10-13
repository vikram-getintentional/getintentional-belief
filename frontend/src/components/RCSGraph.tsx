import React, { useEffect, useRef, useMemo } from "react";
import ForceGraph2D from "react-force-graph-2d";
import * as d3 from "d3";

type GraphLink = {
  id?: string;
  source: any;        // force-graph allows id or node object
  target: any;
  likelihood?: number;
  relevance?: number;
};

type GraphNode = Omit<d3.SimulationNodeDatum, "fx" | "fy"> & {
  fx?: number;
  fy?: number;
  node_type?: string;
  label?: string;
  id?: string;
  type?: string;
  temporal_depth?: number;
  candidate_score?: number;
  causal_depth?: number;
  cumulative_likelihood?: number;
};

type Graph = { nodes: GraphNode[]; links: GraphLink[] };




export default function RCSNetworkGraph({
  graphData,
}: {
  graphData?: Graph | null;
}) {
  const fgRef = useRef<any>(null);
  const [highlightedNodes, setHighlightedNodes] = React.useState<string[]>([]);
  const [highlightedLinks, setHighlightedLinks] = React.useState<string[]>([]);
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const [dims, setDims] = React.useState({ w: 0, h: 0 });

  // new: hovered node id for label-on-hover
  const [hoveredNode, setHoveredNode] = React.useState<string | null>(null);

  // Always give ForceGraph a safe object
  const safeGraph: Graph = useMemo(() => {
    if (!graphData || !Array.isArray(graphData.nodes) || !Array.isArray(graphData.links)) {
      return { nodes: [], links: [] };
    }
    return graphData;
  }, [graphData]);

  // new: only render nodes (supply empty links array)
  const graphForRender: Graph = useMemo(
    () => ({ nodes: safeGraph.nodes, links: safeGraph.links || [] }),
    [safeGraph]
  );

  useEffect(() => {
      if (!wrapRef.current) return;
      const obs = new ResizeObserver(([entry]) => {
        const cr = entry.contentRect;
        setDims({ w: Math.max(320, cr.width), h: Math.max(320, cr.height) });
      });
      obs.observe(wrapRef.current);
      return () => obs.disconnect();
    }, []);

  useEffect(() => {
    if (!fgRef.current || safeGraph.nodes.length === 0) return;

    const depths = safeGraph.nodes.map((n) => n.causal_depth ?? 0);
    const minDepth = Math.min(...depths);
    const maxDepth = Math.max(...depths);

    const canvasHeight = 400;
    const topY = 0;
    const bottomY = canvasHeight;

    fgRef.current.d3Force("collide", d3.forceCollide().radius(20).strength(0.7));
    fgRef.current.d3Force("charge", d3.forceManyBody().strength(-120));
    fgRef.current.d3Force("center", d3.forceCenter(0, canvasHeight / 2));

    fgRef.current.d3Force(
      "y",
      d3.forceY().strength(2).y((node: any) => {
        if (node.node_type === "product") return bottomY;
        if (node.node_type === "archetype") return topY;
        const depth = node.causal_depth ?? minDepth;
        if (maxDepth === minDepth) return topY;
        const fy = topY + ((depth - minDepth) / (maxDepth - minDepth)) * (bottomY - topY);
        return Math.max(topY + 1, Math.min(bottomY - 1, fy));
      })
    );

    fgRef.current.d3Force(
      "x",
      d3.forceX().strength(2).x((node: any) => {
        if (node.node_type === "product") return 0;
        const sameDepthNodes = safeGraph.nodes.filter((n) => n.causal_depth === node.causal_depth);
        const idx = sameDepthNodes.findIndex((n) => n.id === node.id);
        const total = Math.max(1, sameDepthNodes.length);
        const spread = 300;
        return (idx - total / 2) * (spread / total);
      })
    );

    // Fit after forces tick a bit
    const t = setTimeout(() => {
      try {
        fgRef.current?.zoomToFit(400, 40);
      } catch {}
    }, 500);
    return () => clearTimeout(t);
  }, [safeGraph]);

  function getCausalChain(nodeId: string, nodes: GraphNode[], links: GraphLink[]) {
    const nodeIds = new Set<string>();
    const linkIds = new Set<string>();

    // downstream
    let queue: string[] = [nodeId];
    while (queue.length) {
      const current = queue.pop()!;
      nodeIds.add(current);
      links.forEach((link) => {
        const src = typeof link.source === "object" ? link.source.id : link.source;
        const tgt = typeof link.target === "object" ? link.target.id : link.target;
        if (src === current && !nodeIds.has(tgt)) {
          queue.push(tgt);
          linkIds.add(`${src}->${tgt}`);
        }
      });
    }

    // upstream
    queue = [nodeId];
    while (queue.length) {
      const current = queue.pop()!;
      nodeIds.add(current);
      links.forEach((link) => {
        const src = typeof link.source === "object" ? link.source.id : link.source;
        const tgt = typeof link.target === "object" ? link.target.id : link.target;
        if (tgt === current && !nodeIds.has(src)) {
          queue.push(src);
          linkIds.add(`${src}->${tgt}`);
        }
      });
    }
    return { nodeIds: Array.from(nodeIds), linkIds: Array.from(linkIds) };
  }

  return (
    <div ref={wrapRef} style={{ width: "100%", height: "100%" }}>
      <ForceGraph2D
        ref={fgRef}
        width={dims.w}
        height={dims.h}
        graphData={graphForRender}               // links intentionally empty => only nodes rendered
        nodeAutoColorBy="type"
        onNodeHover={(node: any) => setHoveredNode(node ? (node.id as string) : null)} // show labels only on hover
        nodeLabel={undefined}                    // disable default tooltip/label (we draw labels manually)
        linkDirectionalParticles={() => 0}       // ensure no particle effects
        // small line normally, thicker when part of selected causal chain
        linkWidth={(link: any) => {
          const src = typeof link.source === "object" ? link.source.id : link.source;
          const tgt = typeof link.target === "object" ? link.target.id : link.target;
          const id = `${src}->${tgt}`;
          return highlightedLinks.includes(id) ? 2.0 : 0.8;
        }}
        linkColor={() => "rgba(150,150,150,0.9)"}
        linkDirectionalArrowLength={0}
        onNodeClick={(node: any) => {
          const id = node?.id as string | undefined;
          if (!id) return;
          const { nodeIds, linkIds } = getCausalChain(id, safeGraph.nodes, safeGraph.links);
          setHighlightedNodes(nodeIds);
          setHighlightedLinks(linkIds);
        }}
        nodeCanvasObject={(node: any, ctx: CanvasRenderingContext2D, globalScale: number) => {
          const isHighlighted = highlightedNodes.includes(node.id ?? "");
          const isHovered = hoveredNode === node.id;
          ctx.save();
          ctx.beginPath();
          ctx.arc(node.x ?? 0, node.y ?? 0, isHighlighted ? 12 : 8, 0, 2 * Math.PI, false);
          ctx.fillStyle = isHighlighted ? "orange" : node.color || "#888";
          ctx.fill();


          // label: only draw when hovered (or highlighted via click)
          if (isHovered || isHighlighted) {
            const label = node.label || node.id || "";
            if (label) {
              // responsive font sizing
              const fontSize = Math.max(10, 12 / Math.max(0.5, globalScale));
              ctx.font = `${fontSize}px Sans-Serif`;
              ctx.textAlign = "center";
              ctx.textBaseline = "top";
              ctx.fillStyle = isHighlighted ? "black" : "#333";
              ctx.fillText(label, node.x ?? 0, (node.y ?? 0) + 10);
            }
          }
          ctx.restore();
        }}
        backgroundColor="#ffffff"
      />
    </div>
  );
}
