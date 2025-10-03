import React, { useEffect, useRef } from "react";
import ForceGraph2D from "react-force-graph-2d";
import * as d3 from "d3";

type GraphNode = Omit<d3.SimulationNodeDatum, 'fx' | 'fy'> & {
  fx?: number;
  fy?: number;
  node_type?: string;
  label?: string;
  id?: string;
  type?: string;
  temporal_depth?: number;
  candidate_score?: number;
  causal_depth?: number;
};

export default function RCSNetworkGraph({ graphData }: { graphData: { nodes: GraphNode[]; links: any[] } }) {
  const fgRef = useRef<any>(null);
  const [highlightedNodes, setHighlightedNodes] = React.useState<string[]>([]);
  const [highlightedLinks, setHighlightedLinks] = React.useState<string[]>([]);

  useEffect(() => {
    if (fgRef.current && graphData.nodes.length > 0) {
      const depths = graphData.nodes.map(n => n.causal_depth ?? 0);
      const minDepth = Math.min(...depths);
      const maxDepth = Math.max(...depths);

      const canvasHeight = 400;
      const topY = 0;
      const bottomY = canvasHeight;
      fgRef.current.d3Force('collide', d3.forceCollide().radius(20).strength(0.7));
      fgRef.current.d3Force('charge', d3.forceManyBody().strength(-120));
      fgRef.current.d3Force('center', d3.forceCenter(0, canvasHeight / 2));

      fgRef.current.d3Force('y', d3.forceY().strength(2).y((node: any) => {
        if (node.node_type === "product") {
          return bottomY;
        } else if (node.node_type === "archetype") {
          return topY;
        }
        const depth = node.causal_depth ?? minDepth;
        if (maxDepth === minDepth) return topY;
        const fy = (topY + ((depth - minDepth) / (maxDepth - minDepth)) * (bottomY - topY));
        return Math.max(topY + 1, Math.min(bottomY - 1, fy));
      }));

      fgRef.current.d3Force('x', d3.forceX().strength(2).x((node: any) => {
        if (node.node_type === "product") return 0;
        const sameDepthNodes = graphData.nodes.filter(n => n.causal_depth === node.causal_depth);
        const idx = sameDepthNodes.findIndex(n => n.id === node.id);
        const total = sameDepthNodes.length;
        const spread = 300;
        return ((idx - total / 2) * (spread / Math.max(1, total)));
      }));
      // Center and fit the graph after layout
    setTimeout(() => {
      if (fgRef.current) {
        fgRef.current.zoomToFit(400, 40); // Only call if ref is valid
      }
    }, 500); // Delay to allow layout to stabilize
    
    }
    
  }, [graphData]);

  function getCausalChain(nodeId: string, nodes: GraphNode[], links: any[]) {
    const nodeIds = new Set<string>();
    const linkIds = new Set<string>();
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
      linkDirectionalParticles={link => (link.likelihood * link.relevance) || 0}
      nodeLabel={node =>
        `${node.label || node.id}
        ${node.type ? `Type: ${node.type}` : ""}
        ${node.causal_depth ? `Causal Depth: ${node.causal_depth}` : ""}
        ${node.cumulative_likelihood ? `Cumulative Likelihood: ${node.cumulative_likelihood}` : ""}
        ${node.candidate_score ? `Score: ${node.candidate_score}` : ""}`
      }
      linkWidth={link => link.likelihood ? Math.max(1, link.likelihood * 100) : 1}
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