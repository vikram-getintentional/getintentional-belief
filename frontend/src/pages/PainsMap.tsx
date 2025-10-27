// PainsMap.tsx — full working version
// Adds: Edge "Evidence (optional)" field, Persona "LinkedIn Profiles" (URL + Bio)
// Bulk-save menu (edited vs entire), unified add/edit modal, all columns & trail

import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  Box, Typography, Select, MenuItem, FormControl, InputLabel,
  LinearProgress, Toolbar, Card, CardContent, CardActions, Button,
  IconButton, Collapse, Dialog, DialogTitle, DialogContent, DialogActions,
  List, ListItem, ListItemText, Divider, Chip, Stack, TextField, Menu
} from "@mui/material";
import { useTheme } from "@mui/material/styles";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import AddIcon from "@mui/icons-material/Add";
import EditIcon from "@mui/icons-material/Edit";
import ChevronRightIcon from "@mui/icons-material/ChevronRight";
import SaveIcon from "@mui/icons-material/Save";
import DeleteIcon from "@mui/icons-material/Delete";

/* ===== Types ===== */
type RawNode = {
  id: string;
  type?: string;
  title?: string;
  label?: string;
  content?: string;
  relevance?: number;
  likelihood?: number;
  properties?: Record<string, any>;
  sources?: Record<string, any>;
  raw?: any;
};

type RawEdge = {
  source: string;
  target: string;
  relation?: string | null;
  weight?: number | null;
  relevance?: number | null;
  likelihood?: number | null;
  boost?: number | null;
  raw?: any;
};

type GraphPayload = {
  nodes: RawNode[];
  edges: RawEdge[];
  nodesMap: Map<string, RawNode>;
  outgoing: Map<string, RawEdge[]>;
  incoming: Map<string, RawEdge[]>;
};

type TrailSeg = {
  painId: string;
  selectedTriggerId?: string | null;
  selectedJobId?: string | null;
  selectedZmotId?: string | null;
};

type EditCtx = {
  mode: "add" | "edit";
  node: RawNode;
  edge?: RawEdge;
  fromId?: string;
  toId?: string;
  addFromType?: string;
  addTargetType?: string;
};

/* ===== Helpers ===== */
const randId = (prefix: string) =>
  `${prefix}:${Math.random().toString(36).slice(2, 10)}${Math.random().toString(36).slice(2, 6)}`;

const DIMENSION_OPTIONS = ["industry","employee_range","revenue_range","funding_Stage","geography"] as const;

const TARGETS_BY_SOURCE: Record<string, Array<{type: string; defaultRelation: string}>> = {
  capability: [{ type: "pain", defaultRelation: "solve" }],
  pain: [
    { type: "pain_trigger", defaultRelation: "triggered_by" },
    { type: "perceived_metric", defaultRelation: "expressed_as" },
    { type: "job", defaultRelation: "felt_in" },
  ],
  pain_trigger: [
    { type: "attribute_value", defaultRelation: "prevalent_in" },
    { type: "zmot_event", defaultRelation: "leads_to_zmot" },
  ],
  attribute_value: [
    { type: "zmot_event", defaultRelation: "associated_zmot" },
  ],
  job: [
    { type: "persona", defaultRelation: "performed_by" },
    { type: "pain", defaultRelation: "solve" },
  ],
  zmot_event: [
    { type: "observable_moment", defaultRelation: "observed_as" },
    { type: "keyword", defaultRelation: "keyword" },
  ],
};

const REL_EDGE_APPLIES = new Set([
  "product-capability",
  "capability-pain",
  "pain-job",
  "job-persona",
  "job-pain",
  "pain-perceived_metric",
  "pain-pain_trigger",
  "pain_trigger-attribute_value",
  "attribute_value-zmot_event",
  "zmot_event-observable_moment",
  "zmot_event-keyword",
]);

const BOOST_APPLIES = (fromType?: string, toType?: string) =>
  (fromType?.toLowerCase() === "pain_trigger" && toType?.toLowerCase() === "zmot_event");

const edgeKey = (e: RawEdge) => `${String(e.source)}->${String(e.target)}`;

/* =========================================
   Component
========================================= */
export default function PainsMap() {
  const theme = useTheme();
  const token = localStorage.getItem("token") ?? "";

  const COLUMN_WIDTH = 360;
  const outerColsRef = useRef<HTMLDivElement | null>(null);
  const lastColumnRef = useRef<HTMLDivElement | null>(null);

  const [products, setProducts] = useState<{ id: string; name: string }[]>([]);
  const [selectedProductId, setSelectedProductId] = useState<string>("");
  const [graph, setGraph] = useState<GraphPayload | null>(null);
  const [loading, setLoading] = useState(false);
  const [statusMsg, setStatusMsg] = useState("");
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});
  const [detailNode, setDetailNode] = useState<RawNode | null>(null);
  const [detailOpen, setDetailOpen] = useState(false);

  /* base path selections */
  const [selectedCapability, setSelectedCapability] = useState<string | null>(null);
  const [selectedPain, setSelectedPain] = useState<string | null>(null);
  const [selectedTrigger, setSelectedTrigger] = useState<string | null>(null);
  const [selectedJob, setSelectedJob] = useState<string | null>(null);

  /* trail */
  const [trail, setTrail] = useState<TrailSeg[]>([]);

  /* unified editor */
  const [editCtx, setEditCtx] = useState<EditCtx | null>(null);

  /* pending changes for "Save edited changes" */
  const [pendingNodes, setPendingNodes] = useState<Map<string, RawNode>>(new Map());
  const [pendingEdges, setPendingEdges] = useState<Map<string, RawEdge>>(new Map());

  /* Save button menu */
  const [saveAnchor, setSaveAnchor] = useState<null | HTMLElement>(null);
  const saveMenuOpen = Boolean(saveAnchor);

  /* ===== bootstrap ===== */
  useEffect(() => {
    (async () => {
      try {
        const me = await fetch("http://localhost:8000/me", {
          headers: { Authorization: `Bearer ${token}` },
        }).then((r) => r.json());

        const prod = await fetch(
          `http://localhost:8000/get-products/${me.company_id}`,
          { headers: { Authorization: `Bearer ${token}` } }
        ).then((r) => r.json());

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

  /* ===== load graph ===== */
  useEffect(() => {
    if (!selectedProductId) return;
    (async () => {
      try {
        setLoading(true);
        setStatusMsg("");
        const res = await fetch(
          `http://localhost:8000/export-product-graph/${selectedProductId}`,
          { headers: { Authorization: `Bearer ${token}` } }
        );
        const payload = await res.json();

        const nodes: RawNode[] = Array.isArray(payload.nodes) ? payload.nodes : [];
        const edges: RawEdge[] = Array.isArray(payload.edges) ? payload.edges : [];

        const nodesMap = new Map<string, RawNode>();
        nodes.forEach((n) => nodesMap.set(String(n.id), n));

        const outgoing = new Map<string, RawEdge[]>();
        const incoming = new Map<string, RawEdge[]>();
        edges.forEach((e) => {
          const s = String(e.source);
          const t = String(e.target);
          if (!outgoing.has(s)) outgoing.set(s, []);
          outgoing.get(s)!.push(e);
          if (!incoming.has(t)) incoming.set(t, []);
          incoming.get(t)!.push(e);
        });

        setGraph({ nodes, edges, nodesMap, outgoing, incoming });

        /* reset */
        setSelectedCapability(null);
        setSelectedPain(null);
        setSelectedTrigger(null);
        setSelectedJob(null);
        setTrail([]);
        setExpanded({});
        setPendingNodes(new Map());
        setPendingEdges(new Map());
      } catch {
        setStatusMsg("Failed to load graph export");
      } finally {
        setLoading(false);
      }
    })();
  }, [selectedProductId, token]);

  /* ===== helpers ===== */
  const normalizeNode = (n: RawNode): RawNode => {
    const idStr = String(n.id);
    const idPrefix = idStr.includes(":") ? idStr.split(":")[0] : idStr;
    return {
      ...n,
      id: idStr,
      type: n.type ?? n.raw?.type ?? n.raw?.node_type ?? idPrefix,
      label: n.label ?? n.raw?.label ?? n.title ?? idStr,
      title: n.title ?? n.raw?.title ?? n.raw?.label ?? idStr,
      content: n.content ?? n.raw?.content ?? n.raw?.description ?? "",
      raw: n.raw ?? {},
    };
  };

  const displayLabel = (node: RawNode | undefined) =>
    node ? (normalizeNode(node).label || normalizeNode(node).title || normalizeNode(node).id) : "";

  const getNode = (id?: string | null) =>
    id && graph ? normalizeNode(graph.nodesMap.get(String(id)) || { id }) : undefined;

  const getOutgoing = (nodeId?: string) => {
    if (!graph || !nodeId) return [] as { edge: RawEdge; node: RawNode }[];
    const es = graph.outgoing.get(nodeId) || [];
    return es
      .map((e) => ({ edge: e, node: graph.nodesMap.get(String(e.target))! }))
      .filter((p) => p.node);
  };

  const getIncoming = (nodeId?: string) => {
    if (!graph || !nodeId) return [] as { edge: RawEdge; node: RawNode }[];
    const es = graph.incoming.get(nodeId) || [];
    return es
      .map((e) => ({ edge: e, node: graph.nodesMap.get(String(e.source))! }))
      .filter((p) => p.node);
  };

  const isTrigger = (node?: RawNode, edge?: RawEdge) => {
    const t = (node?.type ?? node?.raw?.type ?? node?.raw?.node_type ?? "").toString().toLowerCase();
    const r = (edge?.relation ?? "").toString().toLowerCase();
    const i = (node?.id ?? "").toString().toLowerCase();
    return t.includes("trigger") || r.includes("trigger") || i.includes("trigger");
  };

  /* ===== base column data ===== */
  const capabilities = useMemo(() => {
    if (!graph || !selectedProductId) return [] as RawNode[];
    const prodOut = getOutgoing(selectedProductId);
    const capsFromProduct = prodOut
      .filter((p) =>
        (p.node?.type ?? "").toLowerCase().includes("capability") ||
        (p.edge.relation && String(p.edge.relation).toLowerCase().includes("cap"))
      )
      .map((p) => ({ ...normalizeNode(p.node), edgeToProduct: p.edge }));
    if (capsFromProduct.length) return capsFromProduct;
    return graph.nodes.filter((n) => (n.type ?? "").toLowerCase().includes("capability")).map(normalizeNode);
  }, [graph, selectedProductId]);

  const painsForCapability = useMemo(() => {
    if (!graph || !selectedCapability) return [] as { node: RawNode; edge: RawEdge }[];
    return getOutgoing(selectedCapability)
      .filter((p) =>
        (p.node?.type ?? "").toLowerCase().includes("pain") ||
        (p.edge.relation && String(p.edge.relation).toLowerCase().includes("solve"))
      )
      .map((p) => ({ node: normalizeNode(p.node), edge: p.edge }));
  }, [graph, selectedCapability]);

  const triggersForPainId = (painId: string) =>
    getOutgoing(painId)
      .filter(({ node, edge }) => isTrigger(node, edge))
      .map(({ node, edge }) => ({ node: normalizeNode(node), edge }));

  const metricsForPainId = (painId: string) =>
    getOutgoing(painId)
      .filter((p) =>
        (p.node?.type ?? "").toLowerCase().includes("metric") ||
        (p.edge.relation && String(p.edge.relation).toLowerCase().includes("metric"))
      )
      .map((p) => ({ node: normalizeNode(p.node), edge: p.edge }));

  const jobsForPainId = (painId: string) =>
    getOutgoing(painId)
      .filter((p) =>
        (p.node?.type ?? "").toLowerCase().includes("job") ||
        (p.edge.relation && String(p.edge.relation).toLowerCase().includes("felt"))
      )
      .map((p) => ({ node: normalizeNode(p.node), edge: p.edge }));

  const personasForJobId = (jobId: string) =>
    getOutgoing(jobId)
      .filter((p) => (p.node?.type ?? "").toLowerCase().includes("persona"))
      .map((p) => ({ node: normalizeNode(p.node), edge: p.edge }));

  const solvesForJobId = (jobId: string) =>
    getOutgoing(jobId)
      .filter((p) =>
        (p.node?.type ?? "").toLowerCase().includes("pain") ||
        (p.edge.relation && String(p.edge.relation).toLowerCase().includes("solve"))
      )
      .map((p) => ({ node: normalizeNode(p.node), edge: p.edge }));

  const attributesForTriggerId = (triggerId: string) =>
    getOutgoing(triggerId)
      .filter((p) => (p.node?.type ?? "").toLowerCase().includes("attribute"))
      .map((p) => ({ node: normalizeNode(p.node), edge: p.edge }));

  const zmotForTriggerId = (triggerId: string) =>
    getOutgoing(triggerId)
      .filter((p) => (p.node?.type ?? "").toLowerCase().includes("zmot"))
      .map((p) => ({ node: normalizeNode(p.node), edge: p.edge }));

  const observableForZmotId = (zid: string) =>
    getOutgoing(zid)
      .filter((p) => (p.node?.type ?? "").toLowerCase().includes("observable"))
      .map((p) => ({ node: normalizeNode(p.node), edge: p.edge }));

  const keywordsForZmotId = (zid: string) =>
    getOutgoing(zid)
      .filter((p) => (p.node?.type ?? "").toLowerCase().includes("keyword"))
      .map((p) => ({ node: normalizeNode(p.node), edge: p.edge }));

  /* ===== styles ===== */
  const cardSx = (isSelected: boolean, columnHasSelection: boolean) => {
    if (isSelected) {
      return {
        backgroundColor: theme.palette.background.paper,
        border: `2px solid ${theme.palette.primary.main}`,
        boxShadow: theme.shadows[4],
        transform: "translateY(-1px)",
      };
    }
    if (columnHasSelection) {
      return { opacity: 0.7, filter: "grayscale(0.2)" };
    }
    return { backgroundColor: theme.palette.action.hover, boxShadow: theme.shadows[1] };
  };

  /* ===== auto scroll ===== */
  useEffect(() => {
    const el = lastColumnRef.current;
    if (!el) return;
    requestAnimationFrame(() => {
      try { el.scrollIntoView({ behavior: "smooth", inline: "end", block: "nearest" }); }
      catch { const outer = outerColsRef.current; if (outer) outer.scrollLeft = el.offsetLeft; }
    });
  }, [selectedCapability, selectedPain, selectedTrigger, selectedJob, trail]);

  /* ===== upserts (local + mark pending) ===== */
  const markPendingNode = (n: RawNode) =>
    setPendingNodes((prev) => {
      const next = new Map(prev);
      next.set(String(n.id), normalizeNode(n));
      return next;
    });

  const markPendingEdge = (e: RawEdge) =>
    setPendingEdges((prev) => {
      const next = new Map(prev);
      next.set(edgeKey(e), { ...e });
      return next;
    });

  const upsertNodeLocal = (node: RawNode) => {
    setGraph((g) => {
      if (!g) return g;
      const n = normalizeNode(node);
      const nodes = [...g.nodes];
      const idx = nodes.findIndex((x) => String(x.id) === n.id);
      if (idx >= 0) nodes[idx] = { ...nodes[idx], ...n };
      else nodes.push(n);
      const nodesMap = new Map(g.nodesMap);
      nodesMap.set(n.id, n);
      return { ...g, nodes, nodesMap };
    });
    markPendingNode(node);
  };

  const upsertEdgeLocal = (edge: RawEdge) => {
    setGraph((g) => {
      if (!g) return g;
      const edges = [...g.edges];
      const i = edges.findIndex(
        (e) => String(e.source) === String(edge.source) && String(e.target) === String(edge.target)
      );
      if (i >= 0) edges[i] = { ...edges[i], ...edge };
      else edges.push(edge);

      const outgoing = new Map(g.outgoing);
      const incoming = new Map(g.incoming);
      const s = String(edge.source);
      const t = String(edge.target);
      outgoing.set(s, (outgoing.get(s) || []).filter((e) => !(String(e.target) === t)));
      outgoing.get(s)!.push(edge);
      incoming.set(t, (incoming.get(t) || []).filter((e) => !(String(e.source) === s)));
      incoming.get(t)!.push(edge);

      return { ...g, edges, outgoing, incoming };
    });
    markPendingEdge(edge);
  };

  /* ===== Save (unified) ===== */
  const saveEditCtx = async () => {
    if (!editCtx) return;

    const t = (editCtx.node.type || editCtx.addTargetType || "").toLowerCase();
    const n = { ...editCtx.node };

    const sync = (labelFrom?: string, titleFrom?: string, contentFrom?: string) => {
      if (labelFrom) n.label = labelFrom;
      if (titleFrom) n.title = titleFrom;
      if (contentFrom !== undefined) n.content = contentFrom;
    };

    if (t.includes("capability")) sync(n.raw?.name, n.raw?.name, n.raw?.description);
    else if (t === "pain") sync(n.label, n.title, n.raw?.description);
    else if (t === "perceived_metric") sync(n.raw?.metric, n.raw?.metric);
    else if (t === "job") sync(n.label, n.title, n.raw?.description);
    else if (t === "persona") sync(n.raw?.title, n.raw?.title);
    else if (t === "pain_trigger") sync(n.raw?.attribute, n.raw?.attribute);
    else if (t === "attribute_value") sync(n.raw?.name, n.raw?.name);
    else if (t === "zmot_event") sync(n.raw?.event, n.raw?.event);
    else if (t === "observable_moment" || t === "keyword") sync(n.raw?.text, n.raw?.text);

    if (editCtx.mode === "edit") {
      upsertNodeLocal(n);
      if (editCtx.edge) upsertEdgeLocal({ ...editCtx.edge });
    } else {
      const newId = n.id || randId(editCtx.addTargetType || t || "node");
      n.id = newId;
      n.type = editCtx.addTargetType || n.type;
      upsertNodeLocal(n);

      if (editCtx.edge && editCtx.fromId) {
        const e: RawEdge = {
          ...editCtx.edge,
          source: String(editCtx.fromId),
          target: String(newId),
        };
        upsertEdgeLocal(e);
      }
    }

    setEditCtx(null);
  };

  const relApplies = (fromType?: string, toType?: string) => {
    const key = `${(fromType||"").toLowerCase()}-${(toType||"").toLowerCase()}`;
    return REL_EDGE_APPLIES.has(key);
  };

  /* ===== Add helper ===== */
  const openAddFrom = (fromId: string, sourceType: string, choices: Array<{type:string; defaultRelation:string}>) => {
    const target = choices[0];
    const draftNode: RawNode = { id: "", type: target.type, label: "", title: "", content: "", raw: {} };
    const draftEdge: RawEdge = {
      source: fromId,
      target: "",
      relation: target.defaultRelation,
      relevance: 0.6, likelihood: 0.6, weight: 0.6, boost: null,
      raw: { evidence: "" }, // initialize
    };
    setEditCtx({
      mode: "add",
      node: draftNode,
      edge: draftEdge,
      fromId,
      addFromType: sourceType,
      addTargetType: target.type,
    });
  };

  /* ===== base selections ===== */
  const selectCapability = (id: string) => { setSelectedCapability(id); setSelectedPain(null); setSelectedTrigger(null); setSelectedJob(null); setTrail([]); };
  const selectPain = (id: string) => { setSelectedPain(id); setSelectedTrigger(null); setSelectedJob(null); setTrail([]); };
  const selectTrigger = (id: string) => { setSelectedTrigger(id); setSelectedJob(null); setTrail([]); };
  const selectJob = (id: string) => { setSelectedJob(id); setTrail([]); };

  /* ===== trail mutations ===== */
  const pushSolvedPainToTrail = (painId: string) => { setTrail((t) => [...t, { painId }]); };
  const setTrailTrigger = (segIdx: number, triggerId: string) => { setTrail((t) => t.map((s,i)=> i===segIdx?{...s,selectedTriggerId:triggerId, selectedZmotId:null}:s)); };
  const setTrailJob = (segIdx: number, jobId: string) => { setTrail((t) => t.map((s,i)=> i===segIdx?{...s,selectedJobId:jobId, selectedZmotId:null}:s)); };
  const setTrailZmot = (segIdx: number, zmotId: string) => { setTrail((t)=> t.map((s,i)=> i===segIdx?{...s,selectedZmotId:zmotId}:s)); };

  /* ===== Save Graph button actions ===== */
  const hasPending = pendingNodes.size > 0 || pendingEdges.size > 0;
  const openSaveMenu = (e: React.MouseEvent<HTMLElement>) => setSaveAnchor(e.currentTarget);
  const closeSaveMenu = () => setSaveAnchor(null);

  const saveBulk = async (mode: "edited" | "entire") => {
    if (!graph) return;

    const payload =
      mode === "entire"
        ? {
            product_id: selectedProductId,
            nodes: graph.nodes.map(normalizeNode),
            edges: graph.edges,
          }
        : {
            product_id: selectedProductId,
            nodes: Array.from(pendingNodes.values()).map(normalizeNode),
            edges: Array.from(pendingEdges.values()),
          };

    try {
      setLoading(true);
      setStatusMsg("");
      const res = await fetch(`http://localhost:8000/graph/bulk-upsert-graph/${selectedProductId}`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${token}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setStatusMsg(mode === "entire" ? "Saved entire graph." : "Saved edited changes.");
      if (mode === "edited") {
        setPendingNodes(new Map());
        setPendingEdges(new Map());
      }
    } catch (e) {
      setStatusMsg("Failed to save graph updates.");
    } finally {
      setLoading(false);
      closeSaveMenu();
    }
  };

  /* ===== Render bits ===== */
  const getCardBody = (node: RawNode) => (
    <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
      {node.content ? (node.content.length > 140 ? node.content.slice(0, 140) + "…" : node.content) : ""}
    </Typography>
  );

  const breadcrumb = (() => {
    const capNode = getNode(selectedCapability);
    const painNode = getNode(selectedPain);
    const trigNode = getNode(selectedTrigger);
    const jobNode  = getNode(selectedJob);
    return (
      <Stack direction="row" alignItems="center" spacing={1} sx={{ mb: 1, flexWrap: "wrap" }}>
        <Chip variant={capNode ? "filled" : "outlined"} color={capNode ? "primary" : "default"}
          label={capNode ? `Capability: ${capNode.title}` : "Capability"}
          onClick={capNode ? () => { setSelectedPain(null); setSelectedTrigger(null); setSelectedJob(null); setTrail([]);} : undefined}/>
        <ChevronRightIcon fontSize="small" />
        <Chip variant={painNode ? "filled" : "outlined"} color={painNode ? "primary" : "default"}
          label={painNode ? `Pain: ${painNode.title}` : "Pain"}
          onClick={painNode ? () => { setSelectedTrigger(null); setSelectedJob(null); setTrail([]);} : undefined}/>
        <ChevronRightIcon fontSize="small" />
        <Chip variant={trigNode ? "filled" : "outlined"} color={trigNode ? "primary" : "default"}
          label={trigNode ? `Trigger: ${trigNode.title}` : "Trigger"}
          onClick={trigNode ? () => { setSelectedJob(null); setTrail([]);} : undefined}/>
        <ChevronRightIcon fontSize="small" />
        <Chip variant={jobNode ? "filled" : "outlined"} color={jobNode ? "primary" : "default"}
          label={jobNode ? `Job: ${jobNode.title}` : "Job"} />
      </Stack>
    );
  })();

  /* =========================================
     UI
  ========================================= */
  return (
    <Box sx={{ display: "flex", inset: 0, ml: "256px" }} position="absolute">
      <Box component="main" sx={{ flexGrow: 1, p: 3, boxSizing: "border-box", width: "100%", overflowX: "hidden", overflowY: "auto" }}>
        <Toolbar />
        <Stack direction="row" alignItems="center" justifyContent="space-between" sx={{ mb: 1 }}>
          <Typography variant="h6" fontWeight={700}>Product Graph — Explorer</Typography>

          {/* Save Graph Updates (edited vs entire) */}
          <Box>
            <Button
              variant="contained"
              startIcon={<SaveIcon />}
              onClick={openSaveMenu}
              color="primary"
            >
              Save graph updates
            </Button>
            <Menu anchorEl={saveAnchor} open={saveMenuOpen} onClose={closeSaveMenu}>
              <MenuItem disabled={!hasPending} onClick={() => saveBulk("edited")}>
                Save edited changes {hasPending ? `(${pendingNodes.size} nodes, ${pendingEdges.size} edges)` : ""}
              </MenuItem>
              <Divider />
              <MenuItem onClick={() => saveBulk("entire")}>
                Save entire graph ({graph?.nodes.length ?? 0} nodes, {graph?.edges.length ?? 0} edges)
              </MenuItem>
            </Menu>
          </Box>
        </Stack>

        {breadcrumb}

        {statusMsg && <Typography color={/fail|error/i.test(statusMsg) ? "error" : "success"} sx={{ mb: 2 }}>{statusMsg}</Typography>}
        {loading && <LinearProgress sx={{ my: 2 }} />}

        {/* Horizontal column scroller */}
        <Box
          ref={outerColsRef}
          sx={{
            width: "100%",
            mt: 2,
            pb: 4,
            overflowX: "auto",
            overflowY: "hidden",
            WebkitOverflowScrolling: "touch",
            position: "relative",
          }}
        >
          <Box
            sx={{
              display: "inline-flex",
              width: "max-content",
              gap: 2,
              flexWrap: "nowrap",
              alignItems: "flex-start",
              "& > div": {
                flex: `0 0 ${COLUMN_WIDTH}px`,
                minWidth: `${COLUMN_WIDTH}px`,
                maxWidth: `${COLUMN_WIDTH}px`,
                boxSizing: "border-box",
              },
            }}
          >
            {/* ===== Column 1: Capability ===== */}
            <Box>
              <Stack direction="row" justifyContent="space-between" sx={{ mb: 1 }}>
                <Typography variant="subtitle1" fontWeight={700}>Capability</Typography>
                <IconButton size="small" disabled={!selectedCapability} title="Add Pain"
                  onClick={() => selectedCapability && openAddFrom(selectedCapability, "capability", TARGETS_BY_SOURCE["capability"])}>
                  <AddIcon fontSize="small" />
                </IconButton>
              </Stack>

              {capabilities.length ? (
                capabilities.map((c: any) => {
                  const isSel = selectedCapability === c.id;
                  const labelText = displayLabel(c);
                  const edgeToProduct = (c as any).edgeToProduct as RawEdge | undefined;
                  return (
                    <Card key={c.id} sx={{ mb: 2, ...cardSx(isSel, !!selectedCapability) }}>
                      <CardContent>
                        <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                          <Typography variant="subtitle2" fontWeight={700}>{labelText}</Typography>
                          <IconButton size="small" onClick={() => setExpanded(s=>({...s,[c.id]:!s[c.id]}))}><ExpandMoreIcon fontSize="small"/></IconButton>
                        </Box>
                        {getCardBody(c)}
                      </CardContent>
                      <Collapse in={!!expanded[c.id]}>
                        <CardActions sx={{ pt: 0 }}>
                          <Button size="small" onClick={() => selectCapability(c.id)}>Show Pains</Button>
                          <Button size="small" startIcon={<EditIcon/>}
                            onClick={() => setEditCtx({ mode:"edit", node: normalizeNode(c), edge: edgeToProduct, fromId: String(selectedProductId), toId: c.id })}>Edit</Button>
                          <Button size="small" onClick={() => { setDetailNode(c); setDetailOpen(true); }}>Open</Button>
                        </CardActions>
                      </Collapse>
                    </Card>
                  );
                })
              ) : (
                <Typography variant="body2" color="text.secondary">No capabilities</Typography>
              )}
            </Box>

            {/* ===== Column 2: Pain ===== */}
            <Box>
              <Stack direction="row" justifyContent="space-between" sx={{ mb: 1 }}>
                <Typography variant="subtitle1" fontWeight={700}>Pain</Typography>
                <IconButton size="small" disabled={!selectedCapability} title="Add Pain"
                  onClick={() => selectedCapability && openAddFrom(selectedCapability, "capability", TARGETS_BY_SOURCE["capability"])}>
                  <AddIcon fontSize="small" />
                </IconButton>
              </Stack>

              {!selectedCapability ? (
                <Typography variant="body2" color="text.secondary">Select a capability to see pains</Typography>
              ) : painsForCapability.length ? (
                painsForCapability.map(({ node, edge }) => {
                  const isSel = selectedPain === node.id;
                  const labelText = displayLabel(node);
                  return (
                    <Card key={node.id} sx={{ mb: 2, ...cardSx(isSel, !!selectedPain) }}>
                      <CardContent>
                        <Box sx={{ display:"flex", justifyContent:"space-between", alignItems:"center" }}>
                          <Typography variant="subtitle2" fontWeight={700}>{labelText}</Typography>
                          <IconButton size="small" onClick={() => setExpanded(s=>({...s,[node.id]:!s[node.id]}))}><ExpandMoreIcon fontSize="small"/></IconButton>
                        </Box>
                        {getCardBody(node)}
                      </CardContent>
                      <Collapse in={!!expanded[node.id]}>
                        <CardActions sx={{ pt: 0 }}>
                          <Button size="small" onClick={() => selectPain(node.id)}>Explore</Button>
                          <Button size="small" startIcon={<EditIcon/>}
                            onClick={() => setEditCtx({ mode:"edit", node: normalizeNode(node), edge, fromId: String(selectedCapability), toId: node.id })}>Edit</Button>
                          <Button size="small" onClick={() => { setDetailNode(node); setDetailOpen(true); }}>Open</Button>
                        </CardActions>
                      </Collapse>
                    </Card>
                  );
                })
              ) : (
                <Typography variant="body2" color="text.secondary">No pains</Typography>
              )}
            </Box>

            {/* ===== Column 3: Trigger ===== */}
            <Box>
              <Stack direction="row" justifyContent="space-between" sx={{ mb: 1 }}>
                <Typography variant="subtitle1" fontWeight={700}>Trigger</Typography>
                <IconButton size="small" disabled={!selectedPain} title="Add Pain Trigger"
                  onClick={() => selectedPain && openAddFrom(selectedPain, "pain", [{ type: "pain_trigger", defaultRelation: "triggered_by" }])}>
                  <AddIcon fontSize="small" />
                </IconButton>
              </Stack>

              {!selectedPain ? (
                <Typography variant="body2" color="text.secondary">Select a pain to see triggers</Typography>
              ) : (triggersForPainId(selectedPain).length ? triggersForPainId(selectedPain).map(({ node, edge }) => {
                const isSel = selectedTrigger === node.id;
                const labelText = displayLabel(node);
                return (
                  <Card key={node.id} sx={{ mb: 2, ...cardSx(isSel, !!selectedTrigger), cursor:"pointer" }}
                        onClick={() => selectTrigger(node.id)}>
                    <CardContent>
                      <Box sx={{ display:"flex", justifyContent:"space-between", alignItems:"center" }}>
                        <Typography variant="subtitle2" fontWeight={700}>{labelText}</Typography>
                        <IconButton size="small" onClick={(e)=>{ e.stopPropagation(); setExpanded(s=>({...s,[node.id]:!s[node.id]})); }}>
                          <ExpandMoreIcon fontSize="small"/>
                        </IconButton>
                      </Box>
                      {getCardBody(node)}
                    </CardContent>
                    <Collapse in={!!expanded[node.id]}>
                      <CardActions sx={{ pt: 0 }}>
                        <Button size="small" onClick={(e)=>{ e.stopPropagation(); selectTrigger(node.id); }}>Attributes / ZMOT</Button>
                        <Button size="small" startIcon={<EditIcon/>}
                          onClick={(e)=>{ e.stopPropagation(); setEditCtx({ mode:"edit", node: normalizeNode(node), edge, fromId: String(selectedPain), toId: node.id })}}>Edit</Button>
                      </CardActions>
                    </Collapse>
                  </Card>
                );
              }) : <Typography variant="body2" color="text.secondary">No triggers</Typography>)}
            </Box>

            {/* ===== Column 4: Metrics or Attributes ===== */}
            <Box>
              <Stack direction="row" justifyContent="space-between" sx={{ mb: 1 }}>
                <Typography variant="subtitle1" fontWeight={700}>{selectedTrigger ? "Attributes" : "Metrics"}</Typography>
                <IconButton size="small"
                  disabled={selectedTrigger ? !selectedTrigger : !selectedPain}
                  title={selectedTrigger ? "Add Attribute Value" : "Add Metric"}
                  onClick={() => {
                    if (selectedTrigger) openAddFrom(selectedTrigger, "pain_trigger", [{ type: "attribute_value", defaultRelation: "prevalent_in" }]);
                    else if (selectedPain) openAddFrom(selectedPain, "pain", [{ type: "perceived_metric", defaultRelation: "expressed_as" }]);
                  }}>
                  <AddIcon fontSize="small" />
                </IconButton>
              </Stack>

              {!selectedPain ? (
                <Typography variant="body2" color="text.secondary">Select a pain to see metrics</Typography>
              ) : selectedTrigger ? (
                attributesForTriggerId(selectedTrigger).length ? attributesForTriggerId(selectedTrigger).map(({ node, edge }) => (
                  <Card key={node.id} sx={{ mb: 2, ...cardSx(false,false) }}>
                    <CardContent>
                      <Box sx={{ display:"flex", justifyContent:"space-between", alignItems:"center" }}>
                        <Typography variant="subtitle2" fontWeight={700}>{displayLabel(node)}</Typography>
                        <IconButton size="small" title="Add ZMOT (from attribute value)"
                          onClick={() => openAddFrom(node.id, "attribute_value", TARGETS_BY_SOURCE["attribute_value"])}>
                          <AddIcon fontSize="small" />
                        </IconButton>
                      </Box>
                      {getCardBody(node)}
                    </CardContent>
                    <CardActions sx={{ pt: 0 }}>
                      <Button size="small" startIcon={<EditIcon/>}
                        onClick={() => setEditCtx({ mode:"edit", node: normalizeNode(node), edge, fromId: String(selectedTrigger), toId: node.id })}>Edit</Button>
                    </CardActions>
                  </Card>
                )) : <Typography variant="body2" color="text.secondary">No attributes</Typography>
              ) : (
                metricsForPainId(selectedPain).map(({ node, edge }) => (
                  <Card key={node.id} sx={{ mb: 2, ...cardSx(false,false) }}>
                    <CardContent>
                      <Typography variant="subtitle2" fontWeight={700}>{displayLabel(node)}</Typography>
                      {getCardBody(node)}
                    </CardContent>
                    <CardActions sx={{ pt: 0 }}>
                      <Button size="small" startIcon={<EditIcon/>}
                        onClick={() => setEditCtx({ mode:"edit", node: normalizeNode(node), edge, fromId: String(selectedPain), toId: node.id })}>Edit</Button>
                    </CardActions>
                  </Card>
                ))
              )}
            </Box>

            {/* ===== Column 5: Job or ZMOT ===== */}
            <Box>
              <Stack direction="row" justifyContent="space-between" sx={{ mb: 1 }}>
                <Typography variant="subtitle1" fontWeight={700}>{selectedTrigger ? "ZMOT Events" : "Job"}</Typography>
                <IconButton size="small"
                  disabled={selectedTrigger ? !selectedTrigger : !selectedPain}
                  title={selectedTrigger ? "Add ZMOT Event" : "Add Job"}
                  onClick={() => {
                    if (selectedTrigger) openAddFrom(selectedTrigger, "pain_trigger", [{ type: "zmot_event", defaultRelation: "leads_to_zmot" }]);
                    else if (selectedPain) openAddFrom(selectedPain, "pain", [{ type: "job", defaultRelation: "felt_in" }]);
                  }}>
                  <AddIcon fontSize="small" />
                </IconButton>
              </Stack>

              {selectedTrigger ? (
                zmotForTriggerId(selectedTrigger).length ? zmotForTriggerId(selectedTrigger).map(({ node, edge }) => (
                  <Card key={node.id} sx={{ mb: 2, ...cardSx(false,false) }}>
                    <CardContent>
                      <Box sx={{ display:"flex", justifyContent:"space-between", alignItems:"center" }}>
                        <Typography variant="subtitle2" fontWeight={700}>{displayLabel(node)}</Typography>
                        <Stack direction="row" spacing={0}>
                          <IconButton size="small" title="Add Observable Moment"
                            onClick={() => openAddFrom(node.id, "zmot_event", [{ type: "observable_moment", defaultRelation: "observed_as" }])}>
                            <AddIcon fontSize="small"/>
                          </IconButton>
                          <IconButton size="small" title="Add Keyword"
                            onClick={() => openAddFrom(node.id, "zmot_event", [{ type: "keyword", defaultRelation: "keyword" }])}>
                            <AddIcon fontSize="small"/>
                          </IconButton>
                        </Stack>
                      </Box>
                      {getCardBody(node)}
                    </CardContent>
                    <CardActions sx={{ pt: 0 }}>
                      <Button size="small" startIcon={<EditIcon/>}
                        onClick={() => setEditCtx({ mode:"edit", node: normalizeNode(node), edge, fromId: String(selectedTrigger), toId: node.id })}>Edit</Button>
                    </CardActions>
                  </Card>
                )) : <Typography variant="body2" color="text.secondary">No ZMOT events</Typography>
              ) : !selectedPain ? (
                <Typography variant="body2" color="text.secondary">Select a pain to see jobs</Typography>
              ) : (
                jobsForPainId(selectedPain).map(({ node, edge }) => {
                  const isSel = selectedJob === node.id;
                  return (
                    <Card key={node.id} sx={{ mb: 2, ...cardSx(isSel, !!selectedJob) }}>
                      <CardContent>
                        <Box sx={{ display:"flex", justifyContent:"space-between", alignItems:"center" }}>
                          <Typography variant="subtitle2" fontWeight={700}>{displayLabel(node)}</Typography>
                          <IconButton size="small" onClick={() => setExpanded(s=>({...s,[node.id]:!s[node.id]}))}><ExpandMoreIcon fontSize="small"/></IconButton>
                        </Box>
                        {getCardBody(node)}
                      </CardContent>
                      <Collapse in={!!expanded[node.id]}>
                        <CardActions sx={{ pt: 0 }}>
                          <Button size="small" onClick={() => selectJob(node.id)}>Show Personas / Solves</Button>
                          <Button size="small" startIcon={<EditIcon/>}
                            onClick={() => setEditCtx({ mode:"edit", node: normalizeNode(node), edge, fromId: String(selectedPain), toId: node.id })}>Edit</Button>
                          <Button size="small" onClick={() => { setDetailNode(normalizeNode(node)); setDetailOpen(true); }}>Open</Button>
                        </CardActions>
                      </Collapse>
                    </Card>
                  );
                })
              )}
            </Box>

            {/* ===== Column 6: Persona (when job selected) ===== */}
            {selectedJob && (
              <Box>
                <Stack direction="row" justifyContent="space-between" sx={{ mb: 1 }}>
                  <Typography variant="subtitle1" fontWeight={700}>Persona</Typography>
                  <IconButton size="small" title="Add Persona"
                    onClick={() => openAddFrom(selectedJob, "job", [{ type: "persona", defaultRelation: "performed_by" }])}>
                    <AddIcon fontSize="small" />
                  </IconButton>
                </Stack>
                {personasForJobId(selectedJob).map(({ node, edge }) => (
                  <Card key={node.id} sx={{ mb: 2, ...cardSx(false,false) }}>
                    <CardContent>
                      <Typography variant="subtitle2" fontWeight={700}>{displayLabel(node)}</Typography>
                      {getCardBody(node)}
                    </CardContent>
                    <CardActions sx={{ pt: 0 }}>
                      <Button size="small" startIcon={<EditIcon/>}
                        onClick={() => setEditCtx({ mode:"edit", node: normalizeNode(node), edge, fromId: String(selectedJob), toId: node.id })}>Edit</Button>
                    </CardActions>
                  </Card>
                ))}
              </Box>
            )}

            {/* ===== Column 7: Solves (when job selected) ===== */}
            {selectedJob && (
              <Box>
                <Stack direction="row" justifyContent="space-between" sx={{ mb: 1 }}>
                  <Typography variant="subtitle1" fontWeight={700}>Solves (Pains)</Typography>
                  <IconButton size="small" title="Add Solved Pain"
                    onClick={() => openAddFrom(selectedJob, "job", [{ type: "pain", defaultRelation: "solve" }])}>
                    <AddIcon fontSize="small" />
                  </IconButton>
                </Stack>
                {solvesForJobId(selectedJob).length ? solvesForJobId(selectedJob).map(({ node, edge }) => (
                  <Card key={node.id} sx={{ mb: 2, ...cardSx(false,false) }}>
                    <CardContent>
                      <Typography variant="subtitle2" fontWeight={700}>{displayLabel(node)}</Typography>
                      {getCardBody(node)}
                    </CardContent>
                    <CardActions sx={{ pt: 0 }}>
                      <Button size="small" onClick={() => pushSolvedPainToTrail(node.id)}>Explore</Button>
                      <Button size="small" startIcon={<EditIcon/>}
                        onClick={() => setEditCtx({ mode:"edit", node: normalizeNode(node), edge, fromId: String(selectedJob), toId: node.id })}>Edit</Button>
                    </CardActions>
                  </Card>
                )) : <Typography variant="body2" color="text.secondary">No solved pains</Typography>}
              </Box>
            )}

            {/* ===== TRAIL: explored solved pains ===== */}
            {trail.map((seg, idx) => {
              const pain = getNode(seg.painId);
              const triggers = triggersForPainId(seg.painId);
              const metrics = metricsForPainId(seg.painId);
              const jobs = jobsForPainId(seg.painId);
              const solves = seg.selectedJobId ? solvesForJobId(seg.selectedJobId) : [];
              const attrs = seg.selectedTriggerId ? attributesForTriggerId(seg.selectedTriggerId) : [];
              const zmots = seg.selectedTriggerId ? zmotForTriggerId(seg.selectedTriggerId) : [];
              const obs = seg.selectedZmotId ? observableForZmotId(seg.selectedZmotId) : [];
              const kws = seg.selectedZmotId ? keywordsForZmotId(seg.selectedZmotId) : [];

              return (
                <React.Fragment key={`trail-${idx}-${seg.painId}`}>
                  {/* Trigger (trail) */}
                  <Box>
                    <Stack direction="row" justifyContent="space-between" sx={{ mb: 1 }}>
                      <Typography variant="subtitle1" fontWeight={700}>
                        Trigger <Chip size="small" sx={{ ml: 1 }} label={displayLabel(pain)} />
                      </Typography>
                      <IconButton size="small" title="Add Pain Trigger"
                        onClick={() => openAddFrom(seg.painId, "pain", [{ type: "pain_trigger", defaultRelation: "triggered_by" }])}>
                        <AddIcon fontSize="small" />
                      </IconButton>
                    </Stack>
                    {triggers.length ? triggers.map(({ node, edge }) => {
                      const isSel = seg.selectedTriggerId === node.id;
                      return (
                        <Card key={node.id} sx={{ mb: 2, ...cardSx(isSel, !!seg.selectedTriggerId), cursor:"pointer" }}
                              onClick={() => setTrailTrigger(idx, node.id)}>
                          <CardContent>
                            <Box sx={{ display:"flex", justifyContent:"space-between", alignItems:"center" }}>
                              <Typography variant="subtitle2" fontWeight={700}>{displayLabel(node)}</Typography>
                              <IconButton size="small" onClick={(e)=>{ e.stopPropagation(); setExpanded(s=>({...s,[node.id]:!s[node.id]})); }}>
                                <ExpandMoreIcon fontSize="small"/>
                              </IconButton>
                            </Box>
                            {getCardBody(node)}
                          </CardContent>
                          <Collapse in={!!expanded[node.id]}>
                            <CardActions sx={{ pt: 0 }}>
                              <Button size="small" onClick={(e)=>{ e.stopPropagation(); setTrailTrigger(idx, node.id); }}>Attributes / ZMOT</Button>
                              <Button size="small" startIcon={<EditIcon/>}
                                onClick={(e)=>{ e.stopPropagation(); setEditCtx({ mode:"edit", node: normalizeNode(node), edge, fromId: String(seg.painId), toId: node.id })}}>Edit</Button>
                            </CardActions>
                          </Collapse>
                        </Card>
                      );
                    }) : <Typography variant="body2" color="text.secondary">No triggers</Typography>}
                  </Box>

                  {/* Metrics (trail) */}
                  <Box>
                    <Stack direction="row" justifyContent="space-between" sx={{ mb: 1 }}>
                      <Typography variant="subtitle1" fontWeight={700}>
                        Metrics <Chip size="small" sx={{ ml: 1 }} label={displayLabel(pain)} />
                      </Typography>
                      <IconButton size="small" title="Add Metric"
                        onClick={() => openAddFrom(seg.painId, "pain", [{ type: "perceived_metric", defaultRelation: "expressed_as" }])}>
                        <AddIcon fontSize="small" />
                      </IconButton>
                    </Stack>
                    {metrics.length ? metrics.map(({ node, edge }) => (
                      <Card key={node.id} sx={{ mb: 2, ...cardSx(false,false) }}>
                        <CardContent>
                          <Typography variant="subtitle2" fontWeight={700}>{displayLabel(node)}</Typography>
                          {getCardBody(node)}
                        </CardContent>
                        <CardActions sx={{ pt: 0 }}>
                          <Button size="small" startIcon={<EditIcon/>}
                            onClick={() => setEditCtx({ mode:"edit", node: normalizeNode(node), edge, fromId: String(seg.painId), toId: node.id })}>Edit</Button>
                        </CardActions>
                      </Card>
                    )) : <Typography variant="body2" color="text.secondary">No metrics</Typography>}
                  </Box>

                  {/* Jobs (trail) */}
                  <Box>
                    <Stack direction="row" justifyContent="space-between" sx={{ mb: 1 }}>
                      <Typography variant="subtitle1" fontWeight={700}>
                        Job <Chip size="small" sx={{ ml: 1 }} label={displayLabel(pain)} />
                      </Typography>
                      <IconButton size="small" title="Add Job"
                        onClick={() => openAddFrom(seg.painId, "pain", [{ type: "job", defaultRelation: "felt_in" }])}>
                        <AddIcon fontSize="small" />
                      </IconButton>
                    </Stack>
                    {jobs.length ? jobs.map(({ node, edge }) => {
                      const isSel = seg.selectedJobId === node.id;
                      return (
                        <Card key={node.id} sx={{ mb: 2, ...cardSx(isSel, !!seg.selectedJobId) }}>
                          <CardContent>
                            <Box sx={{ display:"flex", justifyContent:"space-between", alignItems:"center" }}>
                              <Typography variant="subtitle2" fontWeight={700}>{displayLabel(node)}</Typography>
                              <IconButton size="small" onClick={() => setExpanded(s=>({...s,[node.id]:!s[node.id]}))}><ExpandMoreIcon fontSize="small"/></IconButton>
                            </Box>
                            {getCardBody(node)}
                          </CardContent>
                          <Collapse in={!!expanded[node.id]}>
                            <CardActions sx={{ pt: 0 }}>
                              <Button size="small" onClick={() => setTrailJob(idx, node.id)}>Show Solves</Button>
                              <Button size="small" startIcon={<EditIcon/>}
                                onClick={() => setEditCtx({ mode:"edit", node: normalizeNode(node), edge, fromId: String(seg.painId), toId: node.id })}>Edit</Button>
                            </CardActions>
                          </Collapse>
                        </Card>
                      );
                    }) : <Typography variant="body2" color="text.secondary">No jobs</Typography>}
                  </Box>

                  {/* Solves (trail) */}
                  <Box>
                    <Stack direction="row" justifyContent="space-between" sx={{ mb: 1 }}>
                      <Typography variant="subtitle1" fontWeight={700}>Solves (Pains)</Typography>
                      {seg.selectedJobId && (
                        <IconButton size="small" title="Add Solved Pain"
                          onClick={() => openAddFrom(seg.selectedJobId!, "job", [{ type: "pain", defaultRelation: "solve" }])}>
                          <AddIcon fontSize="small" />
                        </IconButton>
                      )}
                    </Stack>
                    {seg.selectedJobId ? (
                      solves.length ? solves.map(({ node, edge }) => (
                        <Card key={node.id} sx={{ mb: 2, ...cardSx(false,false) }}>
                          <CardContent>
                            <Typography variant="subtitle2" fontWeight={700}>{displayLabel(node)}</Typography>
                            {getCardBody(node)}
                          </CardContent>
                          <CardActions sx={{ pt: 0 }}>
                            <Button size="small" onClick={() => pushSolvedPainToTrail(node.id)}>Explore</Button>
                            <Button size="small" startIcon={<EditIcon/>}
                              onClick={() => setEditCtx({ mode:"edit", node: normalizeNode(node), edge, fromId: String(seg.selectedJobId), toId: node.id })}>Edit</Button>
                          </CardActions>
                        </Card>
                      )) : <Typography variant="body2" color="text.secondary">No solved pains</Typography>
                    ) : (
                      <Typography variant="body2" color="text.secondary">Select a job to see solved pains</Typography>
                    )}
                  </Box>

                  {/* Attributes (trail) */}
                  <Box>
                    <Stack direction="row" justifyContent="space-between" sx={{ mb: 1 }}>
                      <Typography variant="subtitle1" fontWeight={700}>
                        {seg.selectedTriggerId ? "Attributes" : "Attributes — select a trigger"}
                      </Typography>
                      {seg.selectedTriggerId && (
                        <IconButton size="small" title="Add Attribute Value"
                          onClick={() => openAddFrom(seg.selectedTriggerId!, "pain_trigger", [{ type: "attribute_value", defaultRelation: "prevalent_in" }])}>
                          <AddIcon fontSize="small" />
                        </IconButton>
                      )}
                    </Stack>
                    {!seg.selectedTriggerId ? (
                      <Typography variant="body2" color="text.secondary">Select a trigger to see attributes</Typography>
                    ) : attrs.length ? attrs.map(({ node, edge }) => (
                      <Card key={node.id} sx={{ mb: 2, ...cardSx(false,false) }}>
                        <CardContent>
                          <Box sx={{ display:"flex", justifyContent:"space-between", alignItems:"center" }}>
                            <Typography variant="subtitle2" fontWeight={700}>{displayLabel(node)}</Typography>
                            <IconButton size="small" title="Add ZMOT from attribute"
                              onClick={() => openAddFrom(node.id, "attribute_value", TARGETS_BY_SOURCE["attribute_value"])}>
                              <AddIcon fontSize="small" />
                            </IconButton>
                          </Box>
                          {getCardBody(node)}
                        </CardContent>
                        <CardActions sx={{ pt: 0 }}>
                          <Button size="small" startIcon={<EditIcon/>}
                            onClick={() => setEditCtx({ mode:"edit", node: normalizeNode(node), edge, fromId: String(seg.selectedTriggerId), toId: node.id })}>Edit</Button>
                        </CardActions>
                      </Card>
                    )) : <Typography variant="body2" color="text.secondary">No attributes</Typography>}
                  </Box>

                  {/* ZMOT (trail) */}
                  <Box>
                    <Stack direction="row" justifyContent="space-between" sx={{ mb: 1 }}>
                      <Typography variant="subtitle1" fontWeight={700}>
                        {seg.selectedTriggerId ? "ZMOT Events" : "ZMOT — select a trigger"}
                      </Typography>
                      {seg.selectedTriggerId && (
                        <IconButton size="small" title="Add ZMOT Event"
                          onClick={() => openAddFrom(seg.selectedTriggerId!, "pain_trigger", [{ type: "zmot_event", defaultRelation: "leads_to_zmot" }])}>
                          <AddIcon fontSize="small" />
                        </IconButton>
                      )}
                    </Stack>
                    {!seg.selectedTriggerId ? (
                      <Typography variant="body2" color="text.secondary">Select a trigger to see ZMOT</Typography>
                    ) : zmots.length ? zmots.map(({ node, edge }) => {
                      const isSel = seg.selectedZmotId === node.id;
                      return (
                        <Card key={node.id} sx={{ mb: 2, ...cardSx(isSel, !!seg.selectedZmotId), cursor:"pointer" }}
                              onClick={() => setTrailZmot(idx, node.id)}>
                          <CardContent>
                            <Box sx={{ display:"flex", justifyContent:"space-between", alignItems:"center" }}>
                              <Typography variant="subtitle2" fontWeight={700}>{displayLabel(node)}</Typography>
                              <Stack direction="row" spacing={0} onClick={(e)=>e.stopPropagation()}>
                                <IconButton size="small" title="Add Observable Moment"
                                  onClick={() => openAddFrom(node.id, "zmot_event", [{ type: "observable_moment", defaultRelation: "observed_as" }])}>
                                  <AddIcon fontSize="small"/>
                                </IconButton>
                                <IconButton size="small" title="Add Keyword"
                                  onClick={() => openAddFrom(node.id, "zmot_event", [{ type: "keyword", defaultRelation: "keyword" }])}>
                                  <AddIcon fontSize="small"/>
                                </IconButton>
                              </Stack>
                            </Box>
                            {getCardBody(node)}
                          </CardContent>
                          <CardActions sx={{ pt: 0 }}>
                            <Button size="small" startIcon={<EditIcon/>}
                              onClick={(e)=>{ e.stopPropagation(); setEditCtx({ mode:"edit", node: normalizeNode(node), edge, fromId: String(seg.selectedTriggerId), toId: node.id })}}>Edit</Button>
                          </CardActions>
                        </Card>
                      );
                    }) : <Typography variant="body2" color="text.secondary">No ZMOT events</Typography>}
                  </Box>

                  {/* Observable Moments (trail) */}
                  <Box>
                    <Stack direction="row" justifyContent="space-between" sx={{ mb: 1 }}>
                      <Typography variant="subtitle1" fontWeight={700}>
                        {seg.selectedZmotId ? "Observable Moments" : "Observable — select a ZMOT"}
                      </Typography>
                      {seg.selectedZmotId && (
                        <IconButton size="small" title="Add Observable Moment"
                          onClick={() => openAddFrom(seg.selectedZmotId!, "zmot_event", [{ type: "observable_moment", defaultRelation: "observed_as" }])}>
                          <AddIcon fontSize="small" />
                        </IconButton>
                      )}
                    </Stack>
                    {!seg.selectedZmotId ? (
                      <Typography variant="body2" color="text.secondary">Select a ZMOT to see Observable Moments</Typography>
                    ) : obs.length ? obs.map(({ node, edge }) => (
                      <Card key={node.id} sx={{ mb: 2, ...cardSx(false,false) }}>
                        <CardContent>
                          <Typography variant="subtitle2" fontWeight={700}>{displayLabel(node)}</Typography>
                          {getCardBody(node)}
                        </CardContent>
                        <CardActions sx={{ pt: 0 }}>
                          <Button size="small" startIcon={<EditIcon/>}
                            onClick={() => setEditCtx({ mode:"edit", node: normalizeNode(node), edge, fromId: String(seg.selectedZmotId), toId: node.id })}>Edit</Button>
                        </CardActions>
                      </Card>
                    )) : <Typography variant="body2" color="text.secondary">No observable moments</Typography>}
                  </Box>

                  {/* Keywords (trail) */}
                  <Box ref={idx === trail.length - 1 ? lastColumnRef : undefined}>
                    <Stack direction="row" justifyContent="space-between" sx={{ mb: 1 }}>
                      <Typography variant="subtitle1" fontWeight={700}>
                        {seg.selectedZmotId ? "Keywords" : "Keywords — select a ZMOT"}
                      </Typography>
                      {seg.selectedZmotId && (
                        <IconButton size="small" title="Add Keyword"
                          onClick={() => openAddFrom(seg.selectedZmotId!, "zmot_event", [{ type: "keyword", defaultRelation: "keyword" }])}>
                          <AddIcon fontSize="small" />
                        </IconButton>
                      )}
                    </Stack>
                    {!seg.selectedZmotId ? (
                      <Typography variant="body2" color="text.secondary">Select a ZMOT to see Keywords</Typography>
                    ) : kws.length ? kws.map(({ node, edge }) => (
                      <Card key={node.id} sx={{ mb: 2, ...cardSx(false,false) }}>
                        <CardContent>
                          <Typography variant="subtitle2" fontWeight={700}>{displayLabel(node)}</Typography>
                          {getCardBody(node)}
                        </CardContent>
                        <CardActions sx={{ pt: 0 }}>
                          <Button size="small" startIcon={<EditIcon/>}
                            onClick={() => setEditCtx({ mode:"edit", node: normalizeNode(node), edge, fromId: String(seg.selectedZmotId), toId: node.id })}>Edit</Button>
                        </CardActions>
                      </Card>
                    )) : <Typography variant="body2" color="text.secondary">No keywords</Typography>}
                  </Box>
                </React.Fragment>
              );
            })}

            {/* Sentinel when trail empty */}
            {!trail.length && <Box ref={lastColumnRef} sx={{ width: 1, minWidth: 1, maxWidth: 1 }} />}
          </Box>
        </Box>

        {/* ===== Unified Add/Edit Dialog ===== */}
        <Dialog open={!!editCtx} onClose={() => setEditCtx(null)} maxWidth="sm" fullWidth>
          <DialogTitle>{editCtx?.mode === "add" ? "Add" : "Edit"}</DialogTitle>
          {editCtx && (
            <>
              <DialogContent dividers>
                {/* NODE SECTION */}
                <Typography variant="subtitle2" sx={{ mb: 1 }}>Node</Typography>
                <Typography variant="caption" color="text.secondary" sx={{ display:"block", mb: 2 }}>
                  Type: {editCtx?.mode === "add" ? (editCtx.addTargetType ?? "—") : (editCtx.node.type ?? "—")} • Data source: {editCtx.node.raw?.data_source ?? "unknown"} (server will set to "user_input" on save)
                </Typography>

                {editCtx.mode === "add" && editCtx.addFromType && (
                  <FormControl fullWidth sx={{ mb: 2 }}>
                    <InputLabel id="new-type-label">Target Type</InputLabel>
                    <Select
                      labelId="new-type-label"
                      label="Target Type"
                      value={editCtx.addTargetType ?? ""}
                      onChange={(e) => {
                        const val = String(e.target.value);
                        setEditCtx((ctx) => {
                          if (!ctx) return ctx;
                          const opts = TARGETS_BY_SOURCE[ctx.addFromType!] || [];
                          const found = opts.find(o => o.type === val);
                          return {
                            ...ctx,
                            addTargetType: val,
                            node: { ...ctx.node, type: val },
                            edge: { ...(ctx.edge || {}), relation: found?.defaultRelation || "" },
                          };
                        });
                      }}
                    >
                      {(TARGETS_BY_SOURCE[editCtx.addFromType] || []).map(o =>
                        <MenuItem key={o.type} value={o.type}>{o.type}</MenuItem>
                      )}
                    </Select>
                  </FormControl>
                )}

                {(() => {
                  const n = editCtx.node;
                  const t = (editCtx.mode === "add" ? editCtx.addTargetType : n.type) || "";
                  const setNode = (np: Partial<RawNode>) => setEditCtx({ ...editCtx, node: { ...n, ...np } });
                  const setRaw = (k: string, v: any) => setNode({ raw: { ...(n.raw || {}), [k]: v } });
                  const text = (label: string, value: any, set:(v:any)=>void, extra:any={}) =>
                    <TextField label={label} value={value ?? ""} onChange={(e)=>set(e.target.value)} fullWidth sx={{ mb:2 }} {...extra} />;

                  const lower = (t || "").toLowerCase();

                  if (lower.includes("capability")) {
                    return (
                      <Stack>
                        {text("Name", n.raw?.name ?? n.label, (v)=>{ setNode({ label:v, title:v }); setRaw("name", v); })}
                        {text("Description", n.raw?.description ?? n.content, (v)=>{ setNode({ content:v }); setRaw("description", v); }, { multiline:true, minRows:4 })}
                      </Stack>
                    );
                  }
                  if (lower === "pain") {
                    return (
                      <Stack>
                        {text("Description", n.raw?.description ?? n.content, (v)=>{ setNode({ content:v }); setRaw("description", v); }, { multiline:true, minRows:4 })}
                        {text("pain_source", n.raw?.pain_source, (v)=>setRaw("pain_source", v))}
                      </Stack>
                    );
                  }
                  if (lower === "perceived_metric") {
                    return (<Stack>{text("metric", n.raw?.metric ?? n.label, (v)=>{ setNode({ label:v, title:v }); setRaw("metric", v); })}</Stack>);
                  }
                  if (lower === "job") {
                    return (<Stack>{text("Description", n.raw?.description ?? n.content, (v)=>{ setNode({ content:v }); setRaw("description", v); }, { multiline:true, minRows:4 })}</Stack>);
                  }
                  if (lower === "persona") {
                    // NEW: LinkedIn Profiles editor (array of {url, bio})
                    const profiles: Array<{url:string; bio:string}> = Array.isArray(n.raw?.linkedin_profiles) ? n.raw.linkedin_profiles : [];
                    const updateProfile = (i: number, key: "url"|"bio", value: string) => {
                      const next = [...profiles];
                      const row = { ...(next[i] || {url:"", bio:""}) };
                      row[key] = value;
                      next[i] = row;
                      setRaw("linkedin_profiles", next);
                    };
                    const addProfile = () => setRaw("linkedin_profiles", [...profiles, { url: "", bio: "" }]);
                    const removeProfile = (i: number) => {
                      const next = [...profiles];
                      next.splice(i, 1);
                      setRaw("linkedin_profiles", next);
                    };

                    return (
                      <Stack>
                        {text("Title", n.raw?.title ?? n.label, (v)=>{ setNode({ label:v, title:v }); setRaw("title", v); })}
                        {text("Department", n.raw?.department, (v)=>setRaw("department", v))}
                        {text("Seniority", n.raw?.seniority, (v)=>setRaw("seniority", v))}
                        <Divider sx={{ my: 1.5 }} />
                        <Typography variant="subtitle2" sx={{ mb: 1 }}>LinkedIn Profiles (for realness validation)</Typography>
                        <Stack spacing={2}>
                          {profiles.map((p, i) => (
                            <Box key={`li-${i}`} sx={{ p: 1.5, border: `1px solid ${theme.palette.divider}`, borderRadius: 1 }}>
                              <Stack direction="row" alignItems="center" spacing={1}>
                                <TextField
                                  label="Profile URL"
                                  value={p.url}
                                  onChange={(e)=>updateProfile(i, "url", e.target.value)}
                                  fullWidth
                                  sx={{ mb: 1 }}
                                  placeholder="https://www.linkedin.com/in/janedoe"
                                />
                                <IconButton aria-label="remove profile" onClick={()=>removeProfile(i)}><DeleteIcon fontSize="small" /></IconButton>
                              </Stack>
                              <TextField
                                label="Short Bio / Why they match"
                                value={p.bio}
                                onChange={(e)=>updateProfile(i, "bio", e.target.value)}
                                fullWidth
                                multiline
                                minRows={2}
                                placeholder="Head of RevOps at mid-market SaaS; 10+ yrs in CRM, ICP matches our target"
                              />
                            </Box>
                          ))}
                          <Button onClick={addProfile} startIcon={<AddIcon/>} variant="outlined">Add LinkedIn Profile</Button>
                        </Stack>
                      </Stack>
                    );
                  }
                  if (lower === "pain_trigger") {
                    return (<Stack>{text("attribute", n.raw?.attribute ?? n.label, (v)=>{ setNode({ label:v, title:v }); setRaw("attribute", v); })}</Stack>);
                  }
                  if (lower === "attribute_value") {
                    return (
                      <Stack>
                        <FormControl fullWidth sx={{ mb:2 }}>
                          <InputLabel id="dim-label">dimension</InputLabel>
                          <Select labelId="dim-label" label="dimension" value={n.raw?.dimension ?? ""} onChange={(e)=>setRaw("dimension", e.target.value)}>
                            {DIMENSION_OPTIONS.map(opt => <MenuItem key={opt} value={opt}>{opt}</MenuItem>)}
                          </Select>
                        </FormControl>
                        {text("name", n.raw?.name ?? n.label, (v)=>{ setNode({ label:v, title:v }); setRaw("name", v); })}
                      </Stack>
                    );
                  }
                  if (lower === "zmot_event") {
                    return (<Stack>{text("event", n.raw?.event ?? n.label, (v)=>{ setNode({ label:v, title:v }); setRaw("event", v); })}</Stack>);
                  }
                  if (lower === "observable_moment" || lower === "keyword") {
                    return (<Stack>{text("text", n.raw?.text ?? n.label, (v)=>{ setNode({ label:v, title:v }); setRaw("text", v); }, { multiline:true, minRows:3 })}</Stack>);
                  }

                  return (
                    <Stack>
                      {text("Label", n.label, (v)=>setNode({ label:v }))}
                      {text("Content", n.content, (v)=>setNode({ content:v }), { multiline:true, minRows:4 })}
                    </Stack>
                  );
                })()}

                {/* EDGE SECTION */}
                {editCtx.edge && (
                  <>
                    <Divider sx={{ my: 2 }} />
                    <Typography variant="subtitle2" sx={{ mb: 1 }}>Edge</Typography>

                    {/* Context — show source (and target) labels */}
                    {(() => {
                      const src = getNode(editCtx.fromId || editCtx.edge?.source);
                      const tgt = getNode(editCtx.toId || editCtx.edge?.target || editCtx.node.id);
                      return (
                        <Typography variant="body2" sx={{ mb: 1 }}>
                          Editing edge from <strong>{displayLabel(src) || (editCtx.edge?.source ?? "—")}</strong>
                          {" "}to{" "}
                          <strong>{displayLabel(tgt) || (editCtx.edge?.target ?? editCtx.node.id ?? "—")}</strong>
                        </Typography>
                      );
                    })()}

                    <Stack spacing={2}>
                      <TextField
                        label="Relation"
                        value={editCtx.edge.relation ?? ""}
                        onChange={(e)=>setEditCtx({ ...editCtx, edge: { ...(editCtx.edge as RawEdge), relation: e.target.value } })}
                        fullWidth
                      />
                      {(() => {
                        const fromType = editCtx.addFromType ?? getNode(editCtx.fromId)?.type;
                        const toType = editCtx.addTargetType ?? editCtx.node.type;
                        const showQuant = REL_EDGE_APPLIES.has(`${(fromType||"").toLowerCase()}-${(toType||"").toLowerCase()}`);
                        const showBoost = BOOST_APPLIES(fromType, toType);
                        return (
                          <>
                            {showQuant && (
                              <>
                                <TextField type="number" inputProps={{ step:"0.1" }} label="Relevance"
                                  value={editCtx.edge.relevance ?? ""}
                                  onChange={(e)=>setEditCtx({ ...editCtx, edge: { ...(editCtx.edge as RawEdge), relevance: Number(e.target.value) } })}
                                  fullWidth/>
                                <TextField type="number" inputProps={{ step:"0.1" }} label="Likelihood"
                                  value={editCtx.edge.likelihood ?? ""}
                                  onChange={(e)=>setEditCtx({ ...editCtx, edge: { ...(editCtx.edge as RawEdge), likelihood: Number(e.target.value) } })}
                                  fullWidth/>
                                <TextField type="number" inputProps={{ step:"0.1" }} label="Weight"
                                  value={editCtx.edge.weight ?? ""}
                                  onChange={(e)=>setEditCtx({ ...editCtx, edge: { ...(editCtx.edge as RawEdge), weight: Number(e.target.value) } })}
                                  fullWidth/>
                              </>
                            )}
                            {showBoost && (
                              <TextField type="number" inputProps={{ step:"0.1" }} label="Boost"
                                value={editCtx.edge.boost ?? ""}
                                onChange={(e)=>setEditCtx({ ...editCtx, edge: { ...(editCtx.edge as RawEdge), boost: Number(e.target.value) } })}
                                fullWidth/>
                            )}
                          </>
                        );
                      })()}

                      {/* NEW: Evidence field */}
                      <TextField
                        label="Evidence (optional)"
                        value={editCtx.edge.raw?.evidence ?? ""}
                        onChange={(e)=>setEditCtx({
                          ...editCtx,
                          edge: { ...(editCtx.edge as RawEdge), raw: { ...(editCtx.edge?.raw || {}), evidence: e.target.value } }
                        })}
                        fullWidth
                        multiline
                        minRows={2}
                        placeholder="Quick reason / source for this relationship or update"
                      />
                    </Stack>
                  </>
                )}
              </DialogContent>
              <DialogActions>
                <Button onClick={() => setEditCtx(null)}>Cancel</Button>
                <Button variant="contained" onClick={saveEditCtx}>Save</Button>
              </DialogActions>
            </>
          )}
        </Dialog>

        {/* Detail (read-only) */}
        <Dialog open={detailOpen} onClose={() => setDetailOpen(false)} maxWidth="md" fullWidth>
          <DialogTitle>{displayLabel(detailNode) || detailNode?.id}</DialogTitle>
          <DialogContent dividers>
            <Typography variant="body1" sx={{ mb: 2 }}>{detailNode?.content}</Typography>
            <Typography variant="subtitle2" sx={{ mt: 1, mb: 1 }}>Incoming sources & edge metrics</Typography>
            <List dense>
              {detailNode && graph ? (
                getIncoming(detailNode.id).map(({ edge, node }) => (
                  <React.Fragment key={`${node.id}-${edge.source}-${edge.target}`}>
                    <ListItem alignItems="flex-start">
                      <ListItemText
                        primary={displayLabel(node)}
                        secondary={
                          <>
                            <span>Relation: {String(edge.relation ?? "—")}</span><br/>
                            <span>Relevance: {edge.relevance ?? "—"} • Likelihood: {edge.likelihood ?? "—"} • Weight: {edge.weight ?? "—"} • Boost: {edge.boost ?? "—"}</span><br/>
                            {edge.raw?.evidence ? <em>Evidence: {edge.raw.evidence}</em> : null}
                          </>
                        }
                      />
                    </ListItem>
                    <Divider component="li" />
                  </React.Fragment>
                ))
              ) : <ListItem><ListItemText primary="No incoming sources" /></ListItem>}
            </List>
          </DialogContent>
        </Dialog>
      </Box>
    </Box>
  );
}
