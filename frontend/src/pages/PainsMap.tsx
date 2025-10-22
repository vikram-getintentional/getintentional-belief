import { useEffect, useMemo, useState } from "react";
import {
   Box,
   Typography,
   Select,
   MenuItem,
   FormControl,
   InputLabel,
   LinearProgress,
   Toolbar,
   Card,
   CardContent,
   CardActions,
   Button,
   IconButton,
   Collapse,
   Grid,
 } from "@mui/material";
 import { useTheme } from "@mui/material/styles";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";

type Node = {
  id: string;
  type: string;
  title?: string;
  content?: string;
  relevance?: number;
  likelihood?: number;
  // optional: product linkage if backend provides
  product_id?: string;
};

type Edge = {
  source: string;
  target: string;
  // optional metadata
  label?: string;
};

export default function PainsMap() {
    const theme = useTheme();
    const token = localStorage.getItem("token");
    const [products, setProducts] = useState<{ id: string; name: string }[]>([]);
    const [selectedProductId, setSelectedProductId] = useState<string>("");
    const [statusMsg, setStatusMsg] = useState("");
    const [loading, setLoading] = useState(false);

    const [graphNodes, setGraphNodes] = useState<Node[]>([]);
    const [graphEdges, setGraphEdges] = useState<Edge[]>([]);
    const [hopIndex, setHopIndex] = useState(0);
    const [expanded, setExpanded] = useState<Record<string, boolean>>({});

  // fetch company + products
  useEffect(() => {
    (async () => {
      try {
        const me = await fetch("http://localhost:8000/me", {
          headers: { Authorization: `Bearer ${token}` },
        }).then((r) => r.json());
        const prod = await fetch(
          `http://localhost:8000/get-products/${me.company_id}`,
          {
            headers: { Authorization: `Bearer ${token}` },
          }
        ).then((r) => r.json());
        if (!prod.products?.length) {
          setStatusMsg("No products found. Please run Value Prop first.");
          return;
        }
        setProducts(prod.products);
        if (prod.products.length === 1) setSelectedProductId(prod.products[0].id);
      } catch (err) {
        setStatusMsg("Error fetching company or products.");
      }
    })();
  }, [token]);

  // fetch product graph when a product is selected
  useEffect(() => {
    if (!selectedProductId) return;
    setLoading(true);
    setStatusMsg("");
    fetch(
      `http://localhost:8000/get-product-graph/${selectedProductId}`,
      { headers: { Authorization: `Bearer ${token}` } }
    )
      .then((r) => r.json())
      .then((data) => {
        // expect data.nodes and data.edges or a flat array
        console.log("Fetched product graph data:", data);
        const nodes: Node[] = data.nodes || data.graph?.nodes || data;
        const edges: Edge[] = data.edges || data.graph?.edges || [];
        setGraphNodes(Array.isArray(nodes) ? nodes : []);
        setGraphEdges(Array.isArray(edges) ? edges : []);
        setHopIndex(0);
        setExpanded({});
        setLoading(false);
      })
      .catch(() => {
        setStatusMsg("Failed to load product graph.");
        setGraphNodes([]);
        setGraphEdges([]);
        setLoading(false);
      });
  }, [selectedProductId, token]);

  // adjacency map for BFS / connections
  const adjacency = useMemo(() => {
    const map = new Map<string, Set<string>>();
    graphEdges.forEach((e) => {
      if (!map.has(e.source)) map.set(e.source, new Set());
      if (!map.has(e.target)) map.set(e.target, new Set());
      map.get(e.source)!.add(e.target);
      map.get(e.target)!.add(e.source);
    });
    return map;
  }, [graphEdges]);

  const nodesById = useMemo(() => {
    const m = new Map<string, Node>();
    graphNodes.forEach((n) => m.set(n.id, n));
    return m;
  }, [graphNodes]);

  // find capability seed ids: nodes of type 'capability' that are linked to the selected product (or fallback to all capabilities)
  const seedIds = useMemo(() => {
    // if graph contains a product node with id === selectedProductId, find capability neighbors
    const productNode = graphNodes.find((n) => n.id === selectedProductId);
    let seeds: string[] = [];
    if (productNode && adjacency.has(productNode.id)) {
      adjacency.get(productNode.id)!.forEach((nid) => {
        const nn = nodesById.get(nid);
        if (nn && nn.type === "capability") seeds.push(nid);
      });
    }
    if (!seeds.length) {
      // fallback: all capability nodes
      graphNodes.forEach((n) => {
        if (n.type === "capability") seeds.push(n.id);
      });
    }
    // final fallback: first node
    if (!seeds.length && graphNodes.length) seeds = [graphNodes[0].id];
    return seeds;
  }, [graphNodes, adjacency, nodesById, selectedProductId]);

  // BFS distances from seeds
  const distances = useMemo(() => {
    const dist = new Map<string, number>();
    const q: string[] = [];
    seedIds.forEach((id) => {
      dist.set(id, 0);
      q.push(id);
    });
    while (q.length) {
      const cur = q.shift()!;
      const d = dist.get(cur)!;
      const neigh = adjacency.get(cur);
      if (!neigh) continue;
      neigh.forEach((nid) => {
        if (!dist.has(nid)) {
          dist.set(nid, d + 1);
          q.push(nid);
        }
      });
    }
    return dist;
  }, [seedIds, adjacency]);

  // column configuration per hop window
  const getColumnsForHop = (h: number) => {
    if (h === 0) return ["capability", "pain", "job", "persona"];
    return ["job", "pain", "persona"];
  };

  const columns = useMemo(() => getColumnsForHop(hopIndex), [hopIndex]);

  // compute nodes to show in each column for current hopIndex.
  // rule: for column index j, show nodes of that columnType whose distance === j + hopIndex
  const columnNodes = useMemo(() => {
    const result: Record<string, Node[]> = {};
    columns.forEach((col, j) => {
      const targetDistance = j + hopIndex;
      result[col] = graphNodes.filter((n) => {
        const d = distances.get(n.id);
        return n.type === col && typeof d === "number" && d === targetDistance;
      });
    });
    return result;
  }, [columns, graphNodes, distances, hopIndex]);

  const toggleExpand = (id: string) => {
    setExpanded((s) => ({ ...s, [id]: !s[id] }));
  };

  const getConnectedOfTypes = (nodeId: string, types: string[]) => {
    const neigh = adjacency.get(nodeId);
    if (!neigh) return [] as Node[];
    const arr: Node[] = [];
    neigh.forEach((nid) => {
      const n = nodesById.get(nid);
      if (n && types.includes(n.type)) arr.push(n);
    });
    return arr;
  };

  return (
    <Grid size = {12}>
        <Box sx={{ display: "flex", flexDirection: "column" }}>
        <Box sx={{ display: "flex" }}>
            <Box component="main" sx={{ flexGrow: 1, p: 3 }}>
            <Toolbar />
            <Typography variant="h5" fontWeight={700} gutterBottom>
                Product Graph — Kanban
            </Typography>

            {products.length > 1 && (
                <Box 
                    sx={{ 
                        display: "flex",
                        gap: 2,
                        mt: 2,
                        overflow:"auto",
                        pb: 4,
                        width: "100%",
                        px: 0.5
                    }}
                >
                <FormControl fullWidth>
                    <InputLabel id="prod-label">Select Product</InputLabel>
                    <Select
                    labelId="prod-label"
                    label="Select Product"
                    value={selectedProductId}
                    onChange={(e) => setSelectedProductId(String(e.target.value))}
                    >
                    {products.map((p) => (
                        <MenuItem key={p.id} value={p.id}>
                        {p.name}
                        </MenuItem>
                    ))}
                    </Select>
                </FormControl>
                </Box>
            )}

            {statusMsg && <Typography color="error" sx={{ mb: 2 }}>{statusMsg}</Typography>}
            {loading && <LinearProgress sx={{ my: 2 }} />}

            <Box 
                sx={{ 
                    display: "flex", 
                    gap: 2, 
                    mt: 2, 
                    overflowX: "auto", 
                    pb: 4,
                    width: "100%",
                    px: 0.5
                }}
                >
                {columns.map((col) => (
                <Box 
                    key={col} 
                    sx={{ 
                        minWidth: 300,
                        maxWidth: 360,
                        flex: "0 0 auto"
                        }}>
                    <Typography variant="subtitle1" fontWeight={700} sx={{ mb: 1, textTransform: "capitalize" }}>
                    {col.replace("_", " ")}
                    </Typography>
                    {columnNodes[col]?.length ? (
                    columnNodes[col].map((n) => (
                        <Card key={n.id} sx={{ mb: 2 }}>
                        <CardContent>
                            <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                            <Typography variant="subtitle1" fontWeight={700}>{n.title || n.id}</Typography>
                            <IconButton size="small" onClick={() => toggleExpand(n.id)}>
                                <ExpandMoreIcon />
                            </IconButton>
                            </Box>
                            <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                            {n.content ? (n.content.length > 180 ? n.content.slice(0, 180) + "…" : n.content) : ""}
                            </Typography>
                            <Box sx={{ mt: 1 }}>
                            <Typography variant="caption" color="text.secondary">
                                Relevance: {n.relevance ?? "—"} • Likelihood: {n.likelihood ?? "—"}
                            </Typography>
                            </Box>
                        </CardContent>
                        <Collapse in={!!expanded[n.id]}>
                            <CardContent>
                            <Typography variant="body2">{n.content}</Typography>
                            {n.type === "felt_pain" && (
                                <>
                                <Typography variant="subtitle2" sx={{ mt: 2 }}>Triggers</Typography>
                                {getConnectedOfTypes(n.id, ["pain_trigger"]).map((t) => (
                                    <Box key={t.id} sx={{ borderLeft: "2px solid #eee", pl: 1, mb: 1 }}>
                                    <Typography variant="body2" fontWeight={700}>{t.title}</Typography>
                                    <Typography variant="caption">Relevance: {t.relevance ?? "—"} • Likelihood: {t.likelihood ?? "—"}</Typography>
                                    </Box>
                                ))}
                                <Typography variant="subtitle2" sx={{ mt: 1 }}>Metrics</Typography>
                                {getConnectedOfTypes(n.id, ["pain_metric"]).map((m) => (
                                    <Box key={m.id} sx={{ borderLeft: "2px solid #eee", pl: 1, mb: 1 }}>
                                    <Typography variant="body2" fontWeight={700}>{m.title}</Typography>
                                    <Typography variant="caption">Relevance: {m.relevance ?? "—"} • Likelihood: {m.likelihood ?? "—"}</Typography>
                                    </Box>
                                ))}
                                </>
                            )}
                            </CardContent>
                            <CardActions>
                            <Button size="small" onClick={() => { /* placeholder: drill into node */ }}>
                                Open
                            </Button>
                            </CardActions>
                        </Collapse>
                        </Card>
                    ))
                    ) : (
                    <Typography variant="body2" color="text.secondary">No items</Typography>
                    )}
                </Box>
                ))}
            </Box>

            <Box sx={{ display: "flex", gap: 2, mt: 2 }}>
                <Button
                variant="contained"
                onClick={() => setHopIndex((h) => Math.max(0, h - 1))}
                disabled={hopIndex === 0}
                >
                Prev
                </Button>
                <Button
                variant="contained"
                onClick={() => setHopIndex((h) => h + 1)}
                disabled={!graphNodes.length}
                >
                Next
                </Button>
                <Typography sx={{ alignSelf: "center", ml: 2 }}>Hop: {hopIndex}</Typography>
            </Box>
            </Box>
        </Box>
        </Box>
    </Grid>
  );
}