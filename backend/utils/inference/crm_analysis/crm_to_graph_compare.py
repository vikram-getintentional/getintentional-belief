# compare_crm_vs_graph.py
from __future__ import annotations
import math
from typing import Dict, Callable, List, Any
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, brier_score_loss
import networkx as nx

from backend.utils.graph_base.network_graph import get_nodes_list_ids

# ==============
# Helper utils
# ==============
def _clip01(x, eps=1e-6): 
    return float(min(max(x, eps), 1.0 - eps))


def build_canon_maps_from_graph(G: nx.DiGraph):
    """
    Returns
    -------
    canon_maps : dict
        {
          "industry": {
             "Financial Services": "attribute_value:90e04aa197ac0670",
             "financial services": "attribute_value:90e04aa197ac0670",   # normalized alias
             "finserv":            "attribute_value:90e04aa197ac0670"    # if alias_overrides provided
          },
          "geo": { ... },
          ...
        }
    
    """
    canon_maps = {}
    all_attribute_node_ids = get_nodes_list_ids(G, "attribute_value", {})
    for attr in all_attribute_node_ids:
        attr_dimension = G.nodes[attr].get("dimension")
        if attr_dimension not in canon_maps:
            canon_maps[attr_dimension] = {}
        canon_maps[attr_dimension][G.nodes[attr]["name"]] = attr
    return canon_maps

def canonicalize_attrs(row: pd.Series, canon: Dict[str, Dict[str,str]]) -> Dict[str,str]:
    """
    Map raw CRM categories to graph's canonical node IDs.
    If a value is OOV, we leave it as-is (and mark it), so you can decide later.
    """
    def _canon(kind, val):
        return canon.get(kind, {}).get(val, val)  # leave raw if not found (OOV)
    engaged_nodes = []
    for kind in ["industry", "revenue_range", "employee_range", "geography", "funding_stage"]:
        node_id = _canon(kind, row.get(kind, "Unknown"))
        if isinstance(node_id, str) and node_id:  # skip empty
            engaged_nodes.append({"id": node_id, "occurrence": 1.0})
        else:
            print(f"Warning: unexpected non-str node_id for {kind}: {node_id}")
    return engaged_nodes

def _oov_flags(engaged_nodes: list) -> Dict[str, bool]:
    dims = ["industry", "revenue_range", "employee_range", "geography", "funding_stage"]
    oov_flags = {}
    for i, dim in enumerate(dims):
        node_id = engaged_nodes[i]["id"] if i < len(engaged_nodes) else ""
        oov_flags[dim] = (":" not in node_id)
    return oov_flags


# ============================================
# 1) Build aligned comparison dataframe
# ============================================
def build_aligned_df(
    df_accounts_aug: pd.DataFrame,
    win_pipe,  # your trained survivorship-corrected logistic model
    get_graphwin: Callable[[dict, List[str] | None], float],
    canon_maps: Dict[str, Dict[str, str]],
    product_graph: nx.Graph,
) -> pd.DataFrame:
    """
    Returns a tidy DataFrame with p_crm_win, p_graph_win, residuals, and OOV flags.
    Expected df_accounts_aug columns: account_id, Ai/Ar/Ae/Ag (or mappable), won (0/1 optional), _ipw (optional)
    """
    #df = _ensure_attr_columns(df_accounts_aug)
    df = df_accounts_aug
    # Canonicalize for the graph
    df["graph_attrs"] = df.apply(lambda r: canonicalize_attrs(r, canon_maps), axis=1)
    DIM_ORDER = ["industry","revenue_range","employee_range","geography","funding_stage"]

    def _to_id(dim: str, val: str) -> str:
        # if no match, keep val (and OOV flags will catch it)
        return canon_maps.get(dim, {}).get(val, val)

    for dim in DIM_ORDER:
        col = f"{dim}_id"
        df[col] = df[dim].astype(str).map(lambda v: _to_id(dim, v))

    # Mark OOV (any attribute not in canonical map will not contain ':')
    oov_cols = df["graph_attrs"].apply(_oov_flags)
    df["oov_industry"] = oov_cols.apply(lambda d: d.get("industry", False))
    df["oov_revenue_range"]  = oov_cols.apply(lambda d: d.get("revenue_range", False))
    df["oov_employee_range"] = oov_cols.apply(lambda d: d.get("employee_range", False))
    df["oov_geography"]      = oov_cols.apply(lambda d: d.get("geography", False))
    df["oov_funding_stage"]  = oov_cols.apply(lambda d: d.get("funding_stage", False))
    df["oov_any"] = df[["oov_industry","oov_revenue_range","oov_employee_range","oov_geography","oov_funding_stage"]].any(axis=1)

    # CRM model probability
    df["p_crm_win"] = win_pipe.predict_proba(df[["industry","revenue_range","employee_range","geography","funding_stage"]])[:, 1].astype(float)
    
    # Graph probability
    try:
        df["p_graph_win"] = df["graph_attrs"].apply(
            lambda engaged_nodes: _clip01(get_graphwin(product_graph, engaged_nodes)["win_likelihood"])
        )
    except Exception as e:
        print("Error occurred while computing graph win probabilities:", e)
        df["p_graph_win"] = np.nan
    # Residual (CRM minus Graph)
    df["residual"] = df["p_crm_win"] - df["p_graph_win"]

    # Keep a compact view
    cols = [
        "account_id", *DIM_ORDER, *[f"{d}_id" for d in DIM_ORDER], "won",
        "p_crm_win","p_graph_win","residual",
        "oov_industry","oov_revenue_range","oov_employee_range","oov_geography","oov_funding_stage","oov_any",
        "graph_attrs",
    ]
    keep = [c for c in cols if c in df.columns]
    return df[keep].copy()

# ============================================
# 2) Headline metrics & summaries
# ============================================
def evaluate_alignment(df_align: pd.DataFrame) -> Dict[str, float]:
    """
    Returns:
      - MAE_models: MAE between p_crm_win and p_graph_win
      - R2_models : R^2 between the two model outputs
      - Brier_graph / Brier_crm: optional, if 'won' labels exist (drops NaNs)
    """
    d = df_align.dropna(subset=["p_crm_win","p_graph_win"]).copy()
    out = {}
    out["MAE_models"] = float(mean_absolute_error(d["p_crm_win"], d["p_graph_win"]))
    out["R2_models"]  = float(r2_score(d["p_crm_win"], d["p_graph_win"])) if len(d) > 1 else float("nan")
    if "won" in d.columns and d["won"].notna().any():
        d2 = d.dropna(subset=["won"]).copy()
        d2["won"] = d2["won"].astype(int)
        try:
            out["Brier_graph"] = float(brier_score_loss(d2["won"], d2["p_graph_win"]))
            out["Brier_crm"]   = float(brier_score_loss(d2["won"], d2["p_crm_win"]))
        except Exception:
            out["Brier_graph"] = float("nan")
            out["Brier_crm"]   = float("nan")
    return out

def summarize_by_archetype(df_align: pd.DataFrame, top_n: int = 15) -> Dict[str, pd.DataFrame]:
    """
    Aggregates by (Ai,Ar,Ae,Ag):
      - count
      - mean p_crm_win, mean p_graph_win, mean residual
    Returns top under- and over-predicted archetypes.
    """
    d = df_align.dropna(subset=["p_crm_win","p_graph_win"]).copy()
    grp = (d.groupby(["industry","revenue_range","employee_range","geography", "funding_stage"], dropna=False)
             .agg(count=("account_id","count"),
                  p_crm=("p_crm_win","mean"),
                  p_graph=("p_graph_win","mean"),
                  resid=("residual","mean"))
             .reset_index()
             .sort_values("resid"))
    # Biggest under-predictions (graph < CRM)
    under = grp.head(top_n)
    # Biggest over-predictions (graph > CRM)
    over  = grp.tail(top_n).iloc[::-1]
    return {"under_predicted": under, "over_predicted": over}

# ============================================
# 3) Convenience "runner" for this step
# ============================================
def compare_crm_vs_graph(
    product_graph: nx.Graph,
    df_accounts_aug: pd.DataFrame,
    win_pipe,
    get_graphwin: Callable[[dict, List[str] | None], float],
    canon_maps: Dict[str, Dict[str, str]],
) -> Dict[str, Any]:
    """
    Main entrypoint for Step 1.
    Returns:
      - df_align: row-level comparison
      - metrics: headline alignment metrics
      - archetype_summaries: top under/over predicted buckets
      - oov_rows: rows with any OOV attribute (good to inspect before calibration)
    """
    df_align = build_aligned_df(df_accounts_aug, win_pipe, get_graphwin, canon_maps, product_graph=product_graph)
    metrics  = evaluate_alignment(df_align)
    arch     = summarize_by_archetype(df_align)

    oov_rows = df_align[df_align["oov_any"] == True].copy()

    return {
        "df_align": df_align,                 # account_id, attrs, p_crm_win, p_graph_win, residual
        "metrics": metrics,                   # MAE/R² (+ Brier if labels present)
        "archetype_summaries": arch,          # biggest gaps by Ai×Ar×Ae×Ag
        "oov_rows": oov_rows                  # to drive OOV/provisional-node step next
    }
