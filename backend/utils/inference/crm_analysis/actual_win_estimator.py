import pandas as pd
import numpy as np
import networkx as nx
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from typing import Dict, List, Tuple, Any
import re

from backend.utils.graph_base.network_graph import get_nodes_list_ids
from backend.utils.inference.crm_analysis.category_column_schema import CAT_COLS
from backend.utils.inference.crm_analysis.crm_to_graph_compare import build_canon_maps_from_graph, compare_crm_vs_graph
from backend.utils.inference.graph_reconciliations import get_graphwin_override_factory, run_reconciliation
from backend.utils.inference.rcs_generators.rcs_computations.graphwin_runtime import get_graphwin

#-----
# Helpers
#------

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip().lower()

def model_feature_specs(win_pipe) -> List[Tuple[str, List[str]]]:
    """
    Returns [(dimension_name, categories_list), ...] in the same order used by OneHotEncoder.
    """
    pre = win_pipe.named_steps.get("pre") or win_pipe.named_steps.get("preprocessor")
    cat = pre.named_transformers_["cat"]
    cat_cols = list(pre.transformers_[0][2])  # the columns list passed to 'cat'
    specs = []
    for col, cats in zip(cat_cols, cat.categories_):
        specs.append((col, list(cats)))
    return specs  # e.g., [("industry", ["Retail","SaaS",...]), ("geography", ["Europe","North America",...]), ...]

def map_features_to_node_ids(
    win_pipe,
    canon_maps: Dict[str, Dict[str, str]],
    alias_overrides: Dict[str, Dict[str, str]] | None = None
) -> List[Dict[str, Any]]:
    """
    Returns list of dicts:
      {"feature": "industry_Retail",
       "dimension": "industry",
       "raw_value": "Retail",
       "node_id": "attribute_value:...",
       "status": "ok" | "alias" | "oov",
       "note": "..."}
    """
    alias_overrides = alias_overrides or {}
    feats = []
    # Build quick normalized lookup per dim
    canon_norm = {dim: { _norm(k): v for k, v in m.items() } for dim, m in canon_maps.items()}

    # Also allow normalized keys of the *display names* in canon maps
    for dim, m in list(canon_maps.items()):
        for disp, nid in m.items():
            canon_norm[dim][_norm(disp)] = nid

    for dim, categories in model_feature_specs(win_pipe):
        for val in categories:
            feature_name = f"{dim}_{val}"
            val_norm = _norm(val)

            # 1) exact match on display name
            node_id = canon_maps.get(dim, {}).get(val)
            status = "ok"; note = ""

            # 2) normalized match
            if node_id is None:
                node_id = canon_norm.get(dim, {}).get(val_norm)

            # 3) alias override (e.g., Europe -> EMEA)
            if node_id is None and dim in alias_overrides and val in alias_overrides[dim]:
                alias_target = alias_overrides[dim][val]
                node_id = canon_maps.get(dim, {}).get(alias_target) or canon_norm.get(dim, {}).get(_norm(alias_target))
                status, note = ("alias", f"alias {val} -> {alias_target}")

            # 4) still missing? mark OOV (provisional later)
            if node_id is None:
                status, note = ("oov", "no canonical match; propose provisional")

            feats.append({
                "feature": feature_name,
                "dimension": dim,
                "raw_value": val,
                "node_id": node_id,
                "status": status,
                "note": note
            })
    return feats

def flatten_attributes(df: pd.DataFrame) -> pd.DataFrame:
    """Expand df['attributes'] (dicts) into proper columns and drop it."""
    if "attributes" in df.columns:
        attr_df = pd.json_normalize(df["attributes"])
        df = pd.concat([df.drop(columns=["attributes"]), attr_df], axis=1)
    return df

def ensure_cat_cols(df: pd.DataFrame, cat_cols: tuple[str, ...]) -> pd.DataFrame:
    """Make sure categorical columns exist and are strings."""
    df = df.copy()
    for c in cat_cols:
        if c not in df.columns:
            df[c] = "Unknown"
        df[c] = df[c].astype(str).fillna("Unknown")
    return df

# ------------------------------------------------------
# 1. Build feature matrix from CRM + Graph structure
# ------------------------------------------------------
def generate_win_regression(G: nx.DiGraph, accounts: list[dict]) -> pd.DataFrame:
    """
    Given a GetIntentional graph G and a CRM dataframe with columns:
      - account_id
      - attributes (comma-separated attribute_value IDs)
      - deal_status (Closed_Won / Closed_Lost)
    """
    # 2. Prepare CRM DataFrame for regression
    crm_data = []
    for acc in accounts:
        # Compose attributes as a comma-separated string of attribute_value node ids
        attrs = {
            "industry": None,
            "revenue_range": None,
            "employee_range": None,
            "funding_stage": None,
            "geography": None,
            
        }
        if acc.get("industry"):
            attrs["industry"] = acc["industry"]
        if acc.get("revenue_range"):
            attrs["revenue_range"] = acc["revenue_range"]
        if acc.get("employee_range"):
            attrs["employee_range"] = acc["employee_range"]
        if acc.get("funding_stage"):
            attrs["funding_stage"] = acc["funding_stage"]
        if acc.get("geography"):
            attrs["geography"] = acc["geography"]
        
        # Add other attributes as needed

        crm_data.append({
            "account_id": acc["id"],
            "attributes": attrs,
            "deal_status": acc.get("deal_status", "Closed_Lost"),
            "entered_pipeline": 1 if acc.get("deal_status") in ["Closed-Won", "Closed-Lost"] else 0,
            "won": 1 if acc.get("deal_status") == "Closed-Won" else 0
        })

    crm_df = pd.DataFrame(crm_data)
    # Flatten the 'attributes' dict into columns
    
    canon_maps = build_canon_maps_from_graph(G)

    # Enforce canonical labels on CRM BEFORE fitting
    col_to_dim = {
        "industry": "industry",
        "revenue_range": "revenue_range",
        "employee_range": "employee_range",
        "geography": "geography",
        "funding_stage": "funding_stage",
        # keep competitor/tech if you later add nodes for them; for now they can stay as-is
    }
    crm_df = enforce_canonical_labels(crm_df, canon_maps, col_to_dim)

    # ✅ FLATTEN CRM DF 
    crm_df = flatten_attributes(crm_df)

    # Survivorship step input must have concrete cat columns
    crm_df = ensure_cat_cols(crm_df, CAT_COLS)

    # Fit selection/win models on canonicalized labels
    if len(set(crm_df['entered_pipeline'])) < 2:
        crm_df["_p_enter"] = 1.0
        crm_df["_ipw"] = 1.0
        sel_pipe = None
        df_aug = crm_df
    else:
        sel_pipe, df_aug = fit_selection_model(crm_df)


    win_pipe, features_coeffs = fit_win_model(df_aug)

    # Standardize for compare (no aliasing needed now)
    df_for_compare = df_aug.copy()
    df_for_compare["won"] = df_for_compare.get("won", (df_for_compare["deal_status"]=="Closed-Won")).astype(int)
    if "account_id" not in df_for_compare.columns:
        df_for_compare["account_id"] = crm_df["account_id"]
    for c in CAT_COLS:
        if c in df_for_compare.columns:
            df_for_compare[c] = df_for_compare[c].fillna("Unknown").astype(str)
    # 🔑 Compare (no manual alias_overrides)
    out = compare_crm_vs_graph(
        product_graph=G,
        df_accounts_aug=df_for_compare,
        win_pipe=win_pipe,
        get_graphwin=get_graphwin,
        canon_maps=canon_maps,

    )
    df_align = out["df_align"]
    override_eval = get_graphwin_override_factory(G, get_graphwin)
    recon = run_reconciliation(
        G=G,
        df_align=df_align,  # from compare_crm_vs_graph(...)
        get_graphwin_override=override_eval,
        eps=1e-2, lr=0.5, min_support=3, max_depth=5
    )
    proposals = recon["proposals"]

    print("Under-predicted:\n", out["archetype_summaries"]["under_predicted"].head(10))
    print("Over-predicted:\n", out["archetype_summaries"]["over_predicted"].head(10))

    return {
        "features_and_coefficients": features_coeffs,
        "intercept": win_pipe.named_steps["clf"].intercept_.tolist(),
        "df_augmented": df_aug.to_dict(orient="records"),
        # NEW:
        "alignment": out["metrics"],
        "df_align": out["df_align"].to_dict(orient="records"),
        "proposals": proposals,   # your layer-wise MRCA output
        }


# --- add this helper ---
import re

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip().lower()

def enforce_canonical_labels(df: pd.DataFrame,
                             canon_maps: dict[str, dict[str,str]],
                             col_to_dim: dict[str,str]) -> pd.DataFrame:
    """
    For each CRM column (industry, revenue_range, ...), replace values with the
    exact display keys present in canon_maps[dimension]. If a value can't be
    matched (even after normalization), raise with details.
    """
    out = df.copy()
    for col, dim in col_to_dim.items():
        if col not in out.columns:
            continue
        # Build normalized reverse index from canon display -> node_id
        canon_display_keys = list(canon_maps.get(dim, {}).keys())
        norm_to_display = { _norm(k): k for k in canon_display_keys }

        bad = []
        mapped = []
        for val in out[col].astype(str).fillna("Unknown"):
            key = norm_to_display.get(_norm(val))
            if key is None:
                bad.append(val)
                mapped.append(None)
            else:
                mapped.append(key)
        if bad:
            uniq = sorted(set(bad))
            raise ValueError(
                f"Non-canonical values in '{col}' for dimension '{dim}': {uniq}. "
                f"Allowed: {sorted(canon_display_keys)}"
            )
        out[col] = mapped
    return out



# -------------------------------
# 1) Build selection (entry) model
# -------------------------------
def fit_selection_model(df: pd.DataFrame,
                        cat_cols=CAT_COLS,
                        entry_col='entered_pipeline'):
    """
    df columns expected:
      - Actual attribute category cols: Industry, Revenue Range, Employee Range, Geography, Funding Stage, Competitor Used, Other Tech Stack
      - entered_pipeline : 0/1 (observed in CRM or qualified)
    """
    X = df[list(cat_cols)]
    y = df[entry_col].astype(int).values

    pre = ColumnTransformer(
        transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), list(cat_cols))],
        remainder='drop'
    )

    pipe = Pipeline(steps=[
        ('pre', pre),
        ('imp', SimpleImputer(strategy='most_frequent')),  # safety
        ('clf', LogisticRegression(max_iter=200, solver='lbfgs'))
    ])
    pipe.fit(X, y)
    p_enter = pipe.predict_proba(X)[:, 1].clip(1e-6, 1-1e-6)
    ipw = 1.0 / p_enter
    df = df.copy()
    df['_p_enter'] = p_enter
    df['_ipw'] = ipw
    return pipe, df

# ---------------------------------
# 2) Build weighted win probability
# ---------------------------------
def fit_win_model(df: pd.DataFrame,
                  cat_cols=CAT_COLS,
                  win_col='won',
                  sample_weight_col='_ipw'):
    """
    df columns expected:
      - Actual attribute category cols: Industry, Revenue Range, Employee Range, Geography, Funding Stage, Competitor Used, Other Tech Stack
      - won : 0/1 (closed_won vs closed_lost)
      - _ipw : inverse-probability weights from selection model
    """
    X = df[list(cat_cols)]
    y = df[win_col].astype(int).values
    w = df[sample_weight_col].values

    pre = ColumnTransformer(
        transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), list(cat_cols))],
        remainder='drop'
    )
    pipe = Pipeline(steps=[
        ('pre', pre),
        ('imp', SimpleImputer(strategy='most_frequent')),
        ('clf', LogisticRegression(max_iter=500, solver='lbfgs'))
    ])
    
    pipe.fit(X, y, clf__sample_weight=w)

    feature_names = (
        pipe.named_steps["pre"]
        .named_transformers_["cat"]
        .get_feature_names_out(list(cat_cols))
    )

    # Merge feature names and coefficients
    coef = pipe.named_steps["clf"].coef_[0]  # shape: (n_features,)
    features_and_coefs = [
        {"feature": fname, "coefficient": float(c)}
        for fname, c in zip(feature_names, coef)
    ]

    return pipe, features_and_coefs



