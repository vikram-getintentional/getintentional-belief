from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Body, Depends, Query, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from backend.utils.crm_management.target_account_manager import get_account_by_id, get_target_account_ids, load_target_accounts_from_db, save_and_update_target_accounts, delete_target_account_handler
from backend.utils.graph_base.graph_utils.graph_confidence import compute_graph_confidence
from backend.utils.graph_base.network_graph import add_capabilities_to_product, build_product_graph, get_product_id_from_subgraph, update_capabilities_by_nodes_list
from backend.utils.graph_base.nodes.capability_nodes import update_capabilities_by_node_id
from backend.utils.inference.crm_analysis.actual_win_estimator import generate_win_regression
from backend.utils.inference.discovery_engine.agentic_engine.agentic_loop import run_agentic_loop, run_frontier_expansion

from backend.auth.jwt_handler import decode_token
from sqlalchemy.orm import Session
from backend.database import get_db
from backend.utils.inference.rcs_generators.rcs_simulator import simulate_rcs
from backend.utils.knowledge_base.arsenal_generation import get_all_arsenals
from backend.utils.knowledge_base.graph_edit_manager import add_or_update_graph, serialize_graph_for_frontend
from backend.utils.knowledge_base.value_prop_analysis import generate_product_value_prop, get_product_id_from_company_id, get_product_value_prop_capabilities

from backend.utils.knowledge_base.persona_generation import (
    get_personas_rcs_priority
)
from backend.utils.knowledge_base.zmot_icp_generation import _collect_attribute_chips, get_attribute_options
import networkx as nx

# Hardcoded path to graph data folder - to be updated in production

import os

from backend.utils.knowledge_base.zmot_icp_generation import Chip, collect_zmots_for_attribute_combo, mine_icp_attribute_uplifts
from backend.utils.strategy_builder.comprehensive_plan_generator import _filter_by_window, build_integrated_portfolio_plan

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GRAPH_DATA_PATH = os.path.join(BASE_DIR,"backend", "utils", "graph_base", "graph_data")

router = APIRouter()

# ========================
# PRODUCT GRAPH ROUTES
# ========================

@router.get("/company/{company_id}/product-subgraph")
def get_product_subgraph(company_id: str):
    product_id = get_product_id_from_company_id(company_id)
    if not product_id:
        return {"exists": False}

    product_subgraph = build_product_graph(product_id)
    product_data = get_product_value_prop_capabilities(product_subgraph)
    if not product_data or not product_data.get("capabilities"):
        return {"exists": False, "capabilities": []}

    return {
        "exists": True,
        "capabilities": product_data["capabilities"],
        "summary": product_data.get("summary", ""),
        "product_node_id": product_data.get("product_node_id", ""),
        "plg_flag": product_data.get("plg_flag", False)
    }
# ========================
# 🔎 Basic scraper-based analysis
# ========================
@router.post("/analyze")
def analyze(payload: dict, request: Request, db: Session = Depends(get_db)):
    url = payload.get("url")
    text = payload.get("text")
    plg_cta = payload.get("plg_cta_found", False)
    footer_features = payload.get("footer_features", [])

    # 🔐 Get company_id from token
    auth_header = request.headers.get("authorization")
    if not auth_header:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    token = auth_header.split(" ")[1]
    decoded = decode_token(token)
    company_id = decoded.get("company_id")
    if not company_id:
        raise HTTPException(status_code=401, detail="Invalid token or company ID not found")

    if not url:
        return {"error": "URL required."}


    # Call the core logic function
    try:
        result = generate_product_value_prop(company_id, url, text, plg_cta, footer_features)
        print("Dict output summary:", result["summary"]," \n Dict output capabilitites: ", result["capabilities"])
        return result
    except Exception as e:
        print("❌ Analyze error:", e)
        raise HTTPException(status_code=500, detail="Could not run analysis")

@router.post("/update-capabilities")
def update_capabilities_route(payload: dict, request: Request, db: Depends = None):
    """
    Updates existing capabilities by node_id.
    Expects payload with:
    - company_id: str
    - capabilities: list of {node_id, name, description}
    """
    try:
        # 🔐 Auth
        auth_header = request.headers.get("authorization")
        if not auth_header:
            raise HTTPException(status_code=401, detail="Missing Authorization header")
        token = auth_header.split(" ")[1]
        decoded = decode_token(token)
        company_id = decoded.get("company_id")
        if not company_id:
            raise HTTPException(status_code=401, detail="Invalid token or company ID not found")

        capabilities = payload.get("capabilities", [])
        if not capabilities:
            raise HTTPException(status_code=400, detail="Capabilities are required.")

        product_id = payload.get("product_id")
        if not product_id:
            raise HTTPException(status_code=400, detail="Product ID is required.")
        updated_nodes = update_capabilities_by_node_id(capabilities)
        return {"message": "Capabilities updated successfully.", "capabilities": [n["id"] for n in updated_nodes]}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print("❌ Update capabilities error:", e)
        raise HTTPException(status_code=500, detail="Could not update capabilities")


@router.post("/update-capabilities")
def update_capabilities_route(payload: dict, request: Request, db: Depends = None):
    """
    Updates existing capabilities by node_id.
    Expects payload with:
    - company_id: str
    - capabilities: list of {node_id, name, description}
    """
    try:
        # 🔐 Auth
        auth_header = request.headers.get("authorization")
        if not auth_header:
            raise HTTPException(status_code=401, detail="Missing Authorization header")
        token = auth_header.split(" ")[1]
        decoded = decode_token(token)
        company_id = decoded.get("company_id")
        if not company_id:
            raise HTTPException(status_code=401, detail="Invalid token or company ID not found")

        capabilities = payload.get("capabilities", [])
        print("Capabilities as recd by API:", capabilities)
        if not capabilities:
            raise HTTPException(status_code=400, detail="Capabilities are required.")

        product_id = payload.get("product_id")
        if not product_id:
            raise HTTPException(status_code=400, detail="Product ID is required.")

        product_subgraph = build_product_graph(product_id)
        cap_list_to_process = []
        for cap in capabilities:
            cap_id = cap.get("id")
            updated_cap_name = cap.get("name", "")
            updated_cap_description = cap.get("description", "")
            updated_capability = { "id": cap_id, "name": updated_cap_name, "description": updated_cap_description }
            cap_list_to_process.append(updated_capability)
        print("Capabilities to process:", cap_list_to_process)
        updated_nodes = update_capabilities_by_nodes_list(product_subgraph,cap_list_to_process)
        return {"message": "Capabilities updated successfully.", "capabilities": [n["id"] for n in updated_nodes]}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print("❌ Update capabilities error:", e)
        raise HTTPException(status_code=500, detail="Could not update capabilities")


@router.post("/add-capabilities")
def add_capabilities_route(payload: dict, request: Request, db: Depends = None):
    """
    Adds new capabilities to a product.
    Expects payload with:
    - url: str
    - capabilities: list of {name, description}
    """
    try:
        # 🔐 Auth
        auth_header = request.headers.get("authorization")
        if not auth_header:
            raise HTTPException(status_code=401, detail="Missing Authorization header")
        token = auth_header.split(" ")[1]
        decoded = decode_token(token)
        company_id = decoded.get("company_id")
        if not company_id:
            raise HTTPException(status_code=401, detail="Invalid token or company ID not found")

        product_id = payload.get("product_id")
        capabilities = payload.get("capabilities", [])
        if not product_id or not capabilities:
            raise HTTPException(status_code=400, detail="Product ID and capabilities are required.")

        product_subgraph = build_product_graph(product_id)

        added_capabilities = add_capabilities_to_product(product_subgraph, capabilities)
        return {
            "message": "New capabilities added successfully.",
            "capabilities": [
                {
                    "node_id": c["id"],
                    "name": c.get("name", ""),
                    "description": c.get("description", "")
                }
                for c in added_capabilities
            ]
        }
    except Exception as e:
        print("❌ Add capabilities error:", e)
        raise HTTPException(status_code=500, detail="Could not add capabilities")


@router.post("/save-summary")
def save_summary(payload: dict, request: Request, db: Depends = None):
    """
    Saves or updates the summary for a product.
    Expects payload with:
    - company_id: str
    - url: str
    - summary: str
    """
    try:
        # 🔐 Auth
        auth_header = request.headers.get("authorization")
        if not auth_header:
            raise HTTPException(status_code=401, detail="Missing Authorization header")
        token = auth_header.split(" ")[1]
        decoded = decode_token(token)
        company_id = decoded.get("company_id")
        if not company_id:
            raise HTTPException(status_code=401, detail="Invalid token or company ID not found")

        product_id = payload.get("product_id")
        summary = payload.get("summary", "").strip()
        if not product_id or not summary:
            raise HTTPException(status_code=400, detail="Product ID and summary are required.")

        product_node = get_or_create_product_node(
            summary=summary,
            company_id=company_id,
            product_id=product_id
        )
        return {"message": "Summary updated successfully.", "summary": summary}
    except Exception as e:
        print("❌ Save summary error:", e)
        raise HTTPException(status_code=500, detail="Could not save summary")

# ========================
# EPISTEMATIC CONFIDENCE
# ========================

@router.get("/graph/confidence/{product_id}")
def get_confidence(product_id: str, request: Request):
    try:
        auth_header = request.headers.get("authorization")
        if not auth_header:
            raise HTTPException(status_code=401, detail="Missing Authorization header")
        token = auth_header.split(" ")[1]
        decoded = decode_token(token)
        company_id = decoded.get("company_id")
        if not company_id:
            raise HTTPException(status_code=401, detail="Invalid token or company ID not found")
        product_subgraph = build_product_graph(product_id)
        if not product_subgraph:
            raise HTTPException(status_code=404, detail="Product subgraph not found")
        report = compute_graph_confidence(product_subgraph)

        return report.__dict__
    except Exception as e:
        print("❌ Get confidence error:", e)
        raise HTTPException(status_code=500, detail="Could not retrieve confidence report")

@router.post("/graph/confidence")
def post_confidence(request: Request):
    return get_confidence(request.product_id)

# ========================
# DEEP INFERENCE
# ========================

@router.post("/analyze/deep")
async def analyze_deep(payload: dict = Body(..., embed=False), request: Request = None):
    """
    Run agentic analysis for a product. Expects JSON body: { "product_id": "<id>" }.
    """
    product_id = (payload or {}).get("product_id")
    print("Deep analyzing product_id:", product_id)
    if not product_id:
        raise HTTPException(status_code=400, detail="Product ID required in body as {\"product_id\": \"...\"}")

    # 🔐 Auth
    auth_header = request.headers.get("authorization") or request.headers.get("Authorization")
    if not auth_header or " " not in auth_header:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    scheme, token = auth_header.split(" ", 1)
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(status_code=401, detail="Invalid Authorization header")
    decoded = decode_token(token)
    company_id = decoded.get("company_id")
    if not company_id:
        raise HTTPException(status_code=401, detail="Invalid token or company ID not found")

    # Build product graph and validate
    product_subgraph = build_product_graph(product_id)
    if not product_subgraph:
        raise HTTPException(status_code=404, detail=f"Product subgraph not found for product_id: {product_id}")
    print("Product subgraph built successfully for product_id:", product_id,". Starting agent")
    # Run the agentic loop (update_graph inside that flow should now receive a valid subgraph)
    try:
        run_agentic_loop(product_subgraph)
    except Exception as e:
        print("❌ Agentic analysis error:", e)
        raise HTTPException(status_code=500, detail="Agentic analysis failed")

    return {"status": "Agentic analysis complete"}

@router.post("/graph/expand-frontier/{product_id}")
async def expand_frontier(product_id: str, request: Request, body: dict = Body(..., embed=False)):
    print("Expanding frontier for product_id:", product_id)
    auth_header = request.headers.get("authorization") or request.headers.get("Authorization")
    if not auth_header:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    token = auth_header.split(" ")[1]
    decoded = decode_token(token)
    if not decoded.get("company_id"):
        raise HTTPException(status_code=401, detail="Invalid token")

    seed_ids = body.get("seed_ids") or []
    only_types = body.get("only_types")
    waves = int(body.get("waves", 1))
    max_items = int(body.get("max_items_per_source", 5))
    print("Seed IDs:", seed_ids)

    G = build_product_graph(product_id)

    # Optional: auto-seed terminals if caller sent no seeds
    if not seed_ids:
        # terminal = out_degree 0, restrict by type if requested
        allow = set(only_types) if only_types else None
        term = []
        for n, data in G.nodes(data=True):
            if G.out_degree(n) == 0 and (allow is None or data.get("type") in allow):
                term.append(n)
        # keep bounded
        seed_ids = term[: max_items * (len(allow) if allow else 3) or 5]
        print("[Frontier] Auto-seeded terminals:", seed_ids)

   

    result = run_frontier_expansion(
        G,
        seed_ids,
        only_types=only_types,
        waves=waves,
        max_items_per_source=max_items,
        model="gpt-4o-mini",
        ctx=None,
    )
    return {"ok": True, **(result or {})}

# ========================    
# GET-Products to get a list of Product Nodes for a company_id
# ========================
# GET-Products to get a list of Product Nodes for a company_id
@router.get("/get-products/{company_id}")
async def get_products(company_id: str, request: Request):
    """
    Returns a list of products for the given company_id.
    """
    try:
        # 🔐 Auth
        auth_header = request.headers.get("authorization")
        if not auth_header:
            raise HTTPException(status_code=401, detail="Missing Authorization header")
        token = auth_header.split(" ")[1]
        decoded = decode_token(token)
        company_id = decoded.get("company_id")

        if not company_id:
            raise HTTPException(status_code=401, detail="Invalid token or company ID not found")

        # 🧠 Inference
        product_lookup = get_product_id_from_company_id(company_id)
        if product_lookup:
            print("Product ID found for company_id:", company_id)
            product_subgraph = build_product_graph(product_lookup)
            print("Product subgraph data loaded")
            product_id = get_product_id_from_subgraph(product_subgraph)
            
            product_node = product_subgraph.nodes[product_id]
            products = [{
                "id": product_lookup,
                "domain": product_node.get("domain", ""),
                "industry": product_node.get("industry", ""),
                "summary": product_node.get("summary", ""),
                "url": product_node.get("url", ""),
            }]
        else: products = []
        
        print("Products found:", products)
        return {"products": products}
    except Exception as e:
        print("❌ Get products error:", e)
        raise HTTPException(status_code=500, detail="Could not retrieve products")

# GET /get-product-capabilities/{product_id}
@router.get("/get-product-capabilities/{product_id}")
async def get_product_capabilities(product_id: str, request: Request):
    print("Getting product capabilities for product_id:", product_id)
    """
    Returns a list of capabilities for the given product_id.
    """
    try:
        # 🔐 Auth
        auth_header = request.headers.get("authorization")
        if not auth_header:
            raise HTTPException(status_code=401, detail="Missing Authorization header")

        token = auth_header.split(" ")[1]
        decoded = decode_token(token)
        company_id = decoded.get("company_id")

        if not company_id:
            raise HTTPException(status_code=401, detail="Invalid token or company ID not found")

        # 🧠 Inference
        product_subgraph = build_product_graph(product_id)
        
        product_summary = get_product_value_prop_capabilities(product_subgraph)

        return product_summary
    except Exception as e:
        print("❌ Get capabilities error:", e)
        raise HTTPException(status_code=500, detail="Could not retrieve capabilities")


#--------------------------
# GRAPH EDITOR & REVIEW
#--------------------------
@router.get("/export-product-graph/{product_id}")
async def export_product_graph(product_id: str, request: Request):
    auth_header = request.headers.get("authorization")
    if not auth_header:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    token = auth_header.split(" ")[1]
    decoded = decode_token(token)
    company_id = decoded.get("company_id")
    if not company_id:
        raise HTTPException(status_code=401, detail="Invalid token or company ID not found")

    try:
        product_subgraph = build_product_graph(product_id)
        payload = serialize_graph_for_frontend(product_subgraph)
        return payload
    except Exception as e:
        print("❌ export-product-graph error:", e)
        raise HTTPException(status_code=500, detail="Could not export product graph")
    
@router.post("/graph/bulk-upsert-graph/{product_id}")
async def bulk_upsert_graph(product_id: str, graph_payload: dict, request: Request):
    """
    Bulk upsert nodes and edges into the product graph.
    Expects payload with:
    - product_id: str
    - nodes: list of node dicts
    - edges: list of edge dicts
    """

    # 🔐 Auth
    auth_header = request.headers.get("authorization")
    if not auth_header:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    token = auth_header.split(" ")[1]
    decoded = decode_token(token)
    company_id = decoded.get("company_id")
    if not company_id:
        raise HTTPException(status_code=401, detail="Invalid token or company ID not found")
    if not product_id:
        raise HTTPException(status_code=400, detail="Product ID is required.")
    
    product_graph = build_product_graph(product_id)
    nodes = graph_payload.get("nodes", [])
    edges = graph_payload.get("edges", [])
    
    try:
        add_or_update_graph(product_graph, nodes, edges)
        print("Graph updated successfully with bulk upsert.")
        return {"message": "Graph updated successfully."}
    except Exception as e:
        print("❌ Error in add_or_update_graph:", e)
        raise HTTPException(status_code=500, detail="Error updating graph data")




# ========================
# PERSONAS
# ========================

@router.get("/get-personas/{product_id}")
async def get_personas(product_id: str, request: Request):
    auth_header = request.headers.get("authorization")
    if not auth_header:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    token = auth_header.split(" ")[1]
    decoded = decode_token(token)
    company_id = decoded.get("company_id")
    if not company_id:
        raise HTTPException(status_code=401, detail="Invalid token or company ID not found")

    product_subgraph = build_product_graph(product_id)
    aggregated_personas = get_personas_rcs_priority(product_subgraph)
    return aggregated_personas


# GET /get-zmot-icp/{product_id}
@router.get("/get-zmot-icp/{product_id}")
async def get_zmot_icp(product_id: str, request: Request, max_len: int = 3, top_k: int = 10):
    """
    Returns graph-driven ICP attribute bundles with win rates and,
    for each combo, the relevant ZMOT events + observable moments + keywords.
    """
    try:
        # 🔐 Auth
        auth_header = request.headers.get("Authorization") or request.headers.get("authorization")
        if not auth_header or " " not in auth_header:
            raise HTTPException(status_code=401, detail="Missing Authorization header")
        scheme, token = auth_header.split(" ", 1)
        if scheme.lower() != "bearer" or not token:
            raise HTTPException(status_code=401, detail="Invalid Authorization header")

        decoded = decode_token(token)
        company_id = decoded.get("company_id")
        if not company_id:
            raise HTTPException(status_code=401, detail="Invalid token or company ID not found")

        # 🧠 Build the product graph
        # (adjust signature if your builder needs company_id)
        product_subgraph: nx.DiGraph = build_product_graph(product_id)

        # 1) Mine attribute uplifts from the graph
        icp_payload = mine_icp_attribute_uplifts(
            product_subgraph,
            max_combo_len=max_len,
            top_k=top_k
        )
        if "error" in icp_payload:
            raise HTTPException(status_code=500, detail=icp_payload["error"])

        # 2) For each combo, attach ZMOTs/moments/keywords
        combos = icp_payload.get("combos", [])
        enriched_combos = []
        for combo_rec in combos:
            # combo_rec["chips"] = [{family,node_id,label}, ...]
            # Convert to Chip dataclasses expected by collector
            chips = [
                Chip(family=c["family"], node_id=c["node_id"], label=c.get("label") or c["node_id"])
                for c in (combo_rec.get("chips") or [])
            ]
            zmot_pack = collect_zmots_for_attribute_combo(
                product_subgraph,
                chips,
                top_k_zmots=15,
                top_k_moments=8,
                top_k_keywords=12
            )
            enriched_combos.append({
                **combo_rec,
                "zmots": zmot_pack
            })

        # 3) Return baseline + enriched combos (keep fingerprint if present)
        return {
            "graph_fingerprint": icp_payload.get("graph_fingerprint"),
            "baseline": icp_payload.get("baseline"),
            "combos": enriched_combos
        }

    except HTTPException:
        raise
    except Exception as e:
        print("❌ Get ZMOT ICP error:", e)
        raise HTTPException(status_code=500, detail="Could not retrieve ZMOT ICP")

# ========================
# ICP ATTRIBUTE MINING
# ========================

@router.get("/get-icp-attributes/{product_id}")
async def get_icp_attributes(product_id: str, request: Request, max_len: int = 3, top_k: int = 10):
    """
    Returns ICP attribute bundles (industry, revenue, employees, funding, geography)
    with win rates + relevant ZMOTs.
    """
    print("Fetching ICP Options")
    auth_header = request.headers.get("authorization")
    if not auth_header:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    token = auth_header.split(" ")[1]
    decoded = decode_token(token)
    company_id = decoded.get("company_id")
    if not company_id:
        raise HTTPException(status_code=401, detail="Invalid token or company ID not found")

    product_subgraph = build_product_graph(product_id)
    icp_payload = mine_icp_attribute_uplifts(product_subgraph, max_combo_len=max_len, top_k=top_k)

    # Get all available attributes for this product
    attribute_chips = _collect_attribute_chips(product_subgraph)
    # attribute_chips: Dict[str, List[Chip]]

    enriched_combos = []
    for combo_rec in icp_payload.get("combos", []):
        chips = [
            Chip(family=c["family"], node_id=c["node_id"], label=c.get("label") or c["node_id"])
            for c in (combo_rec.get("chips") or [])
        ]
        zmot_pack = collect_zmots_for_attribute_combo(
            product_subgraph,
            chips,
            top_k_zmots=15,
            top_k_moments=8,
            top_k_keywords=12
        )
        enriched_combos.append({**combo_rec, "zmots": zmot_pack})

    return {
        "graph_fingerprint": icp_payload.get("graph_fingerprint"),
        "baseline": icp_payload.get("baseline"),
        "combos": enriched_combos,
        "attributes": {
            fam: [chip.label for chip in chips]
            for fam, chips in attribute_chips.items()
        }
    }


@router.get("/get-icp-archetype-options/{product_id}")
async def get_icp_archetype_options(product_id: str):
    G = build_product_graph(product_id)
    return get_attribute_options(G)

# ========================
# ZMOT LOOKUPS
# ========================

@router.post("/get-zmots-for-attributes/{product_id}")
async def get_zmots_for_attributes(product_id: str, payload: dict, request: Request):
    """
    Given a set of attribute_ids, return top ZMOTs, observable moments, keywords.
    """
    attribute_ids = payload.get("attribute_ids", [])
    if not attribute_ids:
        raise HTTPException(status_code=400, detail="Attribute IDs are required.")

    auth_header = request.headers.get("authorization")
    if not auth_header:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    token = auth_header.split(" ")[1]
    decoded = decode_token(token)
    company_id = decoded.get("company_id")
    if not company_id:
        raise HTTPException(status_code=401, detail="Invalid token or company ID not found")

    product_subgraph = build_product_graph(product_id)
    chips = [Chip(family="attribute", node_id=a, label=a) for a in attribute_ids]
    zmot_ops = collect_zmots_for_attribute_combo(product_subgraph, chips)
    print("ZMOT ops:", zmot_ops)
    return zmot_ops

# ========================
# REVERSE CASE STUDIES
# ========================

@router.post("/get-reverse-case-study/{product_id}")
async def get_reverse_case_study(product_id: str, payload: dict, request: Request):
    """
    Generates reverse case studies for product_id, using attribute sets + engaged nodes.
    """
    print("Generating RCS for product_id:", product_id)
    engaged_nodes = payload.get("selected_node_ids", [])
    attributes = payload.get("attribute_ids", [])
    zmot_id = payload.get("zmot_event_id")
    if zmot_id:
        engaged_nodes.append(zmot_id)
    # also drop falsy node ids from UI bugs
    engaged_nodes = [n for n in engaged_nodes if n]

    auth_header = request.headers.get("authorization")
    if not auth_header:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    token = auth_header.split(" ")[1]
    decoded = decode_token(token)
    company_id = decoded.get("company_id")
    if not company_id:
        raise HTTPException(status_code=401, detail="Invalid token or company ID not found")

    product_subgraph = build_product_graph(product_id)
    print("simulating rcs with engaged nodes:", engaged_nodes, "and attr: ", attributes)
    output = simulate_rcs(product_subgraph, engaged_nodes=engaged_nodes, attributes=attributes)
    return output

# ========================
# ARSENAL LIBRARY
# ========================
# GET /get-arsenal-library/{product_id}
@router.get("/get-arsenal-library/{product_id}")
async def get_arsenal_library(product_id: str, request: Request):
    """
    Returns the arsenal library for the given company.
    """
    print("Product ID for arsenal library:", product_id)
    try:
        auth_header = request.headers.get("authorization")
        if not auth_header:
            raise HTTPException(status_code=401, detail="Missing Authorization header")
        token = auth_header.split(" ")[1]
        decoded = decode_token(token)
        company_id = decoded.get("company_id")

        if not company_id:
            raise HTTPException(status_code=401, detail="Invalid token or company ID not found")
        product_subgraph = build_product_graph(product_id)
        product_id_actual = get_product_id_from_subgraph(product_subgraph)

        arsenal_library = get_all_arsenals(product_id_actual)
        return {"arsenal": arsenal_library}
    except Exception as e:
        print("❌ Get Arsenal Library error:", e)
        raise HTTPException(status_code=500, detail="Could not retrieve arsenal library")


# ========================
# TARGET ACCOUNTS
# ========================

@router.get("/get-target-account-entry-parameters/{company_id}")
async def get_target_account_entry_parameters(company_id: str, product_id: str, request: Request):
    """
    Returns permissible attribute families for ICP entry.
    """
    auth_header = request.headers.get("authorization")
    if not auth_header:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    token = auth_header.split(" ")[1]
    decoded = decode_token(token)
    if decoded.get("company_id") != company_id:
        raise HTTPException(status_code=401, detail="Invalid token or company ID not found")

    return {"families": ["industry", "revenue", "employees", "funding", "geography"]}

@router.get("/get-target-accounts/{company_id}")
async def get_target_accounts(company_id: str, product_id: str, request: Request):
    auth_header = request.headers.get("authorization")
    if not auth_header:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    token = auth_header.split(" ")[1]
    decoded = decode_token(token)
    if decoded.get("company_id") != company_id:
        raise HTTPException(status_code=401, detail="Invalid token or company ID not found")

    product_subgraph = build_product_graph(product_id)
    product_id_actual = get_product_id_from_subgraph(product_subgraph)
    accounts = load_target_accounts_from_db(product_id_actual)
    
    return {"accounts": accounts}

@router.get("/get-accounts-for-rcs/{product_id}")
async def get_accounts_for_rcs(product_id: str, request: Request):
    # --- Auth checks ---
    auth_header = request.headers.get("authorization")
    if not auth_header:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    parts = auth_header.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=401, detail="Invalid Authorization header format")

    token = parts[1]
    decoded = decode_token(token)
    company_id = decoded.get("company_id")
    if not company_id:
        raise HTTPException(status_code=401, detail="Invalid token or company ID not found")

    account_ids = get_target_account_ids(product_id, {"status": {"nin": ["Closed-won", "Closed-lost"]}})
    print("account ids:", account_ids)
    accounts = []
    for account_id in account_ids:
        account = get_account_by_id(product_id=product_id, account_id=account_id)
        if account:
            accounts.append(account)

    
    return {"accounts": accounts}

@router.post("/save-target-accounts/{company_id}")
async def save_target_accounts(company_id: str, product_id: str, payload: dict, request: Request):
    """
    Save target account with explicit attributes.
    """
    auth_header = request.headers.get("authorization")
    if not auth_header:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    token = auth_header.split(" ")[1]
    decoded = decode_token(token)
    if decoded.get("company_id") != company_id:
        raise HTTPException(status_code=401, detail="Invalid token or company ID not found")

    product_subgraph = build_product_graph(product_id)
    product_id_actual = get_product_id_from_subgraph(product_subgraph)
    save_and_update_target_accounts([payload], product_id_actual)
    return {"message": "Account saved successfully."}


@router.delete("/delete-target-account/{company_id}/{account_id}")
async def delete_target_account(company_id: str, account_id: str, product_id: str, request: Request):
    print("Attempting deletion of target account:", account_id)
    try:
        auth_header = request.headers.get("authorization")
        if not auth_header:
            raise HTTPException(status_code=401, detail="Missing Authorization header")
        token = auth_header.split(" ")[1]
        decoded = decode_token(token)
        if decoded.get("company_id") != company_id:
            raise HTTPException(status_code=401, detail="Invalid token or company ID not found")
        product_subgraph = build_product_graph(product_id)
        product_id_actual = get_product_id_from_subgraph(product_subgraph)
        success = delete_target_account_handler(account_id, product_id_actual)
        if not success:
            # Don't raise inside try, or re-raise directly
            raise HTTPException(status_code=404, detail="Target account not found")
        return {"message": "Target account deleted"}
    except HTTPException as e:
        # Re-raise HTTPException so FastAPI handles it correctly
        raise e
    except Exception as e:
        print("❌ Delete Target Account error:", e)
        raise HTTPException(status_code=500, detail="Could not delete target account")
    
@router.get("/show-crm-win-model/{product_id}")
async def get_crm_win_model(product_id: str, request: Request):
    auth_header = request.headers.get("authorization")
    if not auth_header:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    token = auth_header.split(" ")[1]
    decoded = decode_token(token)
    company_id = decoded.get("company_id")
    if not company_id:
        raise HTTPException(status_code=401, detail="Invalid token or company ID not found")

    product_subgraph = build_product_graph(product_id)
    product_id_actual = get_product_id_from_subgraph(product_subgraph)
    accounts = load_target_accounts_from_db(product_id_actual)
    try:
        model = generate_win_regression(product_subgraph, accounts)
    except Exception as e:
        print("❌ Win rate model error:", e)
        model = None
    return model
    

# ========================

#--------------
# STATIC ROUTES FOR TESTING - NEED TO BUILD OUT
#--------------

@router.get("/get-metadata")
async def get_metadata():
    return JSONResponse({
        "personas": ["CFO", "VP Finance", "RevOps", "CPO/Pricing", "CTO", "Developers", "Head of RevOps"],
        "channels": ["Website", "Microsite", "SEO", "Email", "LinkedIn", "PR", "Partner", "ABM Ads", "Webinar", "Events", "Dev/OSS", "YouTube", "Conferences"],
        "stages": ["Awareness", "Interest", "Engagement", "Evaluation", "Conversion", "Expansion", "Evangelism"],
        "quarters": ["Q1", "Q2", "Q3", "Q4"]
    })


@router.get("/get-comprehensive-execution-plan/{product_id}")
async def get_comprehensive_execution_plan(
    product_id: str,
    request: Request,
    # explicitly mark these as query params
    account_id: Optional[str] = Query(None, description="Filter to a single account id"),
    window_start: Optional[str] = Query(None, description="YYYY-MM-DD (inclusive)"),
    window_end:   Optional[str] = Query(None, description="YYYY-MM-DD (inclusive)"),
):
    # --- auth (unchanged) ---
    auth_header = request.headers.get("authorization")
    if not auth_header:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    token = auth_header.split(" ")[1]
    decoded = decode_token(token)
    if not decoded.get("company_id"):
        raise HTTPException(status_code=401, detail="Invalid token or company ID not found")

    # normalize blank -> None so the planner doesn’t try to match ""
    if account_id == "":
        account_id = None

    try:
        print(f"[plan] product_id={product_id} account_id={account_id} "
              f"window_start={window_start} window_end={window_end}")

        plan = build_integrated_portfolio_plan(product_id, account_id=account_id)

        if window_start or window_end:
            plan = _filter_by_window(plan, window_start, window_end)

        quarters = plan.get("quarters") or []
        total_campaigns = sum(len(q.get("campaigns") or []) for q in quarters)
        return JSONResponse(content={"plan": plan, "debug": {
            "quarters": len(quarters),
            "total_campaigns": total_campaigns,
        }})
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load plan: {e}")



# ========================
# Strategy Simulator
# ========================
class SimulatePayload(BaseModel):
    account_id: str
    product_id: str
    attributes: List[str] = []
    zmots: List[str] = []
    persona_engagements: List[str] = []
    stage: str = "auto"
    plays_per_step: int = 2
    boost_factor: float = 2.0
    archetype: Dict[str, Any] = {}


@router.post("/simulate-rcs/{product_id}")
def simulate_rcs_endpoint(product_id: str, payload: SimulatePayload, request: Request):
    
    auth_header = request.headers.get("authorization")
    if not auth_header:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    token = auth_header.split(" ")[1]
    decoded = decode_token(token)
    company_id = decoded.get("company_id")
    if not company_id:
        raise HTTPException(status_code=401, detail="Invalid token or company ID not found")

    product_subgraph = build_product_graph(product_id)
    product_id_actual = get_product_id_from_subgraph(product_subgraph)
    
    

    sim = SimulationInput(
        account_id=payload.account_id,
        product_id=payload.product_id,
        product_subgraph=G,
        attributes=payload.attributes,
        zmots=payload.zmots,
        persona_engagements=payload.persona_engagements,
        stage=payload.stage,
        plays_per_step=payload.plays_per_step,
        boost_factor=payload.boost_factor,
        archetype=payload.archetype,
    )

    result = simulate_scenario(sim)
    # convert dataclass to JSON
    return {
        "generatedAt": result.generated_at,
        "keyStats": result.key_stats,
        "rcsReport": result.rcs_report,
        "frozenStrategy": result.frozen_strategy,
        "stagePlan": result.stage_plan,
        "diffs": result.diffs,
    }