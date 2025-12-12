import json
import logging
import os
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, Body, Depends, Request, HTTPException, Query
from pydantic import BaseModel

from backend.utils.crm_management.target_account_manager import get_account_by_id, get_target_account_ids, load_target_accounts_from_db, save_and_update_target_accounts, delete_target_account_handler
from backend.utils.graph_base.graph_utils.graph_confidence import compute_graph_confidence
from backend.utils.graph_base.network_graph import build_product_graph, get_product_id_from_subgraph
from backend.utils.inference.crm_analysis.actual_win_estimator import generate_win_regression
from backend.utils.inference.discovery_engine.agentic_engine.agentic_loop import run_agentic_loop, run_frontier_expansion

from backend.auth.jwt_handler import decode_token
from sqlalchemy.orm import Session
from backend.database import get_db
from backend.utils.inference.rcs_generators.rcs_simulator import simulate_rcs
from backend.utils.knowledge_base.arsenal_generation import get_all_arsenals
from backend.utils.knowledge_base.arsenal.service import (
    enum_options_payload,
    create_asset_record,
    create_channel_record,
    update_asset_metadata,
    update_channel_metadata,
)
from backend.utils.knowledge_base.graph_edit_manager import add_or_update_graph, serialize_graph_for_frontend
from backend.utils.knowledge_base.value_prop_analysis import generate_product_value_prop, get_product_id_from_company_id, get_product_value_prop_capabilities

from backend.utils.knowledge_base.persona_generation import (
    get_personas_rcs_priority
)
from backend.utils.knowledge_base.zmot_icp_generation import _collect_attribute_chips, get_attribute_options
import networkx as nx

# Hardcoded path to graph data folder - to be updated in production

from backend.utils.knowledge_base.zmot_icp_generation import Chip, collect_zmots_for_attribute_combo, mine_icp_attribute_uplifts
from backend.utils.crm_management.engagement_service import get_account_engagements, save_and_update_account_engagements, save_engagements_bulk
from backend.utils.crm_management.enrichment_service import (
    build_account_enrichment,
    upsert_persona_match,
    delete_persona_match,
    add_candidate_persona,
    alias_candidate_persona,
)
from backend.utils.crm_management.hubspot_engagements_ingestion import ingest_hubspot_company
from backend.utils.inference.belief_manager.belief_manager import build_belief_thesis_for_account, incremental_learnings_from_thesis
from backend.utils.inference.belief_manager.cache import (
    get_cached_belief_thesis,
)
from backend.utils.inference.belief_manager.journey.learn_service import (
    record_learning_updates_for_account,
    summarize_global_insights,
    apply_persona_recommendation_to_graph,
    apply_edge_recommendation_to_graph,
)
from backend.utils.inference.belief_manager.journey.bgn_service import rebuild_global_thesis
from backend.utils.inference.belief_manager.journey.shm_service import write_meta_episode_from_thesis
from backend.utils.inference.belief_manager.journey.storage import load_global_thesis
from backend.utils.inference.belief_manager.journey.activity_story import (
    build_activity_story,
    infer_latent_activity,
)
from backend.utils.inference.belief_manager.journey.storyline import (
    collect_filter_options,
    compose_portfolio_story,
    compose_storyline,
)
from backend.utils.strategy_builder.comprehensive_plan_generator import (
    build_account_marketing_blueprint,
    build_product_marketing_plan,
)
from backend.utils.api_contracts import (
    build_comprehensive_execution_plan_contract,
    build_account_plan_contract,
    build_insights_inbox_contract,
    build_personas_atlas_contract,
    build_icp_overview_contract,
)
from backend.utils.strategy_builder.persona_insights import build_persona_insights
from backend.utils.strategy_builder.execution_decision_service import (
    list_decisions as list_intervention_decisions,
    upsert_decision as upsert_intervention_decision,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GRAPH_DATA_PATH = os.path.join(BASE_DIR,"backend", "utils", "graph_base", "graph_data")

router = APIRouter()
LOGGER = logging.getLogger(__name__)


def _log_route_output(route_name: str, payload: Any) -> Any:
    """
    Print the final payload returned by a route to the terminal for quick inspection.
    """
    log_payload = payload
    try:
        # Case 1: the route returns a plain list
        if isinstance(payload, list):
            log_payload = payload[:1]

        # Case 2: the route returns an object with a 'results' list
        elif isinstance(payload, dict) and isinstance(payload.get("results"), list):
            # shallow copy the dict, but truncate only for logging
            log_payload = {
                **payload,
                "results": payload["results"][:1],
                "_truncated_results": {
                    "logged_count": min(len(payload["results"]), 1),
                    "total_count": len(payload["results"]),
                },
            }

        # You can add more patterns here if needed (e.g. 'items', 'data', etc.)

        formatted = json.dumps(log_payload, default=str, separators=(",", ":"))
    except Exception:
        # Fallback: just repr the original payload
        formatted = repr(payload)

    print(f"[{route_name}] response: {formatted}")
    return payload  # important: always return the full original payload


def _decode_request_token(request: Request) -> Dict[str, Any]:
    auth_header = request.headers.get("authorization") or request.headers.get("Authorization")
    if not auth_header or " " not in auth_header:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    token = auth_header.split(" ", 1)[1]
    decoded = decode_token(token)
    if not decoded or not decoded.get("company_id"):
        raise HTTPException(status_code=401, detail="Invalid token")
    return decoded


class ApplyRecommendationBody(BaseModel):
    product_id: str
    type: Literal["persona", "edge"]
    recommendation: Dict[str, Any]


class StorylineRequest(BaseModel):
    filters: Optional[Dict[str, str]] = None
    account_id: Optional[str] = None


class PersonaMatchUpsert(BaseModel):
    product_id: str
    persona_id: str
    person_id: Optional[str] = None
    persona_label: Optional[str] = None
    stage: Optional[str] = None
    match_confidence: Optional[float] = None
    source: Optional[str] = "user"
    notes: Optional[str] = None
    match_id: Optional[str] = None
    person_name: Optional[str] = None
    person_title: Optional[str] = None
    person_department: Optional[str] = None
    person_seniority: Optional[str] = None


class PersonaCandidateActionBody(BaseModel):
    product_id: str
    label: str
    title: Optional[str] = None
    department: Optional[str] = None
    seniority: Optional[str] = None
    action: Literal["add", "match"]
    target_persona_id: Optional[str] = None


class InterventionDecisionBody(BaseModel):
    product_id: str
    intervention_id: str
    scope: Literal["portfolio", "account"]
    account_id: Optional[str] = None
    action: Literal["use", "override"]
    persona: Optional[str] = None
    concern: Optional[str] = None
    asset_type: Optional[str] = None
    channel: Optional[str] = None
    recommended_asset_id: Optional[str] = None
    recommended_asset_name: Optional[str] = None
    selected_asset_id: Optional[str] = None
    selected_asset_name: Optional[str] = None
    selected_asset_type: Optional[str] = None
    selected_channel: Optional[str] = None
    notes: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    user_id: Optional[str] = None

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
        LOGGER.debug(
            "Dict output summary: %s\nDict output capabilities: %s",
            result.get("summary"),
            result.get("capabilities"),
        )
        return result
    except Exception as e:
        LOGGER.exception("Analyze error")
        raise HTTPException(status_code=500, detail="Could not run analysis")

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
    except Exception:
        LOGGER.exception("Get confidence error")
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
    LOGGER.info("Deep analyzing product_id: %s", product_id)
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
    LOGGER.info("Product subgraph built successfully for product_id: %s. Starting agent", product_id)
    # Run the agentic loop (update_graph inside that flow should now receive a valid subgraph)
    try:
        run_agentic_loop(product_subgraph)
    except Exception:
        LOGGER.exception("Agentic analysis error")
        raise HTTPException(status_code=500, detail="Agentic analysis failed")

    return {"status": "Agentic analysis complete"}

@router.post("/graph/expand-frontier/{product_id}")
async def expand_frontier(product_id: str, request: Request, body: dict = Body(..., embed=False)):
    LOGGER.info("Expanding frontier for product_id: %s", product_id)
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
    LOGGER.debug("Seed IDs: %s", seed_ids)

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
        LOGGER.debug("[Frontier] Auto-seeded terminals: %s", seed_ids)

   

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
    print("Getting products for company_id: %s", company_id)
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
            LOGGER.info("Product ID found for company_id: %s", company_id)
            product_subgraph = build_product_graph(product_lookup)
            LOGGER.debug("Product subgraph data loaded")
            product_id = get_product_id_from_subgraph(product_subgraph)
            
            product_node = product_subgraph.nodes[product_id]
            products = [{
                "id": product_lookup,
                "domain": product_node.get("domain", ""),
                "industry": product_node.get("industry", ""),
                "summary": product_node.get("summary", ""),
                "url": product_node.get("url", ""),
            }]
        else: 
            print("No product id found for company_id: %s", company_id)
            products = []
        
        LOGGER.debug("Products found: %s", products)
        print("Products found: %s", products)
        return {"products": products}
    except Exception as e:
        LOGGER.exception("Get products error")
        print("Get products error: %s", e)
        raise HTTPException(status_code=500, detail="Could not retrieve products")

# GET /get-product-capabilities/{product_id}
@router.get("/get-product-capabilities/{product_id}")
async def get_product_capabilities(product_id: str, request: Request):
    LOGGER.info("Getting product capabilities for product_id: %s", product_id)
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
    except Exception:
        LOGGER.exception("Get capabilities error")
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
    except Exception:
        LOGGER.exception("export-product-graph error")
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
        LOGGER.info("Graph updated successfully with bulk upsert for product_id %s", product_id)
        return {"message": "Graph updated successfully."}
    except Exception:
        LOGGER.exception("Error in add_or_update_graph")
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


@router.get("/products/{product_id}/persona_insights")
async def get_persona_insights_endpoint(product_id: str, request: Request):
    auth_header = request.headers.get("authorization") or request.headers.get("Authorization")
    if not auth_header:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    token = auth_header.split(" ")[1]
    decoded = decode_token(token)
    if not decoded.get("company_id"):
        raise HTTPException(status_code=401, detail="Invalid token or company ID not found")
    try:
        return build_persona_insights(product_id)
    except HTTPException:
        raise
    except Exception as exc:
        LOGGER.exception("Error building persona insights")
        raise HTTPException(status_code=500, detail=str(exc))


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
    except Exception:
        LOGGER.exception("Get ZMOT ICP error")
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
    LOGGER.debug("Fetching ICP Options")
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
    LOGGER.debug("ZMOT ops: %s", zmot_ops)
    return zmot_ops

# ========================
# REVERSE CASE STUDIES
# ========================

@router.post("/get-reverse-case-study/{product_id}")
async def get_reverse_case_study(product_id: str, payload: dict, request: Request):
    """
    Generates reverse case studies for product_id, using attribute sets + engaged nodes.
    """
    LOGGER.info("Generating RCS for product_id: %s", product_id)
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
    LOGGER.debug("Simulating RCS with engaged nodes: %s and attributes: %s", engaged_nodes, attributes)
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
    LOGGER.debug("Product ID for arsenal library: %s", product_id)
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
    except Exception:
        LOGGER.exception("Get Arsenal Library error")
        raise HTTPException(status_code=500, detail="Could not retrieve arsenal library")


@router.get("/arsenal/options")
async def get_arsenal_options(request: Request, db: Session = Depends(get_db)):
    auth_header = request.headers.get("authorization")
    if not auth_header:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    token = auth_header.split(" ")[1]
    decoded = decode_token(token)
    if not decoded.get("company_id"):
        raise HTTPException(status_code=401, detail="Invalid token or company ID not found")
    return enum_options_payload(db)


@router.post("/arsenal/assets")
async def create_arsenal_asset_endpoint(
    payload: Dict[str, Any] = Body(...),
    request: Request = None,
    db: Session = Depends(get_db),
):
    if request is None:
        raise HTTPException(status_code=400, detail="Request context required")
    auth_header = request.headers.get("authorization")
    if not auth_header:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    token = auth_header.split(" ")[1]
    decoded = decode_token(token)
    if not decoded.get("company_id"):
        raise HTTPException(status_code=401, detail="Invalid token or company ID not found")
    product_id = (payload.get("product_id") or "").strip()
    if not product_id:
        raise HTTPException(status_code=400, detail="product_id is required")
    try:
        asset = create_asset_record(db, product_id=product_id, payload=payload)
        db.commit()
        return {"asset": asset}
    except ValueError as exc:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        db.rollback()
        LOGGER.exception("Asset creation error")
        raise HTTPException(status_code=500, detail="Failed to create asset")


@router.post("/arsenal/channels")
async def create_arsenal_channel_endpoint(
    payload: Dict[str, Any] = Body(...),
    request: Request = None,
    db: Session = Depends(get_db),
):
    if request is None:
        raise HTTPException(status_code=400, detail="Request context required")
    auth_header = request.headers.get("authorization")
    if not auth_header:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    token = auth_header.split(" ")[1]
    decoded = decode_token(token)
    if not decoded.get("company_id"):
        raise HTTPException(status_code=401, detail="Invalid token or company ID not found")
    product_id = (payload.get("product_id") or "").strip()
    if not product_id:
        raise HTTPException(status_code=400, detail="product_id is required")
    try:
        channel = create_channel_record(db, product_id=product_id, payload=payload)
        db.commit()
        return {"channel": channel}
    except ValueError as exc:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        db.rollback()
        LOGGER.exception("Channel creation error")
        raise HTTPException(status_code=500, detail="Failed to create channel")


@router.patch("/arsenal/assets/{asset_id}")
async def patch_arsenal_asset(
    asset_id: str,
    payload: Dict[str, Any] = Body(...),
    request: Request = None,
    db: Session = Depends(get_db),
):
    if request is None:
        raise HTTPException(status_code=400, detail="Request context required")
    auth_header = request.headers.get("authorization")
    if not auth_header:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    token = auth_header.split(" ")[1]
    decoded = decode_token(token)
    if not decoded.get("company_id"):
        raise HTTPException(status_code=401, detail="Invalid token or company ID not found")

    try:
        updated = update_asset_metadata(db, asset_id=asset_id, patch=payload or {})
        db.commit()
        return {"asset": updated}
    except LookupError as exc:
        db.rollback()
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        db.rollback()
        LOGGER.exception("Asset metadata update error")
        raise HTTPException(status_code=500, detail="Failed to update asset metadata")


@router.post("/arsenal/assets/{asset_id}/approve")
async def approve_arsenal_asset(
    asset_id: str,
    request: Request = None,
    db: Session = Depends(get_db),
):
    if request is None:
        raise HTTPException(status_code=400, detail="Request context required")
    auth_header = request.headers.get("authorization")
    if not auth_header:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    token = auth_header.split(" ")[1]
    decoded = decode_token(token)
    if not decoded.get("company_id"):
        raise HTTPException(status_code=401, detail="Invalid token or company ID not found")
    try:
        updated = update_asset_metadata(db, asset_id=asset_id, patch={"approval_status": "approved"})
        db.commit()
        return {"asset": updated}
    except LookupError as exc:
        db.rollback()
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception:
        db.rollback()
        LOGGER.exception("Asset approval failed")
        raise HTTPException(status_code=500, detail="Failed to approve asset")


@router.patch("/arsenal/channels/{channel_id}")
async def patch_arsenal_channel(
    channel_id: str,
    payload: Dict[str, Any] = Body(...),
    request: Request = None,
    db: Session = Depends(get_db),
):
    if request is None:
        raise HTTPException(status_code=400, detail="Request context required")
    auth_header = request.headers.get("authorization")
    if not auth_header:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    token = auth_header.split(" ")[1]
    decoded = decode_token(token)
    if not decoded.get("company_id"):
        raise HTTPException(status_code=401, detail="Invalid token or company ID not found")

    try:
        updated = update_channel_metadata(db, channel_id=channel_id, patch=payload or {})
        db.commit()
        return {"channel": updated}
    except LookupError as exc:
        db.rollback()
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        db.rollback()
        LOGGER.exception("Channel metadata update error")
        raise HTTPException(status_code=500, detail="Failed to update channel metadata")


@router.post("/arsenal/channels/{channel_id}/approve")
async def approve_arsenal_channel(
    channel_id: str,
    request: Request = None,
    db: Session = Depends(get_db),
):
    if request is None:
        raise HTTPException(status_code=400, detail="Request context required")
    auth_header = request.headers.get("authorization")
    if not auth_header:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    token = auth_header.split(" ")[1]
    decoded = decode_token(token)
    if not decoded.get("company_id"):
        raise HTTPException(status_code=401, detail="Invalid token or company ID not found")
    try:
        updated = update_channel_metadata(db, channel_id=channel_id, patch={"approval_status": "approved"})
        db.commit()
        return {"channel": updated}
    except LookupError as exc:
        db.rollback()
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception:
        db.rollback()
        LOGGER.exception("Channel approval failed")
        raise HTTPException(status_code=500, detail="Failed to approve channel")


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
    # print("account ids:", account_ids)
    accounts = []
    for account_id in account_ids:
        account = get_account_by_id(product_id=product_id, account_id=account_id)
        if account:
            accounts.append(account)


    return _log_route_output("get_accounts_for_rcs", {"accounts": accounts})


@router.get("/accounts/{account_id}/enrichment")
async def get_account_enrichment(
    account_id: str,
    product_id: str,
    request: Request,
    db: Session = Depends(get_db),
):
    auth_header = request.headers.get("authorization")
    if not auth_header:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    token = auth_header.split(" ")[1]
    decoded = decode_token(token)
    if not decoded.get("company_id"):
        raise HTTPException(status_code=401, detail="Invalid token or company ID not found")

    try:
        payload = build_account_enrichment(
            product_id=product_id,
            account_id=account_id,
            db=db,
        )
        return _log_route_output("get_account_enrichment", payload)
    except Exception as exc:
        LOGGER.exception("Account enrichment error: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to build account enrichment summary")


@router.post("/accounts/{account_id}/enrichment/matches")
async def upsert_account_persona_match(
    account_id: str,
    payload: PersonaMatchUpsert,
    request: Request,
    db: Session = Depends(get_db),
):
    auth_header = request.headers.get("authorization")
    if not auth_header:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    token = auth_header.split(" ")[1]
    decoded = decode_token(token)
    if not decoded.get("company_id"):
        raise HTTPException(status_code=401, detail="Invalid token or company ID not found")

    if not payload.persona_label:
        raise HTTPException(status_code=400, detail="persona_label is required")

    try:
        match_payload = upsert_persona_match(
            db,
            product_id=payload.product_id,
            account_id=account_id,
            persona_id=payload.persona_id,
            person_id=payload.person_id,
            persona_label=payload.persona_label,
            stage=payload.stage,
            match_confidence=payload.match_confidence,
            source=payload.source,
            notes=payload.notes,
            match_id=payload.match_id,
            person_name=payload.person_name,
            person_title=payload.person_title,
            person_department=payload.person_department,
            person_seniority=payload.person_seniority,
        )
        db.commit()
        return _log_route_output("upsert_account_persona_match", {"match": match_payload})
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        db.rollback()
        LOGGER.exception("Upsert persona match error: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to save persona match")


@router.delete("/accounts/{account_id}/enrichment/matches/{match_id}")
async def delete_account_persona_match(
    account_id: str,
    match_id: str,
    product_id: str,
    request: Request,
    db: Session = Depends(get_db),
):
    auth_header = request.headers.get("authorization")
    if not auth_header:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    token = auth_header.split(" ")[1]
    decoded = decode_token(token)
    if not decoded.get("company_id"):
        raise HTTPException(status_code=401, detail="Invalid token or company ID not found")
    try:
        delete_persona_match(
            db,
            product_id=product_id,
            account_id=account_id,
            match_id=match_id,
        )
        db.commit()
        return _log_route_output("delete_account_persona_match", {"status": "deleted"})
    except ValueError as exc:
        db.rollback()
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        db.rollback()
        LOGGER.exception("Delete persona match error: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to delete persona match")


@router.post("/accounts/{account_id}/enrichment/personas")
async def handle_candidate_persona_action(
    account_id: str,
    payload: PersonaCandidateActionBody,
    request: Request,
):
    _decode_request_token(request)
    if payload.action == "add":
        result = add_candidate_persona(
            payload.product_id,
            payload.label,
            title=payload.title,
            department=payload.department,
            seniority=payload.seniority,
        )
        return {"persona": result}
    if payload.action == "match":
        if not payload.target_persona_id:
            raise HTTPException(status_code=400, detail="target_persona_id is required for match action.")
        try:
            result = alias_candidate_persona(
                payload.product_id,
                payload.target_persona_id,
                payload.label,
            )
            return {"persona": result}
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
    raise HTTPException(status_code=400, detail="Unsupported action.")


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
        return _log_route_output("delete_target_account", {"message": "Target account deleted"})
    except HTTPException as e:
        # Re-raise HTTPException so FastAPI handles it correctly
        raise e
    except Exception as e:
        LOGGER.exception("Delete Target Account error: %s", e)
        raise HTTPException(status_code=500, detail="Could not delete target account")
    
#------- Engagements Routes---------

@router.post("/save-account-engagements/{product_id}")
async def save_account_engagements(product_id: str, request: Request):
    """
    Save target account engagements.
    """
    auth_header = request.headers.get("authorization")
    if not auth_header:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    token = auth_header.split(" ")[1]
    decoded = decode_token(token)
    company_id = decoded.get("company_id")
    if not company_id:
        raise HTTPException(status_code=401, detail="Invalid token or company ID not found")
    # print("Saving engagements for product_id:", product_id)
    # print("Request body:", await request.body())
    try:
        body: Dict[str, Any] = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    engagements: List[Dict[str, Any]] = (body or {}).get("engagements", [])
    if not engagements or not isinstance(engagements, list):
        raise HTTPException(status_code=400, detail="Engagements data is required (non-empty list)")

    # --- Optional: resolve lookup product ids to canonical ids ---
    try:
        product_subgraph = build_product_graph(product_id)
        product_id_actual = get_product_id_from_subgraph(product_subgraph) or product_id
    except Exception:
        product_id_actual = product_id  # already canonical → continue

    # --- Persist ---
    try:
        # print("Calling save and update engagements with body content:", engagements)
        result = save_and_update_account_engagements(product_id_actual, engagements)
        # e.g., {"created": N, "touched_accounts": [...]}
        return _log_route_output("save_account_engagements", {"ok": True, **result})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save engagements: {e}")
    

@router.get("/load-account-engagements/{product_id}/{account_id}")
async def load_account_engagements(product_id: str, account_id: str, request: Request):
    auth_header = request.headers.get("authorization")
    if not auth_header:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    token = auth_header.split(" ")[1]
    decoded = decode_token(token)
    company_id = decoded.get("company_id")
    if not company_id:
        raise HTTPException(status_code=401, detail="Invalid token or company ID not found")

    rows = get_account_engagements(product_id, account_id)
    return {"engagements": rows}

#---- Persona Matching for Engagements ------
    
@router.get("/get-persona-matches/{product_id}/{account_id}")
async def get_persona_matches(product_id: str, account_id: str, request: Request, db: Session = Depends(get_db)):
    # --- Auth (unchanged) ---
    auth_header = request.headers.get("authorization")
    if not auth_header:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    try:
        token = auth_header.split(" ")[1]
    except Exception:
        raise HTTPException(status_code=401, detail="Malformed Authorization header")
    decoded = decode_token(token)
    if not decoded or not decoded.get("company_id"):
        raise HTTPException(status_code=401, detail="Invalid token")

    try:
        account_record = get_account_by_id(product_id, account_id) or {}
        incremental: Optional[Dict[str, Any]] = None

        def _after_build(local_db: Session, thesis_payload: Dict[str, Any]) -> None:
            nonlocal incremental
            write_meta_episode_from_thesis(
                local_db,
                product_id=product_id,
                account_id=account_id,
                thesis=thesis_payload,
            )
            incremental = incremental_learnings_from_thesis(thesis_payload)
            record_learning_updates_for_account(
                local_db,
                product_id=product_id,
                account_id=account_id,
                incremental_learnings=incremental,
            )

        thesis, rebuilt = get_cached_belief_thesis(
            product_id=product_id,
            account_id=account_id,
            build_fn=lambda: build_belief_thesis_for_account(
                product_id, account_id
            ),
            db=db,
            after_build=_after_build,
        )

        if not rebuilt:
            incremental = incremental_learnings_from_thesis(thesis)
        else:
            db.commit()

        global_insights = summarize_global_insights(db, product_id=product_id)

        activity = (thesis.get("journey") or {}).get("steps") or []
        latent_activity = infer_latent_activity(thesis)
        activity_story = build_activity_story(thesis)
        storyline = compose_storyline(
            account_name=account_record.get("account_name") or account_id,
            account_meta=account_record,
            deal_status=account_record.get("deal_status"),
            journey_steps=activity,
            activity_story=activity_story,
            product_id=product_id,
        )
        thesis["storyline"] = storyline

        response = {
            "product_id": product_id,
            "account_id": account_id,
            "global": global_insights,
            "incremental": incremental,
            "activity": activity,
            "latent_activity": latent_activity,
            "activity_story": activity_story,
            "storyline": storyline,
            "thesis": thesis,
        }
        return _log_route_output("get_persona_matches", response)
    except Exception as e:
        LOGGER.exception("Error in get_persona_matches: %s", e)
        raise HTTPException(status_code=500, detail=f"Persona match failed: {e}")


@router.post("/journey/rebuild-global-thesis/{product_id}")
async def rebuild_global_thesis_route(
    product_id: str,
    request: Request,
    payload: Dict[str, Any] = Body(default=None),
    db: Session = Depends(get_db),
):
    auth_header = request.headers.get("authorization")
    if not auth_header:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    try:
        token = auth_header.split(" ")[1]
    except Exception:
        raise HTTPException(status_code=401, detail="Malformed Authorization header")
    decoded = decode_token(token)
    if not decoded or not decoded.get("company_id"):
        raise HTTPException(status_code=401, detail="Invalid token")

    window = None
    if isinstance(payload, dict):
        try:
            window_val = payload.get("window")
            if window_val is not None:
                window = int(window_val)
        except (TypeError, ValueError):
            window = None

    thesis = rebuild_global_thesis(db, product_id=product_id, window=window)
    return {"ok": True, "thesis": thesis}


@router.get("/journey/global-thesis/{product_id}")
async def fetch_global_thesis(
    product_id: str,
    request: Request,
    db: Session = Depends(get_db),
):
    auth_header = request.headers.get("authorization")
    if not auth_header:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    try:
        token = auth_header.split(" ")[1]
    except Exception:
        raise HTTPException(status_code=401, detail="Malformed Authorization header")
    decoded = decode_token(token)
    if not decoded or not decoded.get("company_id"):
        raise HTTPException(status_code=401, detail="Invalid token")

    thesis = load_global_thesis(product_id)
    insights = summarize_global_insights(db, product_id=product_id)
    return {"product_id": product_id, "thesis": thesis, "insights": insights}


@router.post("/journey/apply-recommendation")
async def apply_recommendation(
    body: ApplyRecommendationBody,
    request: Request,
    db: Session = Depends(get_db),
):
    auth_header = request.headers.get("authorization")
    if not auth_header:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    try:
        token = auth_header.split(" ")[1]
    except Exception:
        raise HTTPException(status_code=401, detail="Malformed Authorization header")
    decoded = decode_token(token)
    if not decoded or not decoded.get("company_id"):
        raise HTTPException(status_code=401, detail="Invalid token")

    try:
        if body.type == "persona":
            result = apply_persona_recommendation_to_graph(
                body.product_id, body.recommendation, db=db
            )
        else:
            result = apply_edge_recommendation_to_graph(
                body.product_id, body.recommendation, db=db
            )
        db.commit()
    except Exception as exc:
        db.rollback()
        raise exc

    insights = summarize_global_insights(db, product_id=body.product_id)
    return {
        "ok": True,
        "applied": result,
        "insights": insights,
    }


@router.get("/get-comprehensive-execution-plan/{product_id}")
async def get_comprehensive_execution_plan(
    product_id: str,
    request: Request,
    account_id: Optional[str] = None,
):
    auth_header = request.headers.get("authorization") or request.headers.get("Authorization")
    if not auth_header or " " not in auth_header:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    token = auth_header.split(" ", 1)[1]
    decoded = decode_token(token)
    if not decoded or not decoded.get("company_id"):
        raise HTTPException(status_code=401, detail="Invalid token")

    try:
        plan = build_product_marketing_plan(product_id, account_id=account_id)
        return _log_route_output("get_comprehensive_execution_plan", plan)
    except HTTPException:
        raise
    except Exception as exc:
        LOGGER.exception("get_comprehensive_execution_plan error: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to build marketing plan")


@router.get("/portfolio/{product_id}/execution-plan")
def portfolio_execution_plan(
    product_id: str,
    request: Request,
    account_id: Optional[str] = None,
):
    decoded = _decode_request_token(request)
    try:
        plan = build_product_marketing_plan(product_id, account_id=account_id)
        contract = build_comprehensive_execution_plan_contract(plan, product_id)
        return _log_route_output("portfolio_execution_plan", contract)
    except Exception as exc:
        LOGGER.exception("portfolio_execution_plan error: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to build execution plan")


@router.get("/accounts/{account_id}/plan")
def account_plan(
    account_id: str,
    request: Request,
    product_id: str = Query(...),
):
    _decode_request_token(request)
    try:
        plan = build_account_marketing_blueprint(product_id, account_id)
        contract = build_account_plan_contract(plan, product_id)
        return _log_route_output("account_plan", contract)
    except Exception as exc:
        LOGGER.exception("account_plan error: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to build account plan")


@router.get("/insights/inbox")
def insights_inbox(
    request: Request,
    product_id: str = Query(...),
    db: Session = Depends(get_db),
):
    _decode_request_token(request)
    insights = summarize_global_insights(db, product_id=product_id)
    contract = build_insights_inbox_contract(insights)
    return _log_route_output("insights_inbox", contract)


@router.get("/personas/atlas")
def personas_atlas(
    request: Request,
    product_id: str = Query(...),
    db: Session = Depends(get_db),
):
    _decode_request_token(request)
    insights = summarize_global_insights(db, product_id=product_id)
    contract = build_personas_atlas_contract(insights.get("product_insights") or {})
    return _log_route_output("personas_atlas", contract)


@router.get("/icps/overview")
def icp_overview(
    request: Request,
    product_id: str = Query(...),
    db: Session = Depends(get_db),
):
    _decode_request_token(request)
    insights = summarize_global_insights(db, product_id=product_id)
    contract = build_icp_overview_contract(insights.get("product_insights") or {})
    return _log_route_output("icp_overview", contract)


@router.get("/execution-interventions/decisions/{product_id}")
def get_execution_intervention_decisions(
    product_id: str,
    request: Request,
    account_id: Optional[str] = None,
    db: Session = Depends(get_db),
):
    _decode_request_token(request)
    decisions = list_intervention_decisions(
        db,
        product_id=product_id,
        account_id=account_id,
    )
    return {"decisions": [record.to_dict() for record in decisions]}


@router.post("/execution-interventions/decide")
def decide_execution_intervention(
    body: InterventionDecisionBody,
    request: Request,
    db: Session = Depends(get_db),
):
    decoded = _decode_request_token(request)
    if body.scope == "account" and not body.account_id:
        raise HTTPException(status_code=400, detail="account_id is required for account scope")

    status = "accepted" if body.action == "use" else "overridden"
    if body.action == "override" and not body.selected_asset_name:
        raise HTTPException(
            status_code=400,
            detail="selected_asset_name is required when overriding a recommendation",
        )

    selected_asset_id = body.selected_asset_id
    selected_asset_name = body.selected_asset_name
    selected_asset_type = body.selected_asset_type or body.asset_type
    selected_channel = body.selected_channel or body.channel
    if body.action == "use":
        selected_asset_id = body.recommended_asset_id
        selected_asset_name = body.recommended_asset_name or body.asset_type
        selected_asset_type = body.asset_type
        selected_channel = body.channel

    record = upsert_intervention_decision(
        db,
        product_id=body.product_id,
        scope=body.scope,
        account_id=body.account_id,
        intervention_id=body.intervention_id,
        status=status,
        persona=body.persona,
        concern=body.concern,
        asset_type=body.asset_type,
        channel=body.channel,
        recommended_asset_id=body.recommended_asset_id,
        recommended_asset_name=body.recommended_asset_name,
        selected_asset_id=selected_asset_id,
        selected_asset_name=selected_asset_name,
        selected_asset_type=selected_asset_type,
        selected_channel=selected_channel,
        notes=body.notes,
        metadata=body.metadata,
        user_id=body.user_id or decoded.get("user_id"),
    )
    return {"decision": record.to_dict()}


def _matches_story_filters(meta: Optional[Dict[str, Any]], filters: Dict[str, str]) -> bool:
    if not filters:
        return True
    meta = meta or {}
    for key, value in filters.items():
        if not value:
            continue
        current = meta.get(key)
        if current is None or current == "":
            return False
        if isinstance(current, (list, tuple)):
            normalized = {str(item).lower() for item in current if item}
            if str(value).lower() not in normalized:
                return False
        else:
            if str(current).lower() != str(value).lower():
                return False
    return True


@router.post("/journey/storyline/{product_id}")
async def generate_storyline_view(
    product_id: str,
    body: StorylineRequest,
    request: Request,
):
    auth_header = request.headers.get("authorization") or request.headers.get("Authorization")
    if not auth_header or " " not in auth_header:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    token = auth_header.split(" ", 1)[1]
    decoded = decode_token(token)
    if not decoded or not decoded.get("company_id"):
        raise HTTPException(status_code=401, detail="Invalid token")

    filters = {k: v for k, v in (body.filters or {}).items() if v} if body else {}
    try:
        plan = build_product_marketing_plan(product_id)
    except Exception as exc:
        LOGGER.exception("Storyline generation failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to build storyline")

    accounts = plan.get("accounts", [])
    available_filters = collect_filter_options(accounts)

    if body and body.account_id:
        matched_accounts = [
            acct for acct in accounts if acct.get("account_id") == body.account_id
        ]
    else:
        matched_accounts = [
            acct for acct in accounts if _matches_story_filters(acct.get("meta"), filters)
        ]

    if not matched_accounts:
        return _log_route_output(
            "generate_storyline_view",
            {
                "storyline": None,
                "available_filters": available_filters,
                "match_count": 0,
            },
        )

    storyline = compose_portfolio_story(matched_accounts, filters)
    return _log_route_output(
        "generate_storyline_view",
        {
            "storyline": storyline,
            "available_filters": available_filters,
            "match_count": len(matched_accounts),
        },
    )

#---- Math Models Routes -----

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
        LOGGER.exception("Win rate model error: %s", e)
        model = None
    return _log_route_output("get_crm_win_model", model)
    

# ========================
# CRM Ingestion Logic
# ========================

@router.post("/engagements/bulk/{product_id}/{account_id}")
def ingest_bulk(product_id: str, account_id: str, payload: dict, request: Request):
    # auth
    auth = request.headers.get("authorization")
    if not auth:
        raise HTTPException(status_code=401, detail="Missing Authorization")
    token = auth.split(" ")[1]
    if not decode_token(token).get("company_id"):
        raise HTTPException(status_code=401, detail="Invalid token")

    items = payload.get("engagements", [])
    # print("Payload for processing:", items)
    try: 
        count = save_engagements_bulk(product_id, account_id, items)
        return _log_route_output("ingest_bulk", {"ok": True, "overwritten_account": account_id, "inserted": count})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Bulk ingest failed: {e}")

@router.post("/ingest/hubspot/{product_id}/{account_id}")
async def ingest_hs(product_id: str, account_id: str, company_name: str, request: Request):
    # auth
    auth = request.headers.get("authorization")
    if not auth:
        raise HTTPException(status_code=401, detail="Missing Authorization")
    token = auth.split(" ")[1]
    if not decode_token(token).get("company_id"):
        raise HTTPException(status_code=401, detail="Invalid token")

    try:
        data = ingest_hubspot_company(company_name)
        # attach account_id to each engagement row
        for e in data["engagements"]:
            e["account_id"] = account_id
        # persist using your existing save path
        save_engagements_bulk(product_id, data["engagements"])
        return {"status": "ok", "ingested": len(data["engagements"])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingest failed: {e}")
