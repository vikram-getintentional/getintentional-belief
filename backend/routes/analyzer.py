from fastapi import APIRouter, Depends, Request, HTTPException
from backend.utils.graph_base.network_graph import add_capabilities_to_product, build_product_graph, get_node_by_id, get_product_id_from_subgraph, update_capabilities_by_nodes_list
from backend.utils.inference.discovery_engine.agentic_engine.agentic_loop import agentic_inference, run_agentic_loop
from backend.utils.inference.discovery_engine.openai_helper import extract_summary_and_capabilities
from backend.auth.jwt_handler import decode_token
from backend.database import SessionLocal
from sqlalchemy.orm import Session
from backend.database import get_db
from backend.utils.inference.rcs_generators.rcs_simulator import simulate_rcs
from backend.utils.knowledge_base.value_prop_analysis import generate_product_value_prop, get_product_id_from_company_id, get_product_value_prop_capabilities

from backend.utils.knowledge_base.persona_generation import (
    get_company_products,
    get_persona_relevance
)

import networkx as nx

# Hardcoded path to graph data folder - to be updated in production

import os

from backend.utils.knowledge_base.zmot_icp_generation import get_archetypes_by_relevance, get_best_zmots_for_archetype, get_zmot_icp_relevance

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GRAPH_DATA_PATH = os.path.join(BASE_DIR,"backend", "utils", "graph_base", "graph_data")

router = APIRouter()

# 
@router.get("/company/{company_id}/product-subgraph")
def get_product_subgraph(company_id: str):
    # Try to load the product subgraph for this company
    product_id = get_product_id_from_company_id(company_id)
    if product_id is None:
        return {"exists": False}

    # Get product value prop and capabilities
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

# 🔎 Basic scraper-based analysis
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

# 🔬 Deep persona + pain inference
@router.post("/analyze/deep")
async def analyze_deep(payload: dict, request: Request):
    product_id = payload.get("product_id")
    print("Analyzing deep for product_id:", product_id)
    if not product_id:
        raise HTTPException(status_code=400, detail="Product ID required.")

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
        print("Extracted product subgraph with total nodes:", len(product_subgraph.nodes))
        print("Starting hop0 inference with agent loop")
        # New code for agentic looping

        run_agentic_loop(product_subgraph)
        print("Agentic analysis completed for product_id:", product_id)
        return {"status": "Agentic analysis complete"}

        #hop_0_results = infer_with_rules_then_fallback(product_id, product_subgraph)
        # Run Hop+ upstream inference
        #hop_plus_results = infer_upstream_with_rules(product_subgraph)
        #return {"hop_0_results": hop_0_results}

    except Exception as e:
        print("❌ Deep inference error:", e)
        raise HTTPException(status_code=500, detail="Could not run deep analysis")
    
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


# GET /get-personas/{product_id}
@router.get("/get-personas/{product_id}")
async def get_personas(product_id: str, request: Request):
    print("Getting personas for product_id:", product_id)
    """
    Returns a list of approved personas for the given product_id.
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
        print("Loading personas for product_id:", product_id)
        product_subgraph = build_product_graph(product_id)
        # Print all extracted nodes in product_subgraph
        aggregated_personas = []
        aggregated_personas = get_persona_relevance(product_subgraph)
        return aggregated_personas
    except Exception as e:
        print("❌ Get personas error:", e)
        raise HTTPException(status_code=500, detail="Could not retrieve personas")
    

# GET /get-zmot-icp/{product_id}
@router.get("/get-zmot-icp/{product_id}")
async def get_zmot_icp(product_id: str, request: Request):
    print("Getting ZMOT ICP for product_id:", product_id)
    """
    Returns the ZMOT ICP for the given product_id.
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
        zmot_icp = get_zmot_icp_relevance(product_subgraph)
        return zmot_icp
    except Exception as e:
        print("❌ Get ZMOT ICP error:", e)
        raise HTTPException(status_code=500, detail="Could not retrieve ZMOT ICP")

# GET /get-archetypes/{product_id}
@router.get("/get-archetypes/{product_id}")
async def get_archetypes(product_id: str, request: Request):
    print("Getting archetypes for product_id:", product_id)
    """
    Returns the archetypes for the given product_id.
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
        archetypes = get_archetypes_by_relevance(product_subgraph)
        print("Archetypes found:", archetypes)
        return archetypes
    except Exception as e:
        print("❌ Get Archetypes error:", e)
        raise HTTPException(status_code=500, detail="Could not retrieve Archetypes")

# GET /get-zmots-for-archetype/{product_id}
@router.get("/get-zmots-for-archetype/{product_id}")
async def get_zmots_for_archetype(product_id: str, archetype_id: str, request: Request):
    print("Getting ZMOTs for selected archetype for product_id:", product_id)
    """
    Returns the ZMOTs for the given product_id and archetype_id.
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
        zmots_for_archetype = get_best_zmots_for_archetype(product_subgraph, archetype_id=archetype_id)
        if not get_zmots_for_archetype:
            raise HTTPException(status_code=404, detail="No ZMOTs found for the given archetype ID")
        print("ZMOTs found:", zmots_for_archetype)
        return zmots_for_archetype
    except Exception as e:
        print("❌ Get ZMOTs error:", e)
        raise HTTPException(status_code=500, detail="Could not retrieve ZMOTs")

#Post /get-reverse-case-study/{product_id}
@router.post("/get-reverse-case-study/{product_id}")
async def get_reverse_case_study(product_id: str, payload:dict, request: Request):
    print("Generating reverse case studies for product_id:", product_id)
    """
    Returns a list of reverse case studies for the product_id.
    Assumes initial traversal is complete
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
        archetype_id = payload.get("archetype_id")
        if not archetype_id:
            raise HTTPException(status_code=400, detail="Archetype ID is required.")
        zmot_event = payload.get("zmot_event_id", None)
        engagement_meta = payload.get("engagement_meta", None)
        print("Calling RCS Simulator")
        reverse_case_study = simulate_rcs(
            product_subgraph,
            archetype_id,
            zmot_event
        )

        return reverse_case_study
    except Exception as e:
        print("❌ Get Reverse Case Studies error:", e)
        raise HTTPException(status_code=500, detail="Could not retrieve Reverse Case Studies")
    
    