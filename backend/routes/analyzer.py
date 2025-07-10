from fastapi import APIRouter, Depends, Request, HTTPException
from backend.utils.openai_helper import extract_summary_and_capabilities
from backend.utils.openai_helper_core import infer_with_rules_then_fallback
from backend.auth.jwt_handler import decode_token
from backend.database import SessionLocal
from sqlalchemy.orm import Session
from backend.database import get_db
from backend.db.schema_templates.save_company_value_prop import save_company_value_prop, get_company_value_prop
from backend.utils.graph_base.graph import Graph
from backend.utils.dev_environment.graph_loader import load_graph_from_folder
from backend.utils.knowledge_base.value_prop_analysis import get_product_value_prop, process_capabilities
from backend.utils.graph_base.nodes.product_nodes import get_or_create_product_node
from backend.utils.graph_base.nodes.capability_nodes import (
    get_or_create_capability_node,
    add_capabilities_to_product,
    update_capabilities_by_node_id
)
from backend.utils.knowledge_base.persona_generation import (
    get_company_products,
    get_persona_relevance
)

from backend.utils.inference.hop_plus_agent import infer_upstream_with_rules

# Hardcoded path to graph data folder - to be updated in production

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GRAPH_DATA_PATH = os.path.join(BASE_DIR,"backend", "utils", "graph_base", "graph_data")

router = APIRouter()

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

    # 🧠 Load graph
    node_registry, graph_edges, edge_weights, cumulative_relevance = load_graph_from_folder(GRAPH_DATA_PATH)
    base_graph = Graph(
        node_registry=node_registry,
        graph_edges=graph_edges,
        edge_weights=edge_weights,
        
    )

    # Call the core logic function
    try:
        result = get_product_value_prop(company_id, base_graph, url, text, plg_cta, footer_features)
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

        # 🧠 Load graph
        node_registry, graph_edges, edge_weights, cumulative_relevance = load_graph_from_folder(GRAPH_DATA_PATH)
        base_graph = Graph(
            node_registry=node_registry,
            graph_edges=graph_edges,
            edge_weights=edge_weights,
            
        )
        product_node = base_graph.get_node_by_id(product_id)
        if not product_node:
            raise HTTPException(status_code=404, detail="Product node not found.")

        added_capabilities = add_capabilities_to_product(base_graph, product_node, capabilities)
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

        # 🧠 Load graph
        node_registry, graph_edges, edge_weights, cumulative_relevance = load_graph_from_folder(GRAPH_DATA_PATH)
        base_graph = Graph(
            node_registry=node_registry,
            graph_edges=graph_edges,
            edge_weights=edge_weights,
            
        )

        # Extract the product subgraph
        product_subgraph = base_graph.extract_product_subgraph(product_id)
        print("Extracted product subgraph with total nodes:", len(product_subgraph.node_registry))
        print("Starting hop0 inference")
        hop_0_results = infer_with_rules_then_fallback(product_id, product_subgraph)
        # Run Hop+ upstream inference
        #hop_plus_results = infer_upstream_with_rules(product_subgraph)
        return {"hop_0_results": hop_0_results}

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

        # 🧠 Load graph
        node_registry, graph_edges, edge_weights, cumulative_relevance = load_graph_from_folder(GRAPH_DATA_PATH)
        base_graph = Graph(
            node_registry=node_registry,
            graph_edges=graph_edges,
            edge_weights=edge_weights,
            
        )

        products = get_company_products(base_graph, company_id)
        # products should be a list of dicts: [{id, name}, ...]
        print("Products found:", products)
        return {"products": products}
    except Exception as e:
        print("❌ Get products error:", e)
        raise HTTPException(status_code=500, detail="Could not retrieve products")

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

        # 🧠 Load graph
        node_registry, graph_edges, edge_weights, cumulative_relevance = load_graph_from_folder(GRAPH_DATA_PATH)
        base_graph = Graph(
            node_registry=node_registry,
            graph_edges=graph_edges,
            edge_weights=edge_weights,
            
        )
        aggregated_personas = []
        print("Loading personas for product_id:", product_id)
        product_subgraph = base_graph.extract_product_subgraph(product_id)
        # Print all extracted nodes in product_subgraph
        print("Extracted product subgraph with total nodes:", len(product_subgraph.node_registry))
        for node_id, node_data in product_subgraph.node_registry.items():
            print(f"Node ID: {node_id}, Data: {node_data}")
        
        aggregated_personas = get_persona_relevance(product_subgraph)

        #personas = get_product_personas(base_graph, product_id)
        
        
        # personas should be a list of persona dicts
        #if personas is None:
         #   print("No personas found for product_id from get_product_personas:", product_id)
         #   return {"detail": "No Summaries or Capabilities Mapped"}
        #print("Personas from get_product_personas:", personas)
        return aggregated_personas
    except Exception as e:
        print("❌ Get personas error:", e)
        raise HTTPException(status_code=500, detail="Could not retrieve personas")

# 🪄 Hop+ recursive dependency analysis
@router.post("/analyze/hop_plus")
async def analyze_hop_plus(payload: dict, request: Request):
    """
    Expects payload with:
    - results: list of {persona, job, pain, capability, relevance}
    """
    print("Analyze Hop+ payload:", payload)
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

        db = SessionLocal()
        product_id = payload.get("product_id")
        if not product_id:
            raise HTTPException(status_code=400, detail="Product ID required.")
        print("Starting Hop0 graph load")
        node_registry, graph_edges, edge_weights, cumulative_relevance = load_graph_from_folder(GRAPH_DATA_PATH)

        base_graph = Graph(
            node_registry=node_registry,
            graph_edges=graph_edges,
            edge_weights=edge_weights,
            
        )

        product_subgraph = base_graph.extract_product_subgraph(product_id)
        
        hop_plus_results = infer_upstream_with_rules(
            product_subgraph=product_subgraph,
            cap_threshold = 0.3,
            relevance_threshold = 0.6,
            max_depth=3
        )

        print("results in analyze_hop_plus:", hop_plus_results)
        return {"hop_plus_results": hop_plus_results}

    except Exception as e:
        print("❌ Hop+ analysis error:", e)
        raise HTTPException(status_code=500, detail="Could not run Hop+ analysis")
