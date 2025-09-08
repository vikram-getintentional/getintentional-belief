from fastapi import APIRouter, Depends, HTTPException, Request
from typing import List, Dict, Any

from backend.auth.jwt_handler import decode_token
from backend.database import get_db
from sqlalchemy.orm import Session

import networkx as nx

from backend.utils.graph_base.network_graph import (
    build_product_graph,
    update_graph,
    get_node_by_id,
    get_node_id,
    get_nodes_list_ids,
    get_target_nodes_by_source_and_type,
    get_edge_attribute,
    get_edge_weight,
    get_product_id_from_subgraph,
)
from backend.utils.graph_base.agent_graph_builder import (
    _capability,
    _pain,
    _job,
    _persona,
    _upsert_edge,
)

router = APIRouter()


def _require_auth_company(request: Request) -> str:
    auth_header = request.headers.get("authorization")
    if not auth_header:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    token = auth_header.split(" ")[1]
    decoded = decode_token(token)
    company_id = decoded.get("company_id")
    if not company_id:
        raise HTTPException(status_code=401, detail="Invalid token or company ID not found")
    return company_id


def _norm_coreness(value: Any) -> str:
    # Keep string label as provided; FE can send {Critical|Core|Supportive|Ancillary}
    if value is None:
        return ""
    return str(value)


def _coreness_to_numeric(label: Any) -> float:
    table = {
        "Critical": 1.0,
        "Core": 0.8,
        "Supportive": 0.6,
        "Ancillary": 0.4,
    }
    return table.get(str(label), 0.0)


def _norm_likelihood(value: Any) -> float:
    try:
        v = float(value)
        return max(0.0, min(100.0, v))
    except Exception:
        return 0.0


@router.post("/graph/capability")
def create_capability(payload: Dict[str, Any], request: Request, db: Session = Depends(get_db)):
    _require_auth_company(request)
    product_id = payload.get("product_id")
    if not product_id:
        raise HTTPException(status_code=400, detail="product_id is required")
    name = payload.get("name", "")
    description = payload.get("description", "")
    coreness = _norm_coreness(payload.get("coreness"))
    buying = _norm_likelihood(payload.get("buying_likelihood"))

    G = build_product_graph(product_id)
    product_node_id = get_product_id_from_subgraph(G)
    cap_id = _capability(G, name=name, description=description)
    # Attach to product with edge attributes
    _upsert_edge(G, product_node_id, "offers", cap_id, weight=1.0, attrs={
        "coreness": coreness,
        "buying_likelihood": buying,
    })
    # Mirror coreness to node numeric for legacy views
    node = get_node_by_id(G, cap_id)
    if node is not None and coreness:
        node["coreness"] = _coreness_to_numeric(coreness)
    G = update_graph(G)
    node = get_node_by_id(G, cap_id)
    if not node:
        raise HTTPException(status_code=500, detail="Failed to create capability")
    return {
        "id": node.get("id"),
        "name": node.get("name", ""),
        "description": node.get("description", ""),
        "coreness": coreness,
        "buying_likelihood": buying,
    }


@router.put("/graph/capability/{capability_id}")
def update_capability(capability_id: str, payload: Dict[str, Any], request: Request, db: Session = Depends(get_db)):
    _require_auth_company(request)
    product_id = payload.get("product_id")
    if not product_id:
        raise HTTPException(status_code=400, detail="product_id is required")
    G = build_product_graph(product_id)
    node = get_node_by_id(G, capability_id)
    if not node:
        raise HTTPException(status_code=404, detail="Capability not found")

    # Update node attrs if provided
    for k in ("name", "description"):
        if payload.get(k) is not None:
            node[k] = payload.get(k)

    # Update edge attrs
    coreness = payload.get("coreness")
    buying = payload.get("buying_likelihood")
    attrs = {}
    if coreness is not None:
        attrs["coreness"] = _norm_coreness(coreness)
    if buying is not None:
        attrs["buying_likelihood"] = _norm_likelihood(buying)
    if attrs:
        product_node_id = get_product_id_from_subgraph(G)
        _upsert_edge(G, product_node_id, "offers", capability_id, weight=1.0, attrs=attrs)
    # Mirror coreness to node numeric for legacy views
    node = get_node_by_id(G, capability_id)
    if node is not None and "coreness" in attrs:
        node["coreness"] = _coreness_to_numeric(attrs["coreness"]) if attrs.get("coreness") else node.get("coreness", 0.0)

    G = update_graph(G)
    node = get_node_by_id(G, capability_id)
    return {
        "id": node.get("id"),
        "name": node.get("name", ""),
        "description": node.get("description", ""),
        "coreness": attrs.get("coreness"),
        "buying_likelihood": attrs.get("buying_likelihood"),
    }


@router.delete("/graph/capability/{capability_id}")
def delete_capability(capability_id: str, product_id: str, request: Request, db: Session = Depends(get_db)):
    _require_auth_company(request)
    if not product_id:
        raise HTTPException(status_code=400, detail="product_id is required")
    G = build_product_graph(product_id)
    if not get_node_by_id(G, capability_id):
        raise HTTPException(status_code=404, detail="Capability not found")
    G.remove_node(capability_id)
    update_graph(G)
    return {"message": "Capability deleted", "id": capability_id}


@router.post("/graph/capability/merge")
def merge_capabilities(payload: Dict[str, Any], request: Request, db: Session = Depends(get_db)):
    _require_auth_company(request)
    product_id = payload.get("product_id")
    source_id = payload.get("source_id")
    target_id = payload.get("target_id")
    if not all([product_id, source_id, target_id]):
        raise HTTPException(status_code=400, detail="product_id, source_id, target_id are required")
    if source_id == target_id:
        raise HTTPException(status_code=400, detail="source and target must differ")
    G = build_product_graph(product_id)
    if not get_node_by_id(G, source_id) or not get_node_by_id(G, target_id):
        raise HTTPException(status_code=404, detail="source or target not found")

    # Rewire predecessors → target
    for u in list(G.predecessors(source_id)):
        ed = G.get_edge_data(u, source_id) or {}
        attrs = {k: v for k, v in ed.items() if k not in ("type",)}
        _upsert_edge(G, u, ed.get("type", "related_to"), target_id, weight=ed.get("weight", 1.0), attrs=attrs)

    # Rewire successors ← target
    for v in list(G.successors(source_id)):
        ed = G.get_edge_data(source_id, v) or {}
        attrs = {k: v2 for k, v2 in ed.items() if k not in ("type",)}
        _upsert_edge(G, target_id, ed.get("type", "related_to"), v, weight=ed.get("weight", 1.0), attrs=attrs)

    G.remove_node(source_id)
    update_graph(G)
    return {"message": "Merged", "source": source_id, "target": target_id}


@router.get("/graph/nodes/{node_type}")
def list_nodes(node_type: str, product_id: str, request: Request, db: Session = Depends(get_db)):
    _require_auth_company(request)
    if node_type not in {"capability", "pain", "job", "persona"}:
        raise HTTPException(status_code=400, detail="Unsupported node type")
    G = build_product_graph(product_id)
    out = []
    for nid, data in G.nodes(data=True):
        if data.get("node_type") != node_type:
            continue
        label = data.get("name") or data.get("description") or data.get("title") or data.get("id")
        # Include common fields for UI
        include_keys = {"title", "department", "seniority", "description", "linkedin_url"}
        extras = {k: v for k, v in data.items() if k in include_keys}
        out.append({"id": nid, "label": label, **extras})
    return {"nodes": out}


def _relevance_to_weight(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    table = {
        "Critical": 1.0,
        "Core": 0.8,
        "Supportive": 0.6,
        "Ancillary": 0.4,
    }
    return table.get(str(value), 0.6)


@router.post("/graph/pain_family")
def create_pain_family(payload: Dict[str, Any], request: Request, db: Session = Depends(get_db)):
    _require_auth_company(request)
    product_id = payload.get("product_id")
    capability_id = payload.get("capability_id")
    if not all([product_id, capability_id]):
        raise HTTPException(status_code=400, detail="product_id and capability_id are required")

    pain_text = payload.get("pain")
    job_text = payload.get("job")
    persona = payload.get("persona", {}) or {}
    relevance = _relevance_to_weight(payload.get("relevance"))
    likelihood = _norm_likelihood(payload.get("likelihood"))

    G = build_product_graph(product_id)
    # Create or reuse
    pain_id = _pain(G, pain_text or "Unnamed pain", pain_source=None) if pain_text else payload.get("pain_id")
    if not pain_id:
        raise HTTPException(status_code=400, detail="Provide pain text or pain_id")
    job_id = _job(G, job_text or "Unnamed job") if job_text else payload.get("job_id")
    if not job_id:
        raise HTTPException(status_code=400, detail="Provide job text or job_id")
    if persona:
        per_id = _persona(G, persona.get("title", ""), persona.get("department", ""), persona.get("seniority", ""))
        if persona.get("linkedin_url"):
            G.nodes[per_id]["linkedin_url"] = str(persona.get("linkedin_url"))
    else:
        per_id = payload.get("persona_id")

    # Optional linkedin_url on job
    if job_text and payload.get("job_linkedin_url"):
        if job_id in G:
            G.nodes[job_id]["linkedin_url"] = str(payload.get("job_linkedin_url"))

    # Edges: capability→solves→pain; pain→felt_in→job; job→performed_by→persona
    _upsert_edge(G, capability_id, "solves", pain_id, weight=relevance, attrs={"relevance": relevance})
    _upsert_edge(G, pain_id, "felt_in", job_id, weight=1.0)
    if per_id:
        _upsert_edge(G, job_id, "performed_by", per_id, weight=likelihood, attrs={"likelihood": likelihood})

    update_graph(G)
    return {
        "pain_id": pain_id,
        "job_id": job_id,
        "persona_id": per_id,
        "relevance": relevance,
        "likelihood": likelihood,
    }


@router.put("/graph/pain_family/{_id}")
def update_pain_family(_id: str, payload: Dict[str, Any], request: Request, db: Session = Depends(get_db)):
    # Note: pain_family has no single node; update edge attributes by provided IDs
    _require_auth_company(request)
    product_id = payload.get("product_id")
    capability_id = payload.get("capability_id")
    pain_id = payload.get("pain_id")
    job_id = payload.get("job_id")
    persona_id = payload.get("persona_id")
    if not all([product_id, capability_id, pain_id, job_id]):
        raise HTTPException(status_code=400, detail="product_id, capability_id, pain_id, job_id required")
    G = build_product_graph(product_id)

    if payload.get("relevance") is not None:
        rel = _relevance_to_weight(payload.get("relevance"))
        _upsert_edge(G, capability_id, "solves", pain_id, weight=rel, attrs={"relevance": rel})
    if persona_id and payload.get("likelihood") is not None:
        lk = _norm_likelihood(payload.get("likelihood"))
        _upsert_edge(G, job_id, "performed_by", persona_id, weight=lk, attrs={"likelihood": lk})

    update_graph(G)
    return {"message": "Updated"}


@router.get("/graph/pain_families")
def list_pain_families(product_id: str, capability_id: str, request: Request, db: Session = Depends(get_db)):
    """Return pain families for a capability as triples (pain, job, persona) with edge attributes."""
    _require_auth_company(request)
    G = build_product_graph(product_id)
    out: list[dict[str, Any]] = []
    # capability --solves--> pain
    for pain_id in get_target_nodes_by_source_and_type(G, capability_id, "solves") or []:
        pn = get_node_by_id(G, pain_id) or {}
        rel = get_edge_attribute(G, capability_id, pain_id, "relevance") or get_edge_weight(G, capability_id, pain_id) or 0.0
        # pain --felt_in--> job
        jobs = get_target_nodes_by_source_and_type(G, pain_id, "felt_in") or []
        if not jobs:
            # still output a family with missing job to let UI fix it
            out.append({
                "id": f"{capability_id}|{pain_id}|",
                "capability_id": capability_id,
                "pain": {"id": pain_id, "text": pn.get("description") or pn.get("name") or pn.get("id")},
                "relevance": float(rel),
            })
            continue
        for job_id in jobs:
            jn = get_node_by_id(G, job_id) or {}
            # job --performed_by--> persona
            personas = get_target_nodes_by_source_and_type(G, job_id, "performed_by") or []
            if not personas:
                out.append({
                    "id": f"{capability_id}|{pain_id}|{job_id}|",
                    "capability_id": capability_id,
                    "pain": {"id": pain_id, "text": pn.get("description") or pn.get("name") or pn.get("id")},
                    "job": {"id": job_id, "text": jn.get("description") or jn.get("name") or jn.get("id")},
                    "relevance": float(rel),
                    "likelihood": 0.0,
                })
                continue
            for per_id in personas:
                per = get_node_by_id(G, per_id) or {}
                lk = get_edge_attribute(G, job_id, per_id, "likelihood") or get_edge_weight(G, job_id, per_id) or 0.0
                out.append({
                    "id": f"{capability_id}|{pain_id}|{job_id}|{per_id}",
                    "capability_id": capability_id,
                    "pain": {"id": pain_id, "text": pn.get("description") or pn.get("name") or pn.get("id")},
                    "job": {"id": job_id, "text": jn.get("description") or jn.get("name") or jn.get("id")},
                    "persona": {
                        "id": per_id,
                        "title": per.get("title"),
                        "department": per.get("department"),
                        "seniority": per.get("seniority"),
                    },
                    "relevance": float(rel),
                    "likelihood": float(lk),
                })
    return {"items": out}


@router.put("/graph/persona/{persona_id}")
def update_persona(persona_id: str, payload: Dict[str, Any], request: Request, db: Session = Depends(get_db)):
    _require_auth_company(request)
    product_id = payload.get("product_id")
    if not product_id:
        raise HTTPException(status_code=400, detail="product_id is required")
    G = build_product_graph(product_id)
    node = get_node_by_id(G, persona_id)
    if not node or node.get("node_type") != "persona":
        raise HTTPException(status_code=404, detail="Persona not found")
    for k in ("title", "department", "seniority", "linkedin_url"):
        if payload.get(k) is not None:
            node[k] = str(payload.get(k))
    update_graph(G)
    return {"message": "Persona updated", "id": persona_id, **{k: node.get(k) for k in ("title", "department", "seniority", "linkedin_url")}}


@router.put("/graph/job/{job_id}")
def update_job(job_id: str, payload: Dict[str, Any], request: Request, db: Session = Depends(get_db)):
    _require_auth_company(request)
    product_id = payload.get("product_id")
    if not product_id:
        raise HTTPException(status_code=400, detail="product_id is required")
    G = build_product_graph(product_id)
    node = get_node_by_id(G, job_id)
    if not node or node.get("node_type") != "job":
        raise HTTPException(status_code=404, detail="Job not found")
    if payload.get("linkedin_url") is not None:
        node["linkedin_url"] = str(payload.get("linkedin_url"))
    # Optional label update
    if payload.get("description") is not None:
        node["description"] = str(payload.get("description"))
    update_graph(G)
    return {"message": "Job updated", "id": job_id, "linkedin_url": node.get("linkedin_url"), "description": node.get("description")}
