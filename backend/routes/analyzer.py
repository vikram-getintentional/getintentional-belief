from fastapi import APIRouter, Depends, Request, HTTPException
from backend.utils.openai_helper import extract_summary_and_capabilities
from backend.utils.openai_helper_core import infer_with_rules_then_fallback
from backend.auth.jwt_handler import decode_token
from backend.database import SessionLocal
from sqlalchemy.orm import Session
from fastapi import Depends
from backend.database import get_db
from backend.db.schema_templates.save_company_value_prop import save_company_value_prop, get_company_value_prop
from backend.utils.graph_base.graph import Graph
from backend.utils.inference.hop_plus_agent import infer_upstream_with_rules


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

    existing = get_company_value_prop(db, company_id)
    if existing:
        return {
            "summary": existing["summary"],
            "capabilities": existing["capabilities"]
        }
    
    result = extract_summary_and_capabilities(text, plg_cta, footer_features)

    if "summary" in result:
        summary = result["summary"]
        capabilities = result.get("capabilities", [])
        print("DB Saving Stuff:","\n","company id: ", company_id, "\n","url: ", url, "\n","summary: ", summary, "\n","capabilities: ", capabilities)
         # 💾 Save value prop for company
        save_company_value_prop(db, company_id, url, summary, capabilities)

    return result

# 🔬 Deep persona + pain inference
@router.post("/analyze/deep")
async def analyze_deep(payload: dict, request: Request):
    url = payload.get("url")
    summary = payload.get("summary")
    capabilities = payload.get("capabilities", [])

    if not summary or not capabilities:
        raise HTTPException(status_code=400, detail="Summary and capabilities required.")

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
        result = infer_with_rules_then_fallback(summary, capabilities)
        # 💾 Save value prop for company
        db = SessionLocal()
        save_company_value_prop(db, company_id, url, summary, capabilities)

        return result

    except Exception as e:
        print("❌ Deep inference error:", e)
        raise HTTPException(status_code=500, detail="Could not run deep analysis")

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
        hop_0_graph = payload.get("results", [])
        hop_plus_graph = Graph()
        hop_plus_results = infer_upstream_with_rules(
            base_nodes=hop_0_graph,
            graph=hop_plus_graph,
            max_depth=3
        )

        print("results in analyze_hop_plus:", hop_plus_results)
        return {"hop_plus_results": hop_plus_results}

    except Exception as e:
        print("❌ Hop+ analysis error:", e)
        raise HTTPException(status_code=500, detail="Could not run Hop+ analysis")