from datetime import datetime
from typing import Any, Dict, List, Optional
from backend.database import get_db, Base
from backend.utils.crm_management.target_account_models_dto import TargetAccount
from sqlalchemy.orm import Session
from sqlalchemy import Column, String, Integer, JSON, func
import uuid
import networkx as nx

from backend.utils.graph_base.network_graph import build_product_graph

"""
For the list of account data provided this function looks in the target_list_db if an existing record is found, else creates one.
1. For this we need to first create the target_list_db in the database
DB has columns: Product_ID | Target Account ID (uuid generated) | Target Account | Revenue | Employees | Funding | Industry | Geography
2. For the given product ID check if the target account already exists - else add to DB

"""

class TargetAccount(Base):
    __tablename__ = "target_accounts"

    id = Column(String, primary_key=True, index=True)
    product_id = Column(String, index=True)
    account_name = Column(String, index=True)
    industry = Column(String)
    revenue_range = Column(String)
    employee_range = Column(String)
    funding_stage = Column(String)
    geography = Column(String)
    competitor_used = Column(JSON, default=[])
    other_tech_stack = Column(JSON, default=[])
    deal_status = Column(String, default="New")  # e.g., New, In-Progress, Closed-Won, Closed-Lost

# ---------------- RCS Mapping Table ----------------
class TargetAccountRCS(Base):
    """
    Each row maps a TargetAccount → RCS JSON
    """
    __tablename__ = "target_account_rcs"

    id = Column(String, primary_key=True, index=True)
    target_account_id = Column(String, index=True)
    product_id = Column(String, index=True)
    account_name = Column(String, index=True)
    rcs_json = Column(JSON)
    created_at = Column(String, default=lambda: datetime.now(datetime.timezone.utc).isoformat())

def save_and_update_target_accounts(accounts, product_id):
    
    db: Session = next(get_db())

    for account in accounts:
        print("Processing account:", account)
        account_name = account.get("account_name")
        industry = account.get("industry")
        revenue_range = account.get("revenue_range") or account.get("revenue")
        employee_range = account.get("employee_range") or account.get("employees")
        funding_stage = account.get("funding_stage") or account.get("funding")
        geography = account.get("geography") 
        competitor_used = account.get("competitor_used", [])
        other_tech_stack = account.get("other_tech_stack", [])
        deal_status = account.get("deal_status", "New")


        if not account_name:
            print(f"Skipping account with missing name: {account}")
            continue

        # Check if the target account already exists for the given product_id
        existing_account = db.query(TargetAccount).filter_by(
            product_id=product_id,
            account_name=account_name,
        ).first()

        if existing_account:
            # Update existing record
            existing_account.industry = industry
            existing_account.revenue_range = revenue_range
            existing_account.employee_range = employee_range
            existing_account.funding_stage = funding_stage
            existing_account.geography = geography
            existing_account.competitor_used = competitor_used
            existing_account.other_tech_stack = other_tech_stack
            existing_account.deal_status = deal_status
            print(f"Updated existing account: {account_name}")
        else:
            # Create new record
            new_account = TargetAccount(
                id=str(uuid.uuid4()),
                product_id=product_id,
                account_name=account_name,
                industry=industry,
                revenue_range=revenue_range,
                employee_range=employee_range,
                funding_stage=funding_stage,
                geography=geography,
                competitor_used=competitor_used,
                other_tech_stack=other_tech_stack,
                deal_status=deal_status
            )
            db.add(new_account)
            print(f"Added new account: {account_name}")

    db.commit()
    db.close()

def load_target_accounts_from_db(product_id):
    db: Session = next(get_db())
    accounts = db.query(TargetAccount).filter_by(product_id=product_id).all()
    db.close()
    # Serialize for API response
    return [account_to_dict(a) for a in accounts]

def account_to_dict(account):
    return {
        "id": account.id,
        "product_id": account.product_id,
        "account_name": account.account_name,
        "industry": account.industry,
        "revenue_range": account.revenue_range,
        "employee_range": account.employee_range,
        "funding_stage": account.funding_stage,
        "geography": account.geography,
        "competitor_used": account.competitor_used or [],
        "other_tech_stack": account.other_tech_stack or [],
        "deal_status": account.deal_status,
    }
def delete_target_account_handler(account_id, product_id):
    print("Deletion loop inside handler")
    db: Session = next(get_db())
    all_accounts = db.query(TargetAccount).filter_by(product_id=product_id).all()
    print("Available account IDs for product:", [a.id for a in all_accounts])
    account = db.query(TargetAccount).filter_by(id=account_id, product_id=product_id).first()
    if not account:
        db.close()
        print("No account found to delete")
        return False
    db.delete(account)
    db.commit()
    db.close()
    return True

#--- Logic to fetch target account ids with filters ----


def get_target_account_ids(product_id: str, filters: dict = None):
    """
    Returns a list of account ids for a given product_id applying filters.

    Notes:
    - Accepts filter keys "deal_status" or the alias "status".
    - Supports two input shapes:
        * tuple form: { "deal_status": ("!in", ["Closed-Won", "Closed-Lost"]) }
        * dict form:  { "status": { "nin": ["Closed-won", "Closed-lost"] } }
    - Matching is case-insensitive.
    """
    print("running tgt acct getter")
    db: Session = next(get_db())
    try:
        print("Fetching target accounts for product_id:", product_id)
        q = db.query(TargetAccount).filter(TargetAccount.product_id == product_id)
        

        if filters:
            for key, condition in filters.items():
                # normalize alias -> deal_status
                if key in ("deal_status", "status"):
                    # tuple form: (op, values)
                    if isinstance(condition, (list, tuple)) and len(condition) == 2:
                        op, values = condition
                        vals = [v.lower() for v in (values or [])]
                        if op in ("!in", "nin"):
                            q = q.filter(~func.lower(TargetAccount.deal_status).in_(vals))
                        elif op in ("in",):
                            q = q.filter(func.lower(TargetAccount.deal_status).in_(vals))
                        elif op in ("=", "eq"):
                            q = q.filter(func.lower(TargetAccount.deal_status) == (str(values).lower()))
                    # dict form: {"nin": [...]} or {"in": [...]} or {"=": "value"}
                    elif isinstance(condition, dict):
                        if "nin" in condition:
                            vals = [v.lower() for v in (condition.get("nin") or [])]
                            q = q.filter(~func.lower(TargetAccount.deal_status).in_(vals))
                        if "in" in condition:
                            vals = [v.lower() for v in (condition.get("in") or [])]
                            q = q.filter(func.lower(TargetAccount.deal_status).in_(vals))
                        if "=" in condition or "eq" in condition:
                            v = condition.get("=") or condition.get("eq")
                            q = q.filter(func.lower(TargetAccount.deal_status) == (str(v).lower()))
                    else:
                        # unknown shape — ignore but warn
                        print(f"[WARN] Unhandled filter shape for {key}: {condition}")

        rows = q.all()
        # serialize into ids (not ORM objects)
        target_account_ids = [r.id for r in rows]
        print("Fetched target account IDs:", target_account_ids)
        return target_account_ids
    finally:
        db.close()

def get_account_by_id(product_id: str, account_id: str) -> Optional[Dict[str, Any]]:
    """
    Fetch a single TargetAccount record by product_id and account_id.
    Returns a fully serialized dictionary if found, else None.
    """
    db: Session = next(get_db())
    try:
        account = (
            db.query(TargetAccount)
            .filter(
                TargetAccount.product_id == product_id,
                TargetAccount.id == account_id,
            )
            .first()
        )
        if not account:
            return None
        return account_to_dict(account)
    except Exception as e:
        print(f"[ERROR] Failed to fetch account {account_id}: {e}")
        return None
    finally:
        db.close()



#--------------------------------------------
# Logic to get account metadata and map to nodes in graph
#--------------------------------------------
def _norm_token(s: str) -> str:
    return str(s).strip().lower().replace(" ", "_").replace("/", "_").replace("&", "and")

def map_account_meta_to_stable_ids(product_id: str, account_meta: Dict[str, Any] | None) -> List[str]:
    """
    Map account metadata to stable, reusable string IDs (sidecar-only, not added to graph).
    Example outputs:
      attr:industry:saas
      attr:revenue_range:200m_1b
      attr:employee_range:5k_10k
      attr:funding_stage:public
      attr:geography:north_america
      comp:stripe
      tech:snowflake
    """
    print("Mapping account meta to stable IDs for product:", product_id)
    if account_meta is None:
        # If caller passed None, fetch from DB
        account_meta = get_account_by_id(product_id, account_meta)  # defensive; no-op pattern

    # If still None or shape unknown, bail gracefully
    if not account_meta:
        return []

    ids: List[str] = []

    # Attributes
    attrs = {
        "industry": account_meta.get("industry"),
        "revenue_range": account_meta.get("revenue_range"),
        "employee_range": account_meta.get("employee_range"),
        "funding_stage": account_meta.get("funding_stage"),
        "geography": account_meta.get("geography"),
    }
    for k, v in attrs.items():
        if not v:
            continue
        val = _norm_token(str(v))
        # small cleanup for ranges like "$200M-$1B" -> "200m_1b"
        val = (
            val.replace("$", "")
               .replace("-", "_")
               .replace("__", "_")
        )
        ids.append(f"attr:{_norm_token(k)}:{val}")

    # Competitors used
    for c in (account_meta.get("competitor_used") or []):
        ids.append(f"comp:{_norm_token(c)}")

    # Tech stack
    for t in (account_meta.get("other_tech_stack") or []):
        ids.append(f"tech:{_norm_token(t)}")
    print("Mapped account meta to stable IDs:", ids)

    return ids