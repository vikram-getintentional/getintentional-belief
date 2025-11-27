# backend/integrations/hubspot_ingest.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional
import os
import requests
from datetime import datetime, timezone

HS_BASE = "https://api.hubapi.com"
HS_TOKEN = os.getenv("HUBSPOT_PRIVATE_APP_TOKEN", "")

class HS:
    def __init__(self, token: Optional[str] = None):
        self.token = token or HS_TOKEN
        if not self.token:
            raise RuntimeError("HUBSPOT_PRIVATE_APP_TOKEN not set")

    def _get(self, path: str, params=None):
        r = requests.get(
            f"{HS_BASE}{path}",
            headers={"Authorization": f"Bearer {self.token}"},
            params=params or {},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, json=None):
        r = requests.post(
            f"{HS_BASE}{path}",
            headers={"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"},
            json=json or {},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()

    # --- Companies ---
    def find_company_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        # /crm/v3/objects/companies/search
        body = {
            "filterGroups": [{
                "filters": [{
                    "propertyName": "name",
                    "operator": "EQ",
                    "value": name
                }]
            }],
            "properties": ["name", "domain", "hs_object_id"],
            "limit": 1
        }
        data = self._post("/crm/v3/objects/companies/search", json=body)
        results = data.get("results") or []
        return results[0] if results else None

    def get_company_contacts(self, company_id: str) -> Dict[str, Dict[str, Any]]:
        # list associations: /crm/v4/objects/companies/{companyId}/associations/contacts
        assoc = self._get(f"/crm/v4/objects/companies/{company_id}/associations/contacts")
        ids = [a["toObjectId"] for a in (assoc.get("results") or []) if a.get("associationTypes")]
        if not ids:
            return {}
        contacts = self.batch_read_contacts(ids)
        return {c["id"]: c for c in contacts}

    def batch_read_contacts(self, contact_ids: List[str]) -> List[Dict[str, Any]]:
        # /crm/v3/objects/contacts/batch/read
        body = {
            "properties": ["firstname","lastname","email","jobtitle","department","hs_lead_status","lifecyclestage"],
            "inputs": [{"id": cid} for cid in contact_ids]
        }
        data = self._post("/crm/v3/objects/contacts/batch/read", json=body)
        return data.get("results") or []

    # --- Activities (calls/emails/meetings/notes) ---
    def get_company_activities(self, company_id: str) -> List[Dict[str, Any]]:
        # You can either:
        #  A) query Unified Timeline API (beta) OR
        #  B) pull per-object types: calls, emails, meetings, notes and union.
        # Below is a simple example for calls+emails; extend as needed.
        activities: List[Dict[str, Any]] = []
        activities += self._paged("/crm/v3/objects/calls", params={"associations": "companies"})
        activities += self._paged("/crm/v3/objects/emails", params={"associations": "companies"})
        activities += self._paged("/crm/v3/objects/meetings", params={"associations": "companies"})
        activities += self._paged("/crm/v3/objects/notes", params={"associations": "companies"})
        # Filter to those associated with this company
        out = []
        for a in activities:
            for assoc in a.get("associations", {}).get("companies", {}).get("results", []):
                if str(assoc.get("id")) == str(company_id):
                    out.append(a)
                    break
        return out

    def _paged(self, path: str, params=None) -> List[Dict[str, Any]]:
        items = []
        after = None
        while True:
            q = dict(params or {})
            if after:
                q["after"] = after
            data = self._get(path, params=q)
            items.extend(data.get("results") or [])
            paging = data.get("paging", {})
            next_link = paging.get("next", {}).get("after")
            if not next_link:
                break
            after = next_link
        return items

# --------- mapping into your Engagement type ----------

def _infer_seniority_from_title(title: str) -> str:
    t = (title or "").lower()
    if any(k in t for k in ["chief", "cxo", "cfo", "ceo", "coo", "cto", "cmo", "vp", "vice president", "head", "director"]):
        return "Executive"
    if any(k in t for k in ["manager", "lead", "owner"]):
        return "Manager"
    if "senior" in t or "sr" in t:
        return "Senior"
    return "Operator"

def _iso_z(ms_or_iso: Any) -> str:
    # HubSpot often sends milliseconds since epoch; or ISO
    if isinstance(ms_or_iso, (int, float)):
        return datetime.fromtimestamp(ms_or_iso/1000.0, tz=timezone.utc).replace(microsecond=0).isoformat().replace("+00:00","Z")
    try:
        return datetime.fromisoformat(str(ms_or_iso).replace("Z","+00:00")).astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00","Z")
    except Exception:
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00","Z")

def normalize_hs_activity(a: Dict[str, Any], contacts_by_id: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    obj_type = (a.get("properties", {}).get("hs_object_type") or a.get("objectType") or "").upper()
    if not obj_type:
        # Fallback: infer from endpoint path in a higher-level wrapper if needed
        obj_type = "ACTIVITY"

    # Who engaged → pick first associated contact (if any)
    contact_id = None
    assoc_contacts = a.get("associations", {}).get("contacts", {}).get("results", [])
    if assoc_contacts:
        contact_id = str(assoc_contacts[0].get("id"))

    contact = contacts_by_id.get(contact_id) if contact_id else None
    props = contact.get("properties", {}) if contact else {}

    firstname = props.get("firstname") or ""
    lastname = props.get("lastname") or ""
    actor_name = " ".join([firstname, lastname]).strip() or None
    actor_role = (props.get("jobtitle") or "").strip()
    actor_dept = (props.get("department") or "").strip()
    actor_snr = _infer_seniority_from_title(actor_role)

    # Timestamp/subject
    p = a.get("properties", {})
    subject = p.get("hs_email_subject") or p.get("hs_call_title") or p.get("hs_meeting_title") or p.get("hs_note_body") or ""
    when = p.get("hs_timestamp") or p.get("hs_meeting_start_time") or p.get("hs_call_callee_object_id")  # many variants
    ts = _iso_z(when or p.get("createdate"))

    eng_type = "EMAIL" if "email" in obj_type.lower() else "CALL" if "call" in obj_type.lower() else "MEETING" if "meeting" in obj_type.lower() else "NOTE"

    return {
        "timestamp": ts,
        "actor": {
            "name": actor_name,
            "role": actor_role,
            "department": actor_dept,
            "seniority": actor_snr,
            "confidence": 1.0 if actor_name else 0.7,  # heuristic
        },
        "channel": eng_type,
        "source": "sales",  # HubSpot activities → usually sales-side
        "raw_activity": subject or eng_type.title(),
        "asset_id": None,
        "inferred": False,
    }

def ingest_hubspot_company(name: str) -> Dict[str, Any]:
    """
    Returns: { company: {...}, contacts: {...}, engagements: [...] }
    engagements are normalized to your internal Engagement shape.
    """
    hs = HS()
    company = hs.find_company_by_name(name)
    if not company:
        return {"company": None, "contacts": {}, "engagements": []}

    company_id = str(company["id"])
    contacts_by_id = hs.get_company_contacts(company_id)
    activities = hs.get_company_activities(company_id)

    out: List[Dict[str, Any]] = []
    for a in activities:
        row = normalize_hs_activity(a, contacts_by_id)
        if row:
            out.append(row)

    # sort earliest → latest
    out.sort(key=lambda x: x["timestamp"])
    return {"company": company, "contacts": contacts_by_id, "engagements": out}
