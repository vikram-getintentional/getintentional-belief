import uuid
import json
from backend.utils.graph_base.graph_utils.json_store import load_json, save_json

COMPANY_PATH = "backend/utils/graph_base/graph_data/company_nodes.json"
ARCHETYPE_PATH = "backend/utils/graph_base/graph_data/archetype_nodes.json"

def get_or_create_industry_node(industry, return_created=False):
    data = load_json(COMPANY_PATH)
    if not isinstance(data, dict):
        data = {"industry": [], "revenue": [], "employees": [], "funding_stage": [], "geography": []}
    if "industry" not in data:
        data["industry"] = []

    # Normalize for matching
    industry = industry.strip().lower()

    for node in data["industry"]:
        if node.get("industry", "").strip().lower() == industry:
            return (node, False) if return_created else node

    new_node = {
        "id": str(uuid.uuid4()),
        "industry": industry
    }


    data["industry"].append(new_node)
    save_json(COMPANY_PATH, data)
    return (new_node, True) if return_created else new_node

def get_or_create_revenue_node(revenue, return_created=False):
    data = load_json(COMPANY_PATH)
    if not isinstance(data, dict):
        data = {"industry": [], "revenue": [], "employees": [], "funding_stage": [], "geography": []}
    if "revenue" not in data:
        data["revenue"] = []

    # Normalize for matching
    revenue = revenue.strip().lower()

    for node in data["revenue"]:
        if node.get("revenue", "").strip().lower() == revenue:
            return (node, False) if return_created else node

    new_node = {
        "id": str(uuid.uuid4()),
        "revenue": revenue
    }

    data["revenue"].append(new_node)
    save_json(COMPANY_PATH, data)
    return (new_node, True) if return_created else new_node

def get_or_create_employees_node(employees, return_created=False):
    data = load_json(COMPANY_PATH)
    if not isinstance(data, dict):
        data = {"industry": [], "revenue": [], "employees": [], "funding_stage": [], "geography": []}
    if "employees" not in data:
        data["employees"] = []

    # Normalize for matching
    employees = employees.strip().lower()

    for node in data["employees"]:
        if node.get("employees", "").strip().lower() == employees:
            return (node, False) if return_created else node

    new_node = {
        "id": str(uuid.uuid4()),
        "employees": employees
    }

    data["employees"].append(new_node)
    save_json(COMPANY_PATH, data)
    return (new_node, True) if return_created else new_node

def get_or_create_funding_stage_node(funding_stage, return_created=False):
    data = load_json(COMPANY_PATH)
    if not isinstance(data, dict):
        data = {"industry": [], "revenue": [], "employees": [], "funding_stage": [], "geography": []}
    if "funding_stage" not in data:
        data["funding_stage"] = []

    # Normalize for matching
    funding_stage = funding_stage.strip().lower()

    for node in data["funding_stage"]:
        if node.get("funding_stage", "").strip().lower() == funding_stage:
            return (node, False) if return_created else node

    new_node = {
        "id": str(uuid.uuid4()),
        "funding_stage": funding_stage
    }

    data["funding_stage"].append(new_node)
    save_json(COMPANY_PATH, data)
    return (new_node, True) if return_created else new_node

def get_or_create_geographies_node(geography, return_created=False):
    data = load_json(COMPANY_PATH)
    if not isinstance(data, dict):
        data = {"industry": [], "revenue": [], "employees": [], "funding_stage": [], "geography": []}
    if "geography" not in data:
        data["geography"] = []

    # Normalize for matching
    geography = geography.strip().lower()

    for node in data["geography"]:
        if node.get("geography", "").strip().lower() == geography:
            return (node, False) if return_created else node

    new_node = {
        "id": str(uuid.uuid4()),
        "geography": geography
    }

    data["geography"].append(new_node)
    save_json(COMPANY_PATH, data)
    return (new_node, True) if return_created else new_node


def get_or_create_archetype_node(industry=None, revenue_range=None, employee_range=None, funding_stage=None, geography=None, return_created=False): 
    data = load_json(ARCHETYPE_PATH)

    # Create a new Archetype node
    new_node = {
        "id": str(uuid.uuid4()),
        "industry": industry,
        "revenue_range": revenue_range,
        "employee_range": employee_range,
        "funding_stage": funding_stage,
        "geography": geography
    }
    data.append(new_node)
    save_json(ARCHETYPE_PATH, data)
    return (new_node, True) if return_created else new_node