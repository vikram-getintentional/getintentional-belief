import json
import os
import networkx as nx

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RCS_JSON_DIR = os.path.join(BASE_DIR, "graph_data", "rcs_json")

def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def _account_dir(product_id: str) -> str:
    d = os.path.join(RCS_JSON_DIR, str(product_id))
    _ensure_dir(d)
    return d

def _account_filename(account_id: str) -> str:
    return f"{account_id}_rcs.json"

def _account_path(product_id: str, account_id: str) -> str:
    return os.path.join(_account_dir(product_id), _account_filename(account_id))

def nx_to_dict(G):
    # Converts a networkx graph to a dict for JSON serialization
    return nx.node_link_data(G)

def attributes_key(attributes: dict) -> str:
    # Sort keys for determinism
    items = sorted(attributes.items())
    return "attrs=" + "|".join(f"{k}={v}" for k, v in items)

def save_rcs_as_json(product_id, attribute_dict, causal_graph, rcs_report, zmot_id=None):
    os.makedirs(RCS_JSON_DIR, exist_ok=True)
    filename = "rcs_output.json"
    full_path = os.path.join(RCS_JSON_DIR, filename)

    # Load existing data if present
    if os.path.exists(full_path):
        with open(full_path, "r") as f:
            all_data = json.load(f)
    else:
        all_data = {}

    # Prepare nested structure
    if product_id not in all_data:
        all_data[product_id] = {}
    attr_key = attributes_key(attribute_dict)
    if attr_key not in all_data[product_id]:
        all_data[product_id][attr_key] = {}

    zmot_key = f"zmot={zmot_id}" if zmot_id is not None else "zmot=none"
    all_data[product_id][attr_key][zmot_key] = {
        "causal_graph": nx_to_dict(causal_graph),
        "rcs_report": rcs_report
    }

    with open(full_path, "w") as f:
        json.dump(all_data, f, indent=2)
    print(f"RCS output for {product_id}/{attr_key}/{zmot_key} saved successfully to: {full_path}")


def load_rcs_from_json(product_id, attribute_dict, zmot_id=None):
    filename = "rcs_output.json"
    full_path = os.path.join(RCS_JSON_DIR, filename)
    if not os.path.exists(full_path):
        print(f"RCS file does not exist: {full_path}")
        return None
    with open(full_path, "r") as f:
        all_data = json.load(f)
    attr_key = attributes_key(attribute_dict)
    zmot_key = f"zmot={zmot_id}" if zmot_id is not None else "zmot=none"
    try:
        entry = all_data[product_id][attr_key][zmot_key]
        # Convert graph dict back to networkx graph if needed
        causal_graph = nx.node_link_graph(entry["causal_graph"])
        rcs_report = entry["rcs_report"]
        output = {"causal_graph": causal_graph, "rcs_report": rcs_report}
        return output
    except KeyError:
        print(f"No RCS output found for {product_id}/{attr_key}/{zmot_key}")
        return None
    

 # ── New account-centric API (preferred) ───────────────────────────────────────
def save_account_rcs_json(
    product_id: str,
    account_id: str,
    causal_graph: nx.Graph | None,
    rcs_report: dict,
) -> str:
    """
    Save one file per account:
      backend/utils/dev_environment/static_jsons/rcs_json/{product_id}/{account_id}_rcs.json
    Returns the full path written.
    """
    full_path = _account_path(product_id, account_id)
    payload = {
        "causal_graph": nx_to_dict(causal_graph or nx.DiGraph()),
        "rcs_report": rcs_report or {},
    }
    with open(full_path, "w") as f:
        json.dump(payload, f, indent=2)
    return full_path

def load_account_rcs_json(product_id: str, account_id: str) -> dict | None:
    """
    Try the new canonical file first. If missing, attempt a legacy fallback:
    - legacy per-account file: graph_data/rcs_json/{product_id}/{account_id}.json
    - legacy monolithic file via load_rcs_from_json (attrs={"account_id": ...})
    """
    # 1) New canonical
    new_path = _account_path(product_id, account_id)
    if os.path.exists(new_path):
        with open(new_path, "r") as f:
            data = json.load(f)
        # Reinflate graph to networkx for parity with older callers (if needed)
        try:
            data["causal_graph"] = nx.node_link_graph(data.get("causal_graph", {}))
        except Exception:
            data["causal_graph"] = nx.DiGraph()
        return data

    return None