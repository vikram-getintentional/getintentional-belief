import uuid
import json
from backend.utils.graph_base.graph_utils.json_store import load_json, save_json

ZMOT_PATH = "backend/utils/graph_base/graph_data/zmot_nodes.json"

def get_or_create_trigger_event_node(trigger_event, return_created=False):
    data = load_json(ZMOT_PATH)
    if not isinstance(data, dict):
        data = {"TriggerEvents": [], "ObservableMoments": [], "Keywords": []}
    if "TriggerEvents" not in data:
        data["TriggerEvents"] = []

    # Normalize for matching
    trigger_event = trigger_event.strip().lower()

    for node in data["TriggerEvents"]:
        if node.get("trigger_event", "").strip().lower() == trigger_event:
            return (node, False) if return_created else node

    new_node = {
        "id": str(uuid.uuid4()),
        "trigger_event": trigger_event
    }

    data["TriggerEvents"].append(new_node)
    save_json(ZMOT_PATH, data)
    return (new_node, True) if return_created else new_node

def get_or_create_observable_moment_node(observable_moment, return_created=False):
    data = load_json(ZMOT_PATH)
    if not isinstance(data, dict):
        data = {"TriggerEvents": [], "ObservableMoments": [], "Keywords": []}
    if "ObservableMoments" not in data:
        data["ObservableMoments"] = []

    # Normalize for matching
    observable_moment = observable_moment.strip().lower()

    for node in data["ObservableMoments"]:
        if node.get("observable_moment", "").strip().lower() == observable_moment:
            return (node, False) if return_created else node

    new_node = {
        "id": str(uuid.uuid4()),
        "observable_moment": observable_moment
    }

    data["ObservableMoments"].append(new_node)
    save_json(ZMOT_PATH, data)
    return (new_node, True) if return_created else new_node

def get_or_create_keyword_node(keyword, return_created=False):
    data = load_json(ZMOT_PATH)
    if not isinstance(data, dict):
        data = {"TriggerEvents": [], "ObservableMoments": [], "Keywords": []}
    if "Keywords" not in data:
        data["Keywords"] = []

    # Normalize for matching
    keyword = keyword.strip().lower()

    for node in data["Keywords"]:
        if node.get("keyword", "").strip().lower() == keyword:
            return (node, False) if return_created else node

    new_node = {
        "id": str(uuid.uuid4()),
        "keyword": keyword
    }

    data["Keywords"].append(new_node)
    save_json(ZMOT_PATH, data)
    return (new_node, True) if return_created else new_node