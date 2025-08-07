
import networkx as nx

"""
we now have a beautiful loop in our graph like: 
source - node type - target
product node - is_icp - archetype node
archetype node - responds_to - zmot trigger event

And then we have
job_id - triggered_by - zmot  node
zmot node - observed_in - observable moment node
zmot node - associated_with - keyword node

And we already have
product node - offered_by - capability node
capability node - solves - pain node
pain node - adresses - job node
job node - performed_by - pain node

job node - solves - pain node

pain node - scales_with - pain_trigger node
pain node - expressed_as - perceived_metric node

1.
The first thing we want to do is - surface the most likely ICP Archetype
def get_top_archetype() =>
We do this by getting the most relevant Archetype Node in the graph.

1.5
We need to make this work for any combo of archetype attributes -
For a given mix of archetype attributes - we should be able to first get the "best match archetypes" from our graph.

2. 
Then for this archetype - we want to get the best-fit ZMOT Events
def get_best_match_zmots(archetype)
We do this by getting the top 5 highest edge-weight zmot events for this archetype

3. 
Then for a given archetype and zmot event we want to get the best match jobs, pains, personas, pain triggers and perceived metrics
def get_zmot_summary(archetype, zmot event) =>
get the highest edge weight job node for this archetype --> zmot event
get all pains of this job
get all pain triggers & metrics for this pain
get all personas of this job
return this along with relevance score for each node


"""

from backend.utils.graph_base.network_graph import get_edge_weight, get_node_by_id, get_node_id, get_nodes_list_ids, get_source_nodes_by_target_and_type, get_target_nodes_by_source_and_type
from backend.utils.graph_base.relevance.cumulative_relevance_manager import get_cumulative_relevance_data


def get_top_archetypes(product_subgraph, threshold = 0.2, top_n = 2):
	"""
	Get top archetypes based on relevance threshold.
	"""
	product_id = get_node_id(product_subgraph, "product", {})
	product_node = get_node_by_id(product_subgraph, product_id)
	if not product_node:
		raise ValueError("Product node not found.")

	archetypes = []
	archetype_ids = get_nodes_list_ids(product_subgraph, "archetype", {})
	for archetype_id in archetype_ids:
		archetype = get_node_by_id(product_subgraph, archetype_id)
		archetype["relevance"] = get_cumulative_relevance_data(product_id, archetype_id)
		archetypes.append(archetype)
	sorted_archetypes = sorted(archetypes, key=lambda x: x["relevance"], reverse=True)
	top_archetypes = [archetype for archetype in sorted_archetypes if archetype["relevance"] >= threshold]
	return top_archetypes[:top_n]

def get_best_match_archetypes(product_subgraph, attribute_combo: dict, threshold=0.2, top_n=5):
    product_id = get_node_id(product_subgraph, "product", {})
    archetype_ids = get_nodes_list_ids(product_subgraph, "archetype", {})
    matches = []

    for archetype_id in archetype_ids:
        archetype = get_node_by_id(product_subgraph, archetype_id)
        match = True
        for key, value in attribute_combo.items():
            if value and key in archetype:
                # Support list of values
                values = value if isinstance(value, list) else [value]
                if not any(v.lower() in str(archetype[key]).lower() for v in values):
                    match = False
                    break
        if match:
            archetype["relevance"] = get_cumulative_relevance_data(product_id, archetype_id)
            matches.append(archetype)

    sorted_matches = sorted(matches, key=lambda x: x["relevance"], reverse=True)
    return [a for a in sorted_matches if a["relevance"] >= threshold][:top_n]

def get_best_match_archetypes(product_subgraph, attribute_combo: dict, threshold=0.2, top_n=5):
    
    product_id = get_node_id(product_subgraph,"product", {})
    archetype_ids = []
    naive_archetype_ids = get_nodes_list_ids(product_subgraph, "archetype", {})
    for naive_archetype_id in naive_archetype_ids:
         related_zmot_ids = get_target_nodes_by_source_and_type(product_subgraph, naive_archetype_id, "responds_to")
         if not related_zmot_ids or len(related_zmot_ids) == 0:
             continue # We want to only consider archetypes that have a zmot downstream
         archetype_ids.append(naive_archetype_id)
         
    matches = []
    
    # Only keep keys that are in priorities
    relevant_keys = {"industry", "revenue_range", "employee_range", "funding_stage", "geography"}
    attribute_combo = {k: v for k, v in attribute_combo.items() if k in relevant_keys}
    # Helper: Related industries (expand as needed)
    related_industries = {
        "automotive": ["transportation", "mobility", "manufacturing"],
        # Add more mappings as needed
    }

    # Helper: Bracket ordering (expand as needed)
    revenue_brackets = ["0-1m", "1-10m", "10-100m", "100-500m", "500m-1b", "1b+"]
    employee_brackets = ["1-50", "51-200", "201-500", "501-1000", "1000-5000", "5000-10000", "10000+"]
    funding_stages = ["seed", "series a", "series b", "series c+", "public"]

    def get_next_bracket(brackets, value, direction="up"):
        try:
            idx = brackets.index(value.lower())
            if direction == "up" and idx < len(brackets) - 1:
                return brackets[idx + 1]
            elif direction == "down" and idx > 0:
                return brackets[idx - 1]
        except ValueError:
            return None
        return None

    # 1. Try exact match
    for archetype_id in archetype_ids:
        archetype = get_node_by_id(product_subgraph, archetype_id)
        match_score = 0
        # Priority order
        priorities = [
            ("industry", 5),
            ("revenue_range", 4),
            ("geography", 3),
            ("employee_range", 2),
            ("funding_stage", 1)
        ]
        exact = True
        for key, weight in priorities:
            val = attribute_combo.get(key)
            arch_val = str(archetype.get(key, "")).lower()
            if val:
                if arch_val == str(val).lower():
                    match_score += weight
                else:
                    exact = False
        if exact:
            archetype["relevance"] = get_cumulative_relevance_data(product_id, archetype_id)
            archetype["match_score"] = match_score
            matches.append(archetype)

    # 2. If no exact match, try softer matching
    if not matches:
        print("No exact matches found, trying softer matching...")
        for archetype_id in archetype_ids:
            archetype = get_node_by_id(product_subgraph, archetype_id)
            match_score = 0
            for key, weight in priorities:
                val = attribute_combo.get(key)
                arch_val = str(archetype.get(key, "")).lower()
                print(f"Matching {key}: input={val}, node={arch_val}")
                if val:
                    # Industry: try related
                    if key == "industry":
                        if arch_val == str(val).lower():
                            match_score += weight
                        elif val.lower() in related_industries and arch_val in related_industries[val.lower()]:
                            match_score += weight - 1
                    # Revenue/Employee/Funding: try next higher, then lower
                    elif key in ["revenue_range", "employee_range", "funding_stage"]:
                        brackets = revenue_brackets if key == "revenue_range" else employee_brackets if key == "employee_range" else funding_stages
                        if arch_val == str(val).lower():
                            match_score += weight
                        else:
                            up = get_next_bracket(brackets, str(val).lower(), "up")
                            down = get_next_bracket(brackets, str(val).lower(), "down")
                            if arch_val == up:
                                match_score += weight - 1
                            elif arch_val == down:
                                match_score += weight - 2
                    # Geography: exact, then skip
                    elif key == "geography":
                        if arch_val == str(val).lower():
                            match_score += weight
            if match_score > 0:
                archetype["relevance"] = get_cumulative_relevance_data(product_id, archetype_id)
                archetype["match_score"] = match_score
                matches.append(archetype)

    # Sort by match_score (priority), then relevance
    sorted_matches = sorted(matches, key=lambda x: (x["match_score"], x["relevance"]), reverse=True)
    return [a for a in sorted_matches if a["relevance"] >= threshold][:top_n]

def get_best_match_zmots_for_archetype(product_subgraph, archetype_id, threshold=0.2, top_n=5):
	product_id = get_node_id(product_subgraph, "product", {})
	related_zmot_ids = get_target_nodes_by_source_and_type(product_subgraph, archetype_id, "responds_to")
	matches = []

	for zmot_id in related_zmot_ids:
		zmot_node = get_node_by_id(product_subgraph, zmot_id)
		zmot_relevance = get_cumulative_relevance_data(product_id, zmot_id)
		if zmot_relevance < threshold:
			continue
		zmot_node["relevance"] = zmot_relevance
		zmot_node["org_relevance"] = get_edge_weight(product_subgraph, archetype_id, zmot_id)
		matches.append(zmot_node)

	sorted_matches = sorted(matches, key=lambda x: x["relevance"], reverse=True)
	return sorted_matches[:top_n]

def build_zmot_summary(product_subgraph, zmot_event_id, archetype_id):
    """
    Final output is a paragraph summary of the format:
    1. Archetype experienced ZMOT Event
    2. This was observed from Observable Moments using Trigger Keywords
    3. The ZMOT Event forced Persona to perform Job
    4. This Job pushed a bunch of downstream Pains
    5. Each Pain was getting worse due to Pain Triggers, as experienced by Perceived Metrics
    """
    product_id = get_node_id(product_subgraph, "product", {})
    
    archetype_node = get_node_by_id(product_subgraph, archetype_id)
    print("RCS for Archetype: \n")
    print(archetype_node)
    print("--------------------------------")
    zmot_node = get_node_by_id(product_subgraph, zmot_event_id)
    if not zmot_node:
        return None
    trigger_event = zmot_node.get("trigger_event", "").strip().lower()
    print("trigger event: ", trigger_event)
    
    observable_moments = []
    trigger_keywords = []
    print("Observable Moments and Trigger Keywords for ZMOT Event:")
    observable_moment_ids = get_target_nodes_by_source_and_type(product_subgraph, zmot_event_id, "observed_in")
    for observable_moment_id in observable_moment_ids:
        observable_moment_node = get_node_by_id(product_subgraph, observable_moment_id)
        observable_moment = observable_moment_node.get("observable_moment", "").strip().lower()
        print(observable_moment)
    keywords = get_target_nodes_by_source_and_type(product_subgraph, zmot_event_id, "associated_with")
    print("\n Where we tracked phrases like:")
    for keyword in keywords:
        keyword_node = get_node_by_id(product_subgraph, keyword)
        keyword = keyword_node.get("keyword", "").strip().lower()
        print(keyword)
	
    shortest_path = nx.shortest_path(product_subgraph, source=product_id, target=zmot_event_id, weight='weight')
    print("Shortest Path from Product to ZMOT Event:", shortest_path)

    for node_id in shortest_path:
        node = get_node_by_id(product_subgraph, node_id)
        node_type = node.get("node_type", "").strip().lower()
        print("Node in shortest path:", node_id, "Type:", node_type)

    impacted_job_ids = get_source_nodes_by_target_and_type(product_subgraph, zmot_event_id, "triggered_by")
    for impacted_job_id in impacted_job_ids:
        responsible_persona_ids = get_target_nodes_by_source_and_type(product_subgraph, impacted_job_id, "performed_by")
        persona_node = get_node_by_id(product_subgraph, responsible_persona_ids[0]) if responsible_persona_ids else None
        persona_title = persona_node.get("title", "").strip().lower() if persona_node else ""
        job_description = get_node_by_id(product_subgraph, impacted_job_id).get("description", "").strip().lower()
        print("Primary Job Owner?")
        print(f"{persona_title} with job: {job_description}")
        print("\nImmediate Domino Effect?")
        downstream_pain_ids = get_source_nodes_by_target_and_type(product_subgraph, impacted_job_id, "solves")
        for downstream_pain_id in downstream_pain_ids:
            pain_node = get_node_by_id(product_subgraph, downstream_pain_id)
            pain_text = pain_node.get("text", "").strip().lower()
            pain_trigger_ids = get_target_nodes_by_source_and_type(product_subgraph, downstream_pain_id, "scales_with")
            pain_triggers = [get_node_by_id(product_subgraph, pt).get("attribute", "").strip().lower() for pt in pain_trigger_ids]
            perceived_metric_ids = get_target_nodes_by_source_and_type(product_subgraph, downstream_pain_id, "expressed_as")
            perceived_metrics = [get_node_by_id(product_subgraph, pm).get("metric", "").strip().lower() for pm in perceived_metric_ids]
            print(f"{perceived_metrics} started looking bad")
            print("\nAnd got worse due to:")
            print(perceived_metrics)

        

    # Get the relevant data
    zmot_summary = {
        "trigger_event": zmot_node.get("trigger_event", ""),
        "relevance": zmot_node.get("relevance", 0),
        "org_relevance": zmot_node.get("org_relevance", 0)
    }

    return zmot_summary