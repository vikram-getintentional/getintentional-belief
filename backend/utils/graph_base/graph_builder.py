from backend.utils.graph_base.nodes.company_nodes import get_or_create_employees_node, get_or_create_funding_stage_node, get_or_create_geographies_node, get_or_create_industry_node, get_or_create_revenue_node
from backend.utils.graph_base.nodes.persona_nodes import get_or_create_persona_node
from backend.utils.graph_base.nodes.job_nodes import get_or_create_job_node
from backend.utils.graph_base.nodes.pain_nodes import get_or_create_pain_node
from backend.utils.graph_base.nodes.pain_trigger_nodes import get_or_create_pain_trigger_node
from backend.utils.graph_base.edges.edge_manager import add_edge
from collections import defaultdict
from datetime import datetime
from backend.utils.graph_base.nodes.capability_nodes import get_or_create_capability_node
from backend.utils.graph_base.nodes.zmot_nodes import get_or_create_trigger_event_node, get_or_create_keyword_node, get_or_create_observable_moment_node
from backend.utils.knowledge_base.canonicalizer import canonicalize_observable_moments, canonicalize_pain_trigger, canonicalize_trigger_events

# This function creates edges for the node IDs in the flattened list
def process_capability_map_to_graph(capabilities_list, capability_map: dict):
    flattened_results = []
    print("Starting process map")
    for entry in capability_map:
            print("Processing entry:", entry)
            capability_id = entry.get("capability_id")
            pain_id = entry.get("pain_node_id")
            relevance_array = entry.get("relevance", [])
            job_id = entry.get("job_node_id")
            persona_node_id = entry.get("persona_node_id")
            job_impact = entry.get("job_impact", 0.5)
            persona_job_importance = entry.get("persona_job_importance", 0.5)
            source = entry.get("source", "unknown")
            now = datetime.utcnow().isoformat()
            

            
            # Looks through cap_list array and sets an edge for each cap to this pain combo with weight = relevance_array[i]
            for i, cap in enumerate(capabilities_list):
                cap_id = cap["id"]
                relevance = relevance_array[i] if i < len(relevance_array) else 0
                # Add edge: Capability → Pain
                add_edge(
                    source_id=cap_id,
                    target_id=pain_id,
                    edge_type="solves",
                    weight=relevance,
                    last_updated=now,
                    source=source
                )
                print(f"Added edge from Capability {cap_id} to Pain {pain_id} with weight {relevance}")
            # Add edge: Pain → Job
            add_edge(
                source_id=pain_id,
                target_id=job_id,
                edge_type="addresses",
                weight=job_impact,
                last_updated=now,
                source=source
            )

            # Add edge: Job → Persona
            add_edge(
                source_id=job_id,
                target_id=persona_node_id,
                edge_type="performed_by",
                weight=persona_job_importance,
                last_updated=now,
                source=source
            )
            # (Optional) Add to flattened_results for downstream use
            flattened_results.append(entry)
    print("Finished final Hop0 flattened edges:", flattened_results)
    return flattened_results


# This function gets flattened gpt results and creates canonicalized nodes. Return is the flattened map with node IDs.
# Updated logic Hop0 (and all subsequent hops) only return pain-job-persona triplets.
def convert_rule_matches_to_capability_map(results):
    from backend.utils.knowledge_base.canonicalizer import (
        canonicalize_pain,
        canonicalize_job,
        canonicalize_persona,
        canonicalize_pain_trigger
    )
    print("Converting rule matches to capability map...")
    pain_cache = set()
    job_cache = set()
    persona_cache = set()
    
    capability_map = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    # Gather unique raw values for canonicalization
    for entry in results:    
        raw_pain = entry.get("pain", "").strip()
        raw_job = entry.get("job", "").strip()
        raw_persona = entry.get("persona", {})
        pain_cache.add(raw_pain)
        job_cache.add(raw_job)
        if isinstance(raw_persona, dict):
            persona_cache.add((
                raw_persona.get("title", "").strip(),
                raw_persona.get("department", "").strip(),
                raw_persona.get("seniority", "").strip()
            ))
        
    # Canonicalize
    canonical_personas = canonicalize_persona([
        {"title": t, "department": d, "seniority": s}
        for (t, d, s) in persona_cache
    ])
    canonical_pains = canonicalize_pain(list(pain_cache))
    canonical_jobs = canonicalize_job(list(job_cache))
    
    # Lookup logic for setting right node IDs to entry list
    persona_lookup = {
        (p["title"].strip().lower(), p["department"].strip().lower(), p["seniority"].strip().lower()):
            get_or_create_persona_node(p["title"], p["department"], p["seniority"])["id"]
        for p in canonical_personas.values()
    }
    pain_lookup = {
        p.strip().lower(): get_or_create_pain_node(p)["id"]
        for p in canonical_pains.values()
    }
    job_lookup = {
        j.strip().lower(): get_or_create_job_node(j)["id"]
        for j in canonical_jobs.values()
    }
        

    # Assign canonical values and node IDs to each entry
    for entry in results:
        # Canonicalize pain
        raw_pain = entry.get("pain", "").strip()
        pain_canonical = canonical_pains.get(raw_pain, raw_pain)
        entry["pain"] = pain_canonical

        # Canonicalize job
        raw_job = entry.get("job", "").strip()
        job_canonical = canonical_jobs.get(raw_job, raw_job)
        entry["job"] = job_canonical

        # Canonicalize persona
        raw_persona = entry.get("persona", {})
        persona_key = str({
            "title": raw_persona.get("title", "").strip(),
            "department": raw_persona.get("department", "").strip(),
            "seniority": raw_persona.get("seniority", "").strip()
        })
        persona_canonical = canonical_personas.get(persona_key, raw_persona)
        entry["persona"] = persona_canonical

    
        # Lookup node IDs using canonicalized values
        persona_lookup_key = (
            entry["persona"]["title"].strip().lower(),
            entry["persona"]["department"].strip().lower(),
            entry["persona"]["seniority"].strip().lower()
        )
        if persona_lookup_key not in persona_lookup:
            print("❌ Persona lookup failed for key:", persona_lookup_key)
            print("Available persona keys:", list(persona_lookup.keys()))

        pain_lookup_key = entry["pain"].strip().lower()
        if pain_lookup_key not in pain_lookup:
            print("❌ Pain lookup failed for key:", pain_lookup_key)
            print("Available pain keys:", list(pain_lookup.keys()))

        job_lookup_key = entry["job"].strip().lower()
        if job_lookup_key not in job_lookup:
            print("❌ Job lookup failed for key:", job_lookup_key)
            print("Available job keys:", list(job_lookup.keys()))
    
        entry["pain_node_id"] = pain_lookup.get(pain_lookup_key)
        entry["job_node_id"] = job_lookup.get(job_lookup_key)
        entry["persona_node_id"] = persona_lookup.get(persona_lookup_key)
        

    print("Updated results blob:", results)
    return results


def canonicalize_and_create_zmot_icp_nodes(results):
    print("Starting node creation & processing for zmot & ICP")
    trigger_cache = set()
    trigger_event_cache = set()
    observable_moment_cache = set()
    keywords_cache = set()

    # Gather unique raw values for canonicalization
    for entry in results:
        raw_pain_trigger = entry.get("pain_trigger", {})
        if isinstance(raw_pain_trigger, dict):
            trigger_cache.add((
                raw_pain_trigger.get("attribute", "").strip(),
                raw_pain_trigger.get("dimension", "").strip(),
                raw_pain_trigger.get("direction", "").strip()
            ))
        zmot_event = entry.get("zmot_event", {})
        trigger_event = zmot_event.get("trigger_event", "").strip()
        observable_moments = zmot_event.get("observable_moments", [])
        keywords = zmot_event.get("trigger_keywords", [])
        if trigger_event:
            trigger_event_cache.add(trigger_event)
        for observable_moment in observable_moments:
            observable_moment_cache.add(observable_moment)
        for kw in keywords:
            kw_stripped = kw.strip()
            if kw_stripped:
                keywords_cache.add(kw_stripped)

    # Canonicalize
    canonical_pain_triggers = canonicalize_pain_trigger([
        {"attribute": a, "dimension": d, "direction": s}
        for (a, d, s) in trigger_cache
    ])
    canonical_zmot_trigger_events = canonicalize_trigger_events(list(trigger_event_cache))
    canonical_zmot_observable_moments = canonicalize_observable_moments(list(observable_moment_cache))
    canonical_zmot_keywords = list(keywords_cache)

    # Lookup logic for setting right node IDs to entry list
    pain_trigger_lookup = {
        f"{pt['attribute'].strip().lower()}|{pt['dimension'].strip().lower()}|{pt['direction'].strip().lower()}":
            get_or_create_pain_trigger_node(pt["attribute"], pt["dimension"], pt["direction"])["id"]
        for pt in canonical_pain_triggers.values()
    }
    zmot_trigger_event_lookup = {
        event.strip().lower(): get_or_create_trigger_event_node(event)["id"]
        for event in canonical_zmot_trigger_events.values()
    }
    zmot_observable_moment_lookup = {
        moment.strip().lower(): get_or_create_observable_moment_node(moment)["id"]
        for moment in canonical_zmot_observable_moments.values()
    }
    zmot_keyword_lookup = {
        kw.strip().lower(): get_or_create_keyword_node(kw)["id"]
        for kw in canonical_zmot_keywords
    }

    # Assign canonical values and node IDs to each entry
    for entry in results:
        # Canonicalize pain_trigger
        raw_pain_trigger = entry.get("pain_trigger", {})
        pt_key = f"{raw_pain_trigger.get('attribute', '').strip().lower()}|{raw_pain_trigger.get('dimension', '').strip().lower()}|{raw_pain_trigger.get('direction', '').strip().lower()}"
        pt_canonical = canonical_pain_triggers.get(pt_key, raw_pain_trigger)
        entry["pain_trigger"] = pt_canonical

        # Lookup node ID for pain_trigger
        pain_trigger_lookup_key = f"{pt_canonical['attribute'].strip().lower()}|{pt_canonical['dimension'].strip().lower()}|{pt_canonical['direction'].strip().lower()}"
        if pain_trigger_lookup_key not in pain_trigger_lookup:
            print("❌ Pain trigger lookup failed for key:", pain_trigger_lookup_key)
            print("Available pain trigger keys:", list(pain_trigger_lookup.keys()))
        
        # Canonicalize and assign node IDs for zmot_event
        zmot_event = entry.get("zmot_event", {})
        zmot_trigger_event_node_ids = []
        zmot_observable_moment_node_ids = []
        zmot_keyword_node_ids = []

        trigger_event_raw = zmot_event.get("trigger_event", "").strip()
        trigger_event_canonical = canonical_zmot_trigger_events.get(trigger_event_raw, trigger_event_raw)
        trigger_event_key = trigger_event_canonical.strip().lower()
        if trigger_event_key in zmot_trigger_event_lookup:
            zmot_trigger_event_node_ids.append(zmot_trigger_event_lookup[trigger_event_key])
        else:
            print("❌ ZMOT trigger event lookup failed for key:", trigger_event_key)
            print("Available trigger event keys:", list(zmot_trigger_event_lookup.keys()))

        observable_moments = zmot_event.get("observable_moments", [])
        if isinstance(observable_moments, str):
            observable_moments = [observable_moments]
        for observable_moment in observable_moments:
            observable_moment = observable_moment.strip().lower()
            if observable_moment in zmot_observable_moment_lookup:
                zmot_observable_moment_node_ids.append(zmot_observable_moment_lookup[observable_moment])

        keywords = zmot_event.get("trigger_keywords", [])
        if isinstance(keywords, str):
            keywords = [keywords]
        for keyword in keywords:
            keyword = keyword.strip().lower()
            if keyword in zmot_keyword_lookup:
                zmot_keyword_node_ids.append(zmot_keyword_lookup[keyword])

        entry["pain_trigger_node_id"] = pain_trigger_lookup.get(pain_trigger_lookup_key)
        entry["zmot_trigger_event_node_ids"] = zmot_trigger_event_node_ids
        entry["zmot_observable_moment_node_ids"] = zmot_observable_moment_node_ids
        entry["zmot_keyword_node_ids"] = zmot_keyword_node_ids

        # Canonicalize and assign node IDs for icp_archetype
        icp = entry.get("icp_archetype", {})
        icp_entry = {
            "industry_nodes": [],
            "revenue_nodes": [],
            "employees_nodes": [],
            "funding_stage_nodes": [],
            "geographies_nodes": []
        }
        for industry in icp.get("industries", []):
            industry_node = get_or_create_industry_node(industry)
            icp_entry["industry_nodes"].append(industry_node["id"])
        for revenue in icp.get("Revenues", []):
            revenue_node = get_or_create_revenue_node(revenue)
            icp_entry["revenue_nodes"].append(revenue_node["id"])
        for employees in icp.get("Employees", []):
            employees_node = get_or_create_employees_node(employees)
            icp_entry["employees_nodes"].append(employees_node["id"])
        for funding_stage in icp.get("Funding Stages", []):
            funding_stage_node = get_or_create_funding_stage_node(funding_stage)
            icp_entry["funding_stage_nodes"].append(funding_stage_node["id"])
        for geographies in icp.get("Geographies", []):
            geographies_node = get_or_create_geographies_node(geographies)
            icp_entry["geographies_nodes"].append(geographies_node["id"])
        entry["icp_nodes"] = [icp_entry] if any(icp_entry.values()) else []

    print("Updated zmot results blob:", results)
    return results


# This function creates edges for the node IDs in the flattened list
def process_pain_triggers_zmot_icp_map_to_graph(pains_map: dict):
    flattened_results = []
    print("Starting process map")
    for entry in pains_map:
            print("Processing zmot entry:", entry)
            pain_id = entry.get("pain_id")
            pain_trigger_node_id = entry.get("pain_trigger_node_id", None)
            icp_archetype = entry.get("icp_archetype", {})
            icp_match_score = icp_archetype.get("icp_match_score", 0.0)

            zmot_event = entry.get("zmot_event", {})
            zmot_match_score = zmot_event.get("zmot_match_score", 0.0)
            
            source = entry.get("source", "unknown")
            now = datetime.utcnow().isoformat()
            
            # Add edge: Pain → Pain Trigger
            if pain_id and pain_trigger_node_id:
                add_edge(
                    source_id=pain_id,
                    target_id=pain_trigger_node_id,
                    edge_type="triggered_by",
                    weight=1.0,
                    last_updated=now,
                    source=source
                )

            # Add edges: Pain Trigger -> Company Data
            # Already created icp_nodes list for each entry
            for icp_entry in entry.get("icp_nodes", []):
                for industry_id in icp_entry.get("industry_nodes", []):
                    # Add edge: Pain Trigger → Industry
                    if pain_trigger_node_id and industry_id:
                        add_edge(
                            source_id=pain_trigger_node_id,
                            target_id=industry_id,
                            edge_type="experienced_in",
                            weight=icp_match_score,
                            last_updated=now,
                            source=source
                        )
                for revenue_id in icp_entry.get("revenue_nodes", []):
                    # Add edge: Pain Trigger → Revenue
                    if pain_trigger_node_id and revenue_id:
                        add_edge(
                            source_id=pain_trigger_node_id,
                            target_id=revenue_id,
                            edge_type="experienced_in",
                            weight=icp_match_score,
                            last_updated=now,
                            source=source
                        )
                for employees_id in icp_entry.get("employees_nodes", []):
                    # Add edge: Pain Trigger → Employees
                    if pain_trigger_node_id and employees_id:
                        add_edge(
                            source_id=pain_trigger_node_id,
                            target_id=employees_id,
                            edge_type="experienced_in",
                            weight=icp_match_score,
                            last_updated=now,
                            source=source
                        )
                for funding_stage_id in icp_entry.get("funding_stage_nodes", []):
                    # Add edge: Pain Trigger → Funding Stage
                    if pain_trigger_node_id and funding_stage_id:
                        add_edge(
                            source_id=pain_trigger_node_id,
                            target_id=funding_stage_id,
                            edge_type="experienced_in",
                            weight=icp_match_score,
                            last_updated=now,
                            source=source
                        )
                for geographies_id in icp_entry.get("geographies_nodes", []):
                    # Add edge: Pain Trigger → Geographies
                    if pain_trigger_node_id and geographies_id:
                        add_edge(
                            source_id=pain_trigger_node_id,
                            target_id=geographies_id,
                            edge_type="experienced_in",
                            weight=icp_match_score,
                            last_updated=now,
                            source=source
                        )
            # Add edges: Pain Trigger → ZMOT Trigger Events
            for zmot_trigger_event_id in entry.get("zmot_trigger_event_node_ids", []):
                if pain_trigger_node_id and zmot_trigger_event_id:
                    add_edge(
                        source_id=pain_trigger_node_id,
                        target_id=zmot_trigger_event_id,
                        edge_type="zmot_trigger_event",
                        weight=zmot_match_score,
                        last_updated=now,
                        source=source
                    )

                # Add edges: Trigger Event → ZMOT Observable Moments
                for zmot_observable_moment_id in entry.get("zmot_observable_moment_node_ids", []):
                    if zmot_trigger_event_id and zmot_observable_moment_id:
                        add_edge(
                            source_id=zmot_trigger_event_id,
                            target_id=zmot_observable_moment_id,
                            edge_type="zmot_observable_moment",
                            weight=1.0,
                            last_updated=now,
                            source=source
                        )

                # Add edges: Pain Trigger → ZMOT Keywords
                for zmot_keyword_id in entry.get("zmot_keyword_node_ids", []):
                    if zmot_trigger_event_id and zmot_keyword_id:
                        add_edge(
                            source_id=zmot_trigger_event_id,
                            target_id=zmot_keyword_id,
                            edge_type="zmot_keyword",
                            weight=1.0,
                            last_updated=now,
                            source=source
                        )
            # (Optional) Add to flattened_results for downstream use
            flattened_results.append(entry)
    print("Finished edge math for zmot & ICP:", flattened_results)
    return flattened_results