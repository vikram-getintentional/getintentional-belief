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
from backend.utils.knowledge_base.canonicalizer import (
        canonicalize_pain,
        canonicalize_job,
        canonicalize_persona,
        canonicalize_pain_trigger
    )

# This function gets flattened gpt results and creates canonicalized nodes. Return is the flattened map with node IDs.
def convert_rule_matches_to_capability_map(results, capability_ids_list, product_id):
    print("Converting rule matches to capability map...")
    pain_cache = set()
    job_cache = set()
    persona_cache = set()
    capability_map = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    """
    results is like:
    - cap_id:
    - capability:
    - "pains": [
        pain: text
        relevance: array
        jobs:[
            description
            impact
            personas:[
                job_importance
                title
                department
                seniority
            ]
        ]]
    """
    # Gather unique raw values for canonicalization
    for entry in results:
        pains = entry.get("pains", [])
        if not isinstance(pains, list):
            print("❌ Invalid pains format in entry:", entry)
            continue
        for pain in pains:
            raw_pain = pain.get("pain", "").strip()
            if raw_pain:
                pain_cache.add(raw_pain)
            else:
                print("❌ Pain text is empty in entry:", entry)
            jobs = pain.get("jobs", [])
            if not isinstance(jobs, list):
                print("❌ Invalid jobs format in entry:", entry)
                continue
            for job in jobs:
                raw_job = job.get("description", "").strip()
                if raw_job:
                    job_cache.add(raw_job)
                else:
                    print("❌ Job description is empty in entry:", entry)
                personas = job.get("personas", [])
                if not isinstance(personas, list):
                    print("❌ Invalid personas format in entry:", entry)
                    continue
                for persona in personas:
                    raw_persona = {
                        "title": persona.get("title", "").strip(),
                        "department": persona.get("department", "").strip(),
                        "seniority": persona.get("seniority", "").strip()
                    }
                    if raw_persona.get("title") or raw_persona.get("department") or raw_persona.get("seniority"):
                        persona_cache.add((
                            raw_persona.get("title"),
                            raw_persona.get("department"),
                            raw_persona.get("seniority")
                        ))
                    else:
                        print("❌ Persona fields are empty in entry:", entry)

    print("Raw caches collected. Now canonicalizing...")
    print("Processed Gpt Results:")
    for entry in results:
        print(entry)
        print("--------------------------------------------------")
    # At this point we have: (1) The exact structure of gpt output, (2) Raw values added to each cache
    # Canonicalize
    canonical_personas = canonicalize_persona([
        {"title": t, "department": d, "seniority": s}
        for (t, d, s) in persona_cache
    ])
    canonical_pains = canonicalize_pain(list(pain_cache))
    canonical_jobs = canonicalize_job(list(job_cache))

    # Lookup logic for setting right node IDs to entry list
    persona_lookup = {
        str({
            "title": p["title"].strip(),
            "department": p["department"].strip(),
            "seniority": p["seniority"].strip()
        }): get_or_create_persona_node(p["title"], p["department"], p["seniority"])["id"]
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
    print("Final Lookup dictionaries created with sizes. Starting Hop0 node and edge processing")
    for entry in results:
        source = entry.get("source", "openai")
        now = datetime.utcnow().isoformat()
        capability_id = entry.get("capability_id", "").strip()
        pains = entry.get("pains", [])
        for pain in pains:
            # Canonicalize pain
            raw_pain = pain.get("pain", "").strip()
            pain_canonical = canonical_pains.get(raw_pain, raw_pain)
            pain["pain"] = pain_canonical
            pain_lookup_key = pain["pain"].strip().lower()
            if pain_lookup_key not in pain_lookup:
                print("❌ Pain lookup failed for key:", pain_lookup_key)
                print("Available pain keys:", list(pain_lookup.keys()))
            pain_node_id = pain_lookup.get(pain_lookup_key)
            relevance_array = pain.get("relevance", [])
            for i, cap_id in enumerate(capability_ids_list):
                relevance = relevance_array[i] if i < len(relevance_array) else 0
                # Add edge: Capability → Pain
                add_edge(
                    product_id=product_id,
                    source_id=cap_id,
                    target_id=pain_node_id,
                    edge_type="solves",
                    weight=relevance,
                    last_updated=now,
                    source=source
                )
                print(f"Added edge from Capability {cap_id} to Pain {pain_node_id} with weight {relevance}")
            print("All pains processed - moving to jobs")
            for job in pain.get("jobs", []):
                # Canonicalize job
                raw_job = job.get("description", "").strip()
                job_canonical = canonical_jobs.get(raw_job, raw_job)
                job["description"] = job_canonical

                job_lookup_key = job_canonical.strip().lower()
                if job_lookup_key not in job_lookup:
                    print("❌ Job lookup failed for key:", job_lookup_key)
                    print("Available job keys:", list(job_lookup.keys()))
                job_node_id = job_lookup.get(job_lookup_key)
                job_impact = job.get("impact", 0.5)  # Default to 0.5 if not specified

                # Add edge: Pain → Job
                add_edge(
                    product_id=product_id,
                    source_id=pain_node_id,
                    target_id=job_node_id,
                    edge_type="addresses",
                    weight=job_impact,
                    last_updated=now,
                    source=source
                )
                print(f"Added edge from Pain {pain_node_id} to Job {job_node_id} with weight {job_impact}")

                for raw_persona in job.get("personas", []):
                    # Canonicalize persona
                    print("Raw persona in lookup:", raw_persona)

                    persona_key = str({
                        "title": raw_persona.get("title", "").strip(),
                        "department": raw_persona.get("department", "").strip(),
                        "seniority": raw_persona.get("seniority", "").strip()
                    })
                    persona_canonical = canonical_personas.get(persona_key, raw_persona)
                    persona = persona_canonical
                    persona_title = persona.get("title", "").strip()
                    persona_department = persona.get("department", "").strip()
                    persona_seniority = persona.get("seniority", "").strip()
                    
                    if persona_title == "" or persona_department == "" or persona_seniority == "":
                        print("❌ Persona fields are empty in lookup:", entry)
                        continue
                    print("Persona canonicalization complete - moving to lookup")
                    # Lookup node IDs using canonicalized values
                    persona_lookup_key = str({
                        "title": persona_title.strip(),
                        "department": persona_department.strip(),
                        "seniority": persona_seniority.strip()
                    })
                    print("Persona lookup key:", persona_lookup_key)
                    if persona_lookup_key not in persona_lookup:
                        print("❌ Persona lookup failed for key:", persona_lookup_key)
                        print("Available persona keys:", list(persona_lookup.keys()))
                    
                    persona_node_id = persona_lookup.get(persona_lookup_key)
                    persona_job_importance = persona.get("job_importance", 0.5)  # Default to 0.5 if not specified

                    # Add edge: Job → Persona
                    add_edge(
                        product_id=product_id,
                        source_id=job_node_id,
                        target_id=persona_node_id,
                        edge_type="performed_by",
                        weight=persona_job_importance,
                        last_updated=now,
                        source=source
                    )
                    print("Added edge from Job", job_node_id, "to Persona", persona_node_id, "with weight", persona_job_importance)

        
        
    print("Updated results blob:", results)
    return results


def canonicalize_and_create_zmot_icp_nodes(gpt_results, product_id):
    print("Starting node creation & processing for zmot & ICP")
    trigger_cache = set()
    trigger_event_cache = set()
    observable_moment_cache = set()
    keywords_cache = set()

    # Gather unique raw values for canonicalization
    for entry in gpt_results:
        entry["source"] = "openai"
        pain_id = entry.get("pain_id").strip()
        if not pain_id:
            print("❌ Pain ID is missing in entry:", entry)
            continue

        raw_pain_triggers = entry.get("pain_triggers", [])
        if isinstance(raw_pain_triggers, list):
            for raw_pain_trigger in raw_pain_triggers:
                trigger_cache.add((
                    raw_pain_trigger.get("attribute", "").strip(),
                    raw_pain_trigger.get("dimension", "").strip(),
                    raw_pain_trigger.get("direction", "").strip()
                ))
                icp_archetypes = raw_pain_trigger.get("icp_archetypes", [])
                if isinstance(icp_archetypes, dict):
                    icp_archetypes = [icp_archetypes]
                
                
                zmot_events = raw_pain_trigger.get("zmot_events", {})
                if isinstance(zmot_events, dict):
                    zmot_events = [zmot_events]
                for zmot_event in zmot_events:
                    raw_trigger_event = zmot_event.get("trigger_event", "").strip()
                    zmot_event["match_score"] = zmot_event.get("trigger_event_match_score", 0.0)
                    if raw_trigger_event:
                        trigger_event_cache.add(raw_trigger_event)
                    observable_moments = zmot_event.get("observable_moments", [])
                    if isinstance(observable_moments, dict):
                        observable_moments = [observable_moments]
                    for observable_moment in observable_moments:
                        observable_moment_text = observable_moment.get("observable_moment", "").strip()
                        observable_moment["match_score"] = observable_moment.get("match_score", 0.0)
                        if observable_moment_text:
                            observable_moment_cache.add(observable_moment_text)
                    keywords = zmot_event.get("trigger_keywords", [])
                    if isinstance(keywords, dict):
                        keywords = [keywords]
                    for kw in keywords:
                        kw_stripped = kw.get("trigger_keyword", "").strip()
                        kw["match_score"] = kw.get("match_score", 0.0)
                        if kw_stripped:
                            keywords_cache.add(kw_stripped)

    # At this point we have: (1) The exact structure of gpt output, (2) Raw values added to each cache
    # Canonicalize raw caches
    canonical_pain_triggers = canonicalize_pain_trigger([
        {"attribute": a, "dimension": d, "direction": s}
        for (a, d, s) in trigger_cache
    ])
    canonical_zmot_trigger_events = canonicalize_trigger_events(list(trigger_event_cache))
    canonical_zmot_observable_moments = canonicalize_observable_moments(list(observable_moment_cache))
    canonical_zmot_keywords = list(keywords_cache)

    # Create Nodes and do Lookup logic for setting right node IDs to each entry
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

    # At this point we still have the raw gpt_entry, and a lookup dict that has canon values & node IDs
    # Assign canonical values and node IDs to each entry, and create edges as we go
    for entry in gpt_results:
        now = datetime.utcnow().isoformat()
        source = entry.get("source", "openai")
        pain_id = entry.get("pain_id", "").strip()
        raw_pain_triggers = entry.get("pain_triggers", [])
        if isinstance(raw_pain_triggers, list):
            for raw_pain_trigger in raw_pain_triggers:
                # For each raw pain trigger, find the canonicalized version and get the node ID
                raw_pain_trigger_key = f"{raw_pain_trigger.get('attribute', '').strip().lower()}|{raw_pain_trigger.get('dimension', '').strip().lower()}|{raw_pain_trigger.get('direction', '').strip().lower()}"
                canonical_trigger = canonical_pain_triggers.get(raw_pain_trigger_key)
                if canonical_trigger:
                    canonical_key = f"{canonical_trigger['attribute'].strip().lower()}|{canonical_trigger['dimension'].strip().lower()}|{canonical_trigger['direction'].strip().lower()}"
                else:
                    canonical_key = raw_pain_trigger_key
                
                if canonical_key not in pain_trigger_lookup:
                    print("❌ Pain trigger lookup failed for key:", canonical_key)
                    print("Available pain trigger keys:", list(pain_trigger_lookup.keys()))
                pain_trigger_node_id = pain_trigger_lookup.get(canonical_key)
                # Add edge: Pain → Pain Trigger
                if pain_id and pain_trigger_node_id:
                    add_edge(
                        product_id=product_id,
                        source_id=pain_id,
                        target_id=pain_trigger_node_id,
                        edge_type="scales_with",
                        weight=1.0,
                        last_updated=now,
                        source=source
                    )
                
                
                for icp in icp_archetypes:
                    for industry in icp.get("industries", []):
                        industry_name = industry.get("industry", "").strip()
                        match_score = industry.get("match_score", 0.0)
                        if industry_name:
                            industry_node = get_or_create_industry_node(industry_name)
                            industry_id = industry_node["id"]
                            # Add edge: Pain Trigger → Industry
                            if pain_trigger_node_id and industry_id:
                                add_edge(
                                    product_id=product_id,
                                    source_id=pain_trigger_node_id,
                                    target_id=industry_id,
                                    edge_type="experienced_in",
                                    weight=match_score,
                                    last_updated=now,
                                    source=source
                                )
                    for revenue in icp.get("revenues", []):
                        revenue_name = revenue.get("revenue", "").strip()
                        match_score = revenue.get("match_score", 0.0)
                        if revenue_name:
                            revenue_node = get_or_create_revenue_node(revenue_name)
                            revenue_id = revenue_node["id"]
                            # Add edge: Pain Trigger → Revenue
                            if pain_trigger_node_id and revenue_id:
                                add_edge(
                                    product_id=product_id,
                                    source_id=pain_trigger_node_id,
                                    target_id=revenue_id,
                                    edge_type="experienced_in",
                                    weight=match_score,
                                    last_updated=now,
                                    source=source
                                )
                    for employees in icp.get("employees", []):
                        employees_name = employees.get("employees", "").strip()
                        match_score = employees.get("match_score", 0.0)
                        if employees_name:
                            employees_node = get_or_create_employees_node(employees_name)
                            employees_id = employees_node["id"]
                            # Add edge: Pain Trigger → Employees
                            if pain_trigger_node_id and employees_id:
                                add_edge(
                                    product_id=product_id,
                                    source_id=pain_trigger_node_id,
                                    target_id=employees_id,
                                    edge_type="experienced_in",
                                    weight=match_score,
                                    last_updated=now,
                                    source=source
                                )
                    for funding_stage in icp.get("funding_stages", []):
                        funding_stage_name = funding_stage.get("funding_stage", "").strip()
                        match_score = funding_stage.get("match_score", 0.0)
                        if funding_stage_name:
                            funding_stage_node = get_or_create_funding_stage_node(funding_stage_name)
                            funding_stage_id = funding_stage_node["id"]
                            # Add edge: Pain Trigger → Funding Stage
                            if pain_trigger_node_id and funding_stage_id:
                                add_edge(
                                    product_id=product_id,
                                    source_id=pain_trigger_node_id,
                                    target_id=funding_stage_id,
                                    edge_type="experienced_in",
                                    weight=match_score,
                                    last_updated=now,
                                    source=source
                                )
                    for geography in icp.get("geographies", []):
                        geography_name = geography.get("geography", "").strip()
                        match_score = geography.get("match_score", 0.0)
                        if geography_name:
                            geography_node = get_or_create_geographies_node(geography_name)
                            geography_id = geography_node["id"]
                            # Add edge: Pain Trigger → Geography
                            if pain_trigger_node_id and geography_id:
                                add_edge(
                                    product_id=product_id,
                                    source_id=pain_trigger_node_id,
                                    target_id=geography_id,
                                    edge_type="experienced_in",
                                    weight=match_score,
                                    last_updated=now,
                                    source=source
                                )
                
                zmot_events = raw_pain_trigger.get("zmot_events")
                for zmot_event in zmot_events:
                    raw_trigger_event = zmot_event.get("trigger_event", "").strip()
                    canonical_trigger_event = canonical_zmot_trigger_events.get(raw_trigger_event, raw_trigger_event)
                    trigger_event_key = canonical_trigger_event.strip().lower()
                    match_score = zmot_event.get("match_score", 0.0)
                    if trigger_event_key in zmot_trigger_event_lookup:
                        zmot_trigger_event_node_id = zmot_trigger_event_lookup[trigger_event_key]
                    else:
                        print("❌ ZMOT trigger event lookup failed for key:", trigger_event_key)
                        print("Available trigger event keys:", list(zmot_trigger_event_lookup.keys()))

                    # Add edge: Pain Trigger → ZMOT Trigger Event
                    if pain_trigger_node_id and zmot_trigger_event_node_id:
                        add_edge(
                            product_id=product_id,
                            source_id=pain_trigger_node_id,
                            target_id=zmot_trigger_event_node_id,
                            edge_type="triggered_by",
                            weight= match_score,
                            last_updated=now,
                            source=source
                        )


                    observable_moments = zmot_event.get("observable_moments", [])
                    if isinstance(observable_moments, dict):
                        observable_moments = [observable_moments]
                    for observable_moment in observable_moments:
                        observable_moment_text = observable_moment.get("observable_moment", "").strip().lower()
                        if observable_moment_text in zmot_observable_moment_lookup:
                            zmot_observable_moment_node_id = zmot_observable_moment_lookup[observable_moment_text]
                            match_score = observable_moment.get("match_score", 0.0)
                            # Add edge: ZMOT Trigger Event → Observable Moment
                            if zmot_trigger_event_node_id and zmot_observable_moment_node_id:
                                add_edge(
                                    product_id=product_id,
                                    source_id=zmot_trigger_event_node_id,
                                    target_id=zmot_observable_moment_node_id,
                                    edge_type="observed_in",
                                    weight= match_score,
                                    last_updated=now,
                                    source=source
                                )
                    keywords = zmot_event.get("trigger_keywords", [])
                    if isinstance(keywords, dict):
                        keywords = [keywords]
                    for keyword in keywords:
                        keyword_text = keyword.get("trigger_keyword", "").strip()
                        if keyword_text in zmot_keyword_lookup:
                            zmot_keyword_node_id = zmot_keyword_lookup[keyword_text]
                            match_score = keyword.get("match_score", 0.0)
                            # Add edge: ZMOT Trigger Event → Keyword
                            if zmot_trigger_event_node_id and zmot_keyword_node_id:
                                add_edge(
                                    product_id=product_id,
                                    source_id=zmot_trigger_event_node_id,
                                    target_id=zmot_keyword_node_id,
                                    edge_type="associated_with",
                                    weight= match_score,
                                    last_updated=now,
                                    source=source
                                )


    print("Finished Graph and Edge math for zmot & ICP:", gpt_results)
    return gpt_results


def canonicalize_and_create_hop_plus_nodes(gpt_results, product_id):
    print("Starting node creation & processing for zmot & ICP")
    jobs_cache = set()
    pains_cache = set()
    personas_cache = set()
    results_job_ids = []

    # Gather unique raw values for canonicalization
    for entry in gpt_results:
        entry["source"] = "openai"
        original_job_id = entry.get("original_job_id").strip()
        if not original_job_id:
            print("❌ Job ID is missing in entry:", entry)
            continue
        if entry.get("pain_source") == "external":
            
            print("⚠️ Job is external, skipping pain processing for this entry in job:", original_job_id)
            continue

        pains = entry.get("pains", [])
        if isinstance(pains, list):
            for pain in pains:
                raw_pain_text = pain.get("pain", "").strip()
                if raw_pain_text:
                    pains_cache.add(raw_pain_text)
                else:
                    print("❌ Pain text is missing in entry:", pain)
                jobs = pain.get("jobs", [])
                if isinstance(jobs, list):
                    for job in jobs:
                        raw_job_description = job.get("description", "").strip()
                        if raw_job_description:
                            jobs_cache.add(raw_job_description)
                        else:
                            print("❌ Job text is missing in entry:", job)
                        personas = job.get("personas", [])
                        if not isinstance(personas, list):
                            print("❌ Invalid personas format in entry:", entry)
                            continue
                        for persona in personas:
                            raw_persona = {
                                "title": persona.get("title", "").strip(),
                                "department": persona.get("department", "").strip(),
                                "seniority": persona.get("seniority", "").strip()
                            }
                            if raw_persona.get("title") or raw_persona.get("department") or raw_persona.get("seniority"):
                                personas_cache.add((
                                    raw_persona.get("title"),
                                    raw_persona.get("department"),
                                    raw_persona.get("seniority")
                                ))
                            else:
                                print("❌ Persona fields are empty in entry:", entry)
    
    # At this point we have: (1) The exact structure of gpt output, (2) Raw values added to each cache
    print(" Gathered raw caches with sizes:")
    print(" - Jobs cache size:", len(jobs_cache))
    print(" - Pains cache size:", len(pains_cache))
    print(" - Personas cache size:", len(personas_cache))

    # Canonicalize raw caches
    canonical_personas = canonicalize_persona([
        {"title": t, "department": d, "seniority": s}
        for (t, d, s) in personas_cache
    ])
    canonical_pains = canonicalize_pain(list(pains_cache))
    canonical_jobs = canonicalize_job(list(jobs_cache))

    # Lookup logic for setting right node IDs to entry list
    persona_lookup = {
        str({
            "title": p["title"].strip(),
            "department": p["department"].strip(),
            "seniority": p["seniority"].strip()
        }): get_or_create_persona_node(p["title"], p["department"], p["seniority"])["id"]
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
    print("Lookup dictionaries created with sizes. Starting final node and edge processing")
    for entry in gpt_results:
        print("--- ---- --- ---- --- ---- ---")
        print("Processing entry:", entry)

        source = entry.get("source", "openai")
        now = datetime.utcnow().isoformat()
        original_job_id = entry.get("original_job_id").strip()
        pains = entry.get("pains", [])
        for pain in pains:
            raw_pain_text = pain.get("pain", "").strip()
            pain_canonical = canonical_pains.get(raw_pain_text, raw_pain_text)

            pain_lookup_key = pain_canonical.strip().lower()
            if pain_lookup_key not in pain_lookup:
                print("❌ Pain lookup failed for key:", pain_lookup_key)
                print("Available pain keys:", list(pain_lookup.keys()))
            pain_node_id = pain_lookup.get(pain_lookup_key)

            severity = pain.get("severity", 0.5)

            # Add original job - pain entry
            add_edge(
                product_id=product_id,
                source_id=original_job_id,
                target_id=pain_node_id,
                edge_type="solves",
                weight=severity,
                last_updated=now,
                source=source
            )
            print(f"Added edge from Original Job {original_job_id} to Pain {pain_node_id} with weight {severity}")
                
            jobs = pain.get("jobs", [])
            for job in jobs:
                raw_job = job.get("description", "").strip()
                job_canonical = canonical_jobs.get(raw_job, raw_job)

                job_lookup_key = job_canonical.strip().lower()
                if job_lookup_key not in job_lookup:
                    print("❌ Job lookup failed for key:", job_lookup_key)
                    print("Available job keys:", list(job_lookup.keys()))
                job_node_id = job_lookup.get(job_lookup_key)
                impact = job.get("impact", 0.5)

                # Add Pain -> Job edge
                add_edge(
                    product_id=product_id,
                    source_id=pain_node_id,
                    target_id=job_node_id,
                    edge_type="addresses",
                    weight=impact,
                    last_updated=now,
                    source=source
                )
                print(f"Added edge from Pain {pain_node_id} to Job {job_node_id} with weight {impact}")
                
                # Extra logic to add newly added jobs to the return loop
                if job_node_id not in results_job_ids:
                    results_job_ids.append(job_node_id)

                personas = job.get("personas", [])
                for raw_persona in personas:
                    persona_key = str({
                        "title": raw_persona.get("title", "").strip(),
                        "department": raw_persona.get("department", "").strip(),
                        "seniority": raw_persona.get("seniority", "").strip()
                    })
                    persona_canonical = canonical_personas.get(persona_key, raw_persona)
                    persona = persona_canonical
                    persona_title = persona.get("title", "").strip()
                    persona_department = persona.get("department", "").strip()
                    persona_seniority = persona.get("seniority", "").strip()
                    
                    if persona_title == "" or persona_department == "" or persona_seniority == "":
                        print("❌ Persona fields are empty in lookup:", entry)
                        continue
                    print("Persona canonicalization complete - moving to lookup")
                    # Lookup node IDs using canonicalized values
                    persona_lookup_key = str({
                        "title": persona_title.strip(),
                        "department": persona_department.strip(),
                        "seniority": persona_seniority.strip()
                    })
                    print("Persona lookup key:", persona_lookup_key)
                    if persona_lookup_key not in persona_lookup:
                        print("❌ Persona lookup failed for key:", persona_lookup_key)
                        print("Available persona keys:", list(persona_lookup.keys()))
                    
                    persona_node_id = persona_lookup.get(persona_lookup_key)
                    job_importance = persona.get("job_importance", 0.5)  # Default to 0.5 if not specified
                    #Add edge: Job → Persona
                    add_edge(
                        product_id=product_id,
                        source_id=job_node_id,
                        target_id=persona_node_id,
                        edge_type="performed_by",
                        weight=job_importance,
                        last_updated=now,
                        source=source
                    )
                    

    print("Finished Graph and Edge math for Hop++:", gpt_results)

    return results_job_ids
