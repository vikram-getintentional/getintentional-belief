from collections import defaultdict
import json
from typing import List, Dict, Any, Set
from backend.utils.graph_base.agent_graph_builder import build_hop0_graph, build_hop_plus_graph, build_icp_graph, build_zmot_nodes_to_graph
from backend.utils.graph_base.network_graph import calculate_soft_or_relevance, get_cumulative_relevance, get_node_by_id, get_node_id, get_nodes_list_ids, get_source_nodes_by_target_and_type, get_target_nodes_by_source_and_type, update_capability_centralities, update_graph
import networkx as nx
from backend.utils.graph_base.nodes.job_nodes import get_or_create_job_node
from backend.utils.inference.discovery_engine.agentic_engine.agent_context import AgentContext
from backend.utils.inference.gpt_prompts.agentic_prompts import  infer_icp, infer_pain_and_source, infer_upstream_for_internal_jobs
from backend.utils.graph_base.network_graph import update_graph

from backend.utils.graph_base.relevance.cumulative_relevance_manager import add_or_update_cumulative_relevance_data
from backend.utils.inference.gpt_prompts.agentic_prompts import infer_zmot_for_external_jobs
from backend.utils.inference.gpt_prompts.capability_to_pain_persona import infer_persona_job_pain_from_capabilities


def pain_source_inference(product_subgraph: nx.DiGraph, jobs_list: List[Dict[str, Any]], context: AgentContext) -> List[Dict[str, Any]]:
    results = []
    product_id = get_node_id(product_subgraph, "product", {})
    product_node = get_node_by_id(product_subgraph, product_id)
    summary = product_node.get("summary", "")
    domain = product_node.get("domain", "")
    industry = product_node.get("industry", "")
    job_sets = []
    print("Starting Pain Source Inference...in disco engine")

    for job_id in jobs_list:
        print(f"Processing pain inf in disco engine for job ID: {job_id}")
        job_node = get_node_by_id(product_subgraph, job_id)
        if not job_node:
            print(f"⚠️ Job node {job_id} not found in the product subgraph.")
            continue
        pain_source = job_node.get("pain_source", "")
        if not pain_source or pain_source.strip().lower() not in ["internal", "external"]:
            print(f"⚠️ Job {job_id} has no valid pain source. Running Pain Source Inference.")
            dependent_pains = get_source_nodes_by_target_and_type(product_subgraph, job_id, "addresses")
            pains = []
            for pain_id in dependent_pains:
                pain_node = get_node_by_id(product_subgraph, pain_id)
                if pain_node:
                    pains.append({
                        "pain": pain_node.get("description", "")
                    })
            personas = []
            performed_personas = get_target_nodes_by_source_and_type(product_subgraph, job_id, "performed_by")
            if not performed_personas:
                print(f"⚠️ No personas found for job {job_id}.")
                continue

            for persona_id in performed_personas:
                persona_node = get_node_by_id(product_subgraph, persona_id)
                if persona_node:
                    personas.append({
                        "persona_title": persona_node.get("title", ""),
                        "persona_seniority": persona_node.get("seniority", ""),
                        "persona_description": persona_node.get("department", "")
                    })
            job_sets.append({
                "original_job_id": job_id,
                "description": job_node.get("description", ""),
                "pains": pains,
                "personas": personas,
            })
    print("Starting GPT loop for pain source inference...")
    pain_source_results = infer_pain_and_source(summary, domain, industry, job_sets)

    if not pain_source_results:
        print("⚠️ No valid pain source results found from GPT. Cannot proceed.")
        return []
    # After calling infer_pain_and_source
    if isinstance(pain_source_results, str):
        print("⚠️ pain_source_results is a string, attempting to parse as JSON...")
        try:
            pain_source_results = json.loads(pain_source_results)
        except Exception as e:
            print("❌ Failed to parse pain_source_results as JSON:", e)
            return []
    # Flatten if pain_source_results is a list of lists
    if pain_source_results and isinstance(pain_source_results[0], list):
        print("⚠️ Detected nested list in pain_source_results, flattening...")
        pain_source_results = [item for sublist in pain_source_results for item in sublist]

    print("Processed pain Source Results from GPT:", pain_source_results)
    final_pain_source_results = []
    print("Starting entry loop")
    for entry in pain_source_results:
        print("Processing entry in pain source inference:", entry)
        if not isinstance(entry, dict):
            print(f"⚠️ Skipping non-dict entry in pain_source_results: {entry} (type: {type(entry)})")
            continue
        original_job_id = entry.get("original_job_id").strip().lower()
        pain_source = entry.get("pain_source", "").strip().lower()
        if pain_source not in ["internal", "external"]:
            print(f"⚠️ Invalid pain source '{pain_source}' for job {original_job_id}. Skipping.")
            continue
        print("Before get or create job node")
        get_or_create_job_node(node_id=original_job_id, pain_source=pain_source)
        print("After get or create job node")
        final_pain_source_results.append(original_job_id)

    print("Pain Source Inference Completed.")
    update_graph(product_subgraph)

    return final_pain_source_results

def hop0_inference(product_id, product_subgraph: nx.DiGraph, context: AgentContext, force_openai=False, retry_depth=0) -> dict:
    print("Starting Hop0 inference loop: ", product_id)
    from datetime import datetime
    product_node = get_node_by_id(product_subgraph, product_id)
    if not product_node:
        raise ValueError("Product node not found.")
    summary = product_node.get("summary")
    domain = product_node.get("domain", "")
    industry = product_node.get("industry", "")
    capability_ids = get_target_nodes_by_source_and_type(product_subgraph, product_id, "offered_by")
    capabilities = [
        get_node_by_id(product_subgraph, cid)
        for cid in capability_ids
        if get_node_by_id(product_subgraph, cid)
    ]

    if not summary or not capabilities:
        print("⚠️ Summary or capabilities are empty. Cannot proceed with inference.")
        return {
            "capability_map": [],
            "personas": [],
            "aggregated_personas": [],
            "source": "empty",
            "error": "Summary or capabilities are empty."
        } 

    # Filter valid capabilities
    filtered_capabilities = []
    for c in capabilities:
        if c.get("name") and c.get("description"):
            filtered_capabilities.append({
                "id": c.get("id"), 
                "name": c.get("name"), 
                "description": c.get("description")
                })
            
    print("Filtered capabilities in infer:" , filtered_capabilities)

    capability_ids_list = [cap.get("id") for cap in filtered_capabilities]

    print("Loaded graph with nodes:", len(product_subgraph.nodes), "edges:", len(product_subgraph.edges))

    try:
        print("🔁 Running GPT reasoning...")
        capability_map = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        
        
        print("🧠 [GPT] Generating capability map...")
        gpt_output = infer_persona_job_pain_from_capabilities(summary, domain, industry, filtered_capabilities)

        """
        Output is of format:
        - capability_id: string
        - capability: string
        - pains: list of
            - pain: string
            - relevance: array of floats
            - pain_triggers: list of
                - attribute: string
                - dimension: string
                - direction: string
            - jobs: list of
                - description: string
                - impact: float
                - personas: list of
                    - job_importance: float
                    - title: string
                    - department: string
                    - seniority: string
        """

        if not gpt_output:
            print("⚠️ No valid data found in OpenAI output. Cannot proceed.")
            return {
                "capability_map": [],
                "personas": [],
                "aggregated_personas": [],
                "source": "empty",
                "error": "No valid data found in OpenAI output."
            }
        print("GPT output:", gpt_output)

        # Code to canonicalize GPT output and convert into nodes and edges
        capability_map = build_hop0_graph(gpt_output, capability_ids_list, product_id)

        print("\n\nFlattened capability map in Hop0 traversal")
        print(json.dumps(capability_map, indent=2))

        # Step 3: Normalize and canonicalize the flattened capability map
        print("📊 [Graph] Canonicalizing capability map...")
        if not capability_map:
            print("⚠️ No valid capability data found in OpenAI output. Cannot proceed.")
            return {
                "capability_map": [],
                "personas": [],
                "aggregated_personas": [],
                "source": "empty",
                "error": "No valid capability data found."
            }
        
        update_graph(product_subgraph)
        hop0_jobs = get_nodes_list_ids(product_subgraph, "job", {})


        return hop0_jobs

    except Exception as e:
        import traceback
        print("❌ Error in Hop0 Reasoning:", e)
        traceback.print_exc()
        return {
            "capability_map": [],
            "personas": [],
            "error": str(e)
        }


#5 logic to process_gpt_cache: 
def process_hop_plus_gpt_cache(gpt_jobs_cache, product_subgraph, context: AgentContext, current_depth = 0):
    try:
        if current_depth > 4:
            print("Max depth reached, stopping further processing.")
            return []
        print("Processing GPT cache at depth: ", current_depth)
        product_id = get_node_id(product_subgraph, "product",{})
        product_node = get_node_by_id(product_subgraph, product_id)
        summary = product_node.get("summary", "")
        domain = product_node.get("domain", "")
        industry = product_node.get("industry", "")
        if not gpt_jobs_cache:
            print("No jobs in GPT cache to process.")
            return []
        
        print("Processing GPT cache with jobs:", gpt_jobs_cache)
        final_gpt_buffer = []
        for job_id in gpt_jobs_cache:
            personas_list = []
            pains_list = []
            job = get_node_by_id(product_subgraph, job_id)
            linked_personas = get_target_nodes_by_source_and_type(product_subgraph, job_id, "performed_by")
            linked_pains = get_source_nodes_by_target_and_type(product_subgraph, job_id, "addresses")
            if not linked_personas or not linked_pains:
                print(f"Something is wrong - {job_id} has no linked pains/ jobs.")
                continue
            for persona in linked_personas:
                persona_node = get_node_by_id(product_subgraph, persona)
                persona_title = persona_node.get("title")
                persona_department = persona_node.get("department")
                persona_seniority = persona_node.get("seniority")
                personas_list.append({
                    "title": persona_title,
                    "department": persona_department,
                    "seniority": persona_seniority
                })
            for pain in linked_pains:
                pain_node = get_node_by_id(product_subgraph, pain)
                pain_text = pain_node.get("text")
                pains_list.append({
                    "pain": pain_text
                })
                
            final_gpt_buffer.append({
                "job_id": job_id,
                "job_to_be_done": job.get("description"),
                "personas": personas_list,
                "pains": pains_list
            })
        print("Final GPT buffer to process: ")
        for buffer_item in final_gpt_buffer:
            print(buffer_item)
            print("--------------------------------------------------")
        

        # Batch the GPT queries
        batch_size = 5 # Setting a soft limit of 5 items for each GPT query
        gpt_outputs = []
        for i in range(0, len(final_gpt_buffer), batch_size):
            batch = final_gpt_buffer[i:i+batch_size]
            print(f"Running GPT query for batch {i//batch_size + 1}: {batch}")
            gpt_output = infer_upstream_for_internal_jobs(summary, domain, industry, batch)
            print("GPT results for batch:", gpt_output)
            if isinstance(gpt_output, str):
                try:
                    gpt_output = json.loads(gpt_output)
                except Exception as e:
                    print("❌ Failed to parse GPT output as JSON:", e)
                    continue
            if isinstance(gpt_output, list):
                gpt_outputs.extend(gpt_output)
            elif isinstance(gpt_output, dict):
                gpt_outputs.append(gpt_output)
            else:
                print("⚠️ Unexpected GPT output type:", type(gpt_output))
            

        # Canonicalize & Process
        print("📊 [Graph] Canonicalizing Hop++ map...")
        if not gpt_outputs:
            print("⚠️ No valid data found in OpenAI output. Cannot proceed.")
            return []
        
        print(json.dumps(gpt_outputs, indent=2))
        hop_plus_new_job_ids = build_hop_plus_graph(gpt_outputs, product_id)
        # Clear the cache after processing
        output_job_ids = set()
        for entry in gpt_outputs:
            # Adjust this if your output structure is different
            output_job_ids.add(entry.get("original_job_id") or entry.get("job_id"))

        for job_id in gpt_jobs_cache:
            if job_id in output_job_ids:
                context.mark_visited(job_id)
        gpt_jobs_cache.clear()

        print("Hop+ Graph Created. Results follow:", hop_plus_new_job_ids)

        print("Updating cumulative relevance")
        relevance_nodes = calculate_soft_or_relevance(product_subgraph)
        cumulative_relevance = {item["node_id"]: item["relevance"] for item in relevance_nodes}
        add_or_update_cumulative_relevance_data(product_id, cumulative_relevance)
        print("Cumulative relevance json updated successfully.")

        for job_id in hop_plus_new_job_ids:
            if job_id not in context.visited_jobs:
                    print("Adding new job id to jobs holder: ", job_id)
                    context.mark_visited(job_id)
        
        product_subgraph = update_graph(product_subgraph)

        return hop_plus_new_job_ids
    finally:
        gpt_jobs_cache.clear()


def process_zmot_gpt_cache(gpt_zmot_cache, product_subgraph, context: AgentContext, current_depth = 0):
    product_id = get_node_id(product_subgraph, "product",{})
    product_node = get_node_by_id(product_subgraph, product_id)
    if not product_node:
        raise ValueError("Product node not found.")
    summary = product_node.get("summary", "")
    domain = product_node.get("domain", "")
    industry = product_node.get("industry", "")

    gpt_input_buffer = []

    print("Starting zmot inference loop: ", product_id)
    from datetime import datetime
    # Updated query logic
    for job_id in gpt_zmot_cache:
        print("Processing ZMOT for job: ", job_id)
        job_node = get_node_by_id(product_subgraph, job_id)
        if not job_node:
            print(f"Job node {job_id} not found in the product subgraph.")
            continue
        # Only process external jobs without existing zmot links
        if job_node.get("pain_source", "") != "external":
            continue

        existing_zmots = get_target_nodes_by_source_and_type(product_subgraph, job_id, "zmot")
        if existing_zmots:
            continue

        job_description = job_node.get("description", "")
        solved_pain_ids = get_source_nodes_by_target_and_type(product_subgraph, job_id, "addresses")
        if not solved_pain_ids:
            print(f"No solved pains found for job {job_id}. Something is wrong.")
            continue
        pains = []
        for pain_id in solved_pain_ids:
            pain_node = get_node_by_id(product_subgraph, pain_id)
            pain_text = pain_node.get("text", "")
            pains.append({"pain_text": pain_text})
            
            # Get personas performing the job
        performed_by_personas = get_target_nodes_by_source_and_type(product_subgraph, job_id, "performed_by")
        personas = []
        for persona_id in performed_by_personas:
            persona_node = get_node_by_id(product_subgraph, persona_id)
            if not persona_node:
                print(f"Persona node {persona_id} not found in the product subgraph.")
                continue
            persona_title = persona_node.get("title", "")
            persona_department = persona_node.get("department", "")
            persona_seniority = persona_node.get("seniority", "")
            personas.append({
                "title": persona_title,
                "department": persona_department,
                "seniority": persona_seniority
            })

            # Prepare the job data for GPT processing
            gpt_input_buffer.append({
                "job_id": job_id,
                "job_to_be_done": job_description,
                "personas": personas,
                "pains": pains
            })

            if not gpt_input_buffer:
                print("⚠️ No eligible external jobs for ZMOT inference.")
                return []


    try:
        print("🔁 Running GPT reasoning...")
        pains_map = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        
        # ---- Batching logic for GPT calls ----
        def batch_list(lst, n):
            for i in range(0, len(lst), n):
                yield lst[i:i + n]

        print("🧠 [GPT] Generating ZMOTs map...")
        gpt_output = []
        batch_size = 3  # You can tune this for your token limits
        for batch in batch_list(gpt_input_buffer, batch_size):
            print("🧠 [GPT] Generating ZMOTs map for batch...")
            batch_output = infer_zmot_for_external_jobs(summary, industry, domain, batch)
            if isinstance(batch_output, str):
                batch_output = json.loads(batch_output)
            gpt_output = batch_output

            print("\n\nCurrent Batch of entries map in ZMOT discovery")
            print(json.dumps(gpt_output, indent=2))

            # Step 3: Normalize and canonicalize the flattened capability map
            print("📊 [Graph] Canonicalizing ZMOT map...")
            if not gpt_output:
                print("⚠️ No valid data found in OpenAI output. Cannot proceed.")
                return {
                    "pains_map": [],
                    "source": "empty",
                    "error": "No valid data found in OpenAI output."
                }
            zmot_map = build_zmot_nodes_to_graph(gpt_output, product_id)
            output_job_ids = set()
            for entry in gpt_output:
                # Adjust this if your output structure is different
                output_job_ids.add(entry.get("original_job_id") or entry.get("job_id"))

            for job_id in gpt_zmot_cache:
                if job_id in output_job_ids:
                    context.mark_visited(job_id)
            print("ICP & ZMOT Graph processed for batch. Results follow:", zmot_map)
        print("All ZMOT processing complete.")

        for job_id in gpt_zmot_cache:
            if job_id not in context.visited_jobs:
                print("Some job ID was not processed in ZMOT: ", job_id)
        gpt_zmot_cache.clear()
        

        #print("Updating cumulative relevance")
        #relevance_nodes = get_cumulative_relevance(product_subgraph)
        #add_or_update_cumulative_relevance_data(product_id, relevance_nodes)
        print("Cumulative relevance json updated successfully.")

        return []

    except Exception as e:
        import traceback
        print("❌ Error in ZMOT Reasoning:", e)
        traceback.print_exc()
        return []

def generate_icps_for_hop0( product_subgraph: nx.DiGraph, context: AgentContext) -> nx.DiGraph:
    product_id = get_node_id(product_subgraph, "product", {})
    if not product_id:
        raise ValueError("❌ Product ID not found in subgraph.")
    icp_archetype_ids = get_nodes_list_ids(product_subgraph, "icp_archetype", {})
    if not icp_archetype_ids:
        print("⚠️ No ICP archetypes found in the product subgraph. Running Inference.")
        capabilities_map = {}
        pains = []
        capabilities = []
        jobs = []
        personas = []
        
        capabilities = get_nodes_list_ids(product_subgraph, "capability", {})
        if not capabilities:
            print("⚠️ No capabilities found in the product subgraph. Cannot proceed with inference.")
            return []
        for cap in capabilities:
            cap_node = get_node_by_id(product_subgraph, cap)
            if cap_node:
                cap_description = cap_node.get("description", "")
                capabilities.append(cap_description)
                primary_pain_ids = get_target_nodes_by_source_and_type(product_subgraph, cap, "solves")
                if not primary_pain_ids:
                    print(f"⚠️ No primary pains found for capability {cap}.")
                    continue
                for pain_id in primary_pain_ids:
                    pain_node = get_node_by_id(product_subgraph, pain_id)
                    if pain_node:
                        pain_text = pain_node.get("text", "")
                        if pain_text not in pains:
                            pains.append(pain_text)
                        primary_job_ids = get_target_nodes_by_source_and_type(product_subgraph, pain_id, "addresses")
                        if not primary_job_ids:
                            print(f"⚠️ No primary jobs found for pain {pain_id}.")
                            continue
                        for job_id in primary_job_ids:
                            job_node = get_node_by_id(product_subgraph, job_id)
                            if job_node:
                                job_description = job_node.get("description", "")
                                if job_description not in jobs:
                                    jobs.append(job_description)
                            primary_persona_ids = get_target_nodes_by_source_and_type(product_subgraph, job_id, "performed_by")
                            if not primary_persona_ids:
                                print(f"⚠️ No primary personas found for job {job_id}.")
                                continue
                            for persona_id in primary_persona_ids:
                                persona_node = get_node_by_id(product_subgraph, persona_id)
                                if persona_node:
                                    persona_title = persona_node.get("title", "")
                                    persona_department = persona_node.get("department", "")
                                    persona_seniority = persona_node.get("seniority", "")
                                    personas.append({
                                        "title": persona_title,
                                        "department": persona_department,
                                        "seniority": persona_seniority
                                    })

        capabilities_map = {
                "pains": pains,
                "jobs_to_be_done": jobs,
                "capabilities_offered": capabilities,
                "personas": personas
            }
        product_node = get_node_by_id(product_subgraph, product_id)
        summary = product_node.get("summary", "")
        domain = product_node.get("domain", "")
        industry = product_node.get("industry", "")

        try:
            print("🔁 Running GPT reasoning for ICPs...")
            icp_gpt_output = infer_icp(summary, industry, domain, capabilities_map)
            if not icp_gpt_output:
                print("⚠️ No valid data found in OpenAI output. Cannot proceed.")
                return product_subgraph
            print("GPT output for ICPs:", icp_gpt_output)
            icp_map = build_icp_graph(icp_gpt_output, product_id)

        except Exception as e:
            print("❌ Error in ICP Reasoning:", e)
            import traceback
            traceback.print_exc()
            return product_subgraph
        


    update_graph(product_subgraph)
    return product_subgraph