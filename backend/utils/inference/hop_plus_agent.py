import json
<<<<<<< Updated upstream:backend/utils/inference/hop_plus_agent.py
from typing import List, Dict, Any, Set
from backend.utils.graph_base.nodes.pain_trigger_nodes import get_or_create_pain_trigger_node
from backend.utils.graph_base.relevance.cumulative_relevance_manager import get_cumulative_relevance_data 
from backend.utils.inference.hop_plus_openai import get_upstream_triplets
=======
>>>>>>> Stashed changes:backend/utils/inference/discovery_engine/hop_plus_agent.py
from backend.utils.graph_base.graph import Graph
from backend.utils.graph_base.graph_builder import canonicalize_and_create_hop_plus_nodes
from backend.utils.graph_base.relevance.cumulative_relevance_manager import add_or_update_cumulative_relevance_data, get_cumulative_relevance_data
from backend.utils.inference.gpt_prompts.hop_plus_openai import get_upstream_triplets


def infer_upstream_with_rules(
    product_subgraph: Graph,
    cap_threshold: float = 0.1,
    relevance_threshold: float = 0.5,
    max_depth: int = 3
):
    #1. Get product ID, and initialize arrays/ sets
    product_id = product_subgraph.get_node_id("product", {})
    if not product_id:
        raise ValueError("Product ID not found in the provided subgraph.")

    results = []
    visited_jobs = set()
    jobs_holder = []
    gpt_jobs_cache = []
    current_depth = 0


    #2. Get all Jobs in subgraph and add them to jobs holder
    jobs_list = product_subgraph.get_nodes_list_ids("job", {})
    if not jobs_list:
        raise ValueError("No jobs found in the provided subgraph.")
    jobs_ids = set(jobs_list)
    for job_id in jobs_ids:
        job_node = product_subgraph.get_node_by_id(job_id)
        if job_node and job_id not in jobs_holder:
            print("Adding job id to jobs holder: ", job_id)
            jobs_holder.append(job_id)
    print("Total jobs holder contents-> ", jobs_holder)
    print("----------------------------")
    traverse_upstream_jobs(jobs_holder, product_subgraph, gpt_jobs_cache, visited_jobs, current_depth)


#3 core traversal logic to get upstream jobs:
# Sets next most relevant job from process holder - then gets upstream jobs for that job. If upstream exists - adds them to the holder for sorting, if upstream doesn't exist - adds current to gpt_cache for future processing.
def traverse_upstream_jobs(jobs_holder, product_subgraph, gpt_jobs_cache, visited_jobs, current_depth=0):
    product_id = product_subgraph.get_node_id("product", {})
    print("Starting traverse while loop")
    i = 0
    while jobs_holder:
        print("While loop at cycle: ", i)
        i+=1

        current_job_id = process_jobs_holder(jobs_holder, product_subgraph)
        print("Current job id: ", current_job_id)
        current_job = product_subgraph.get_node_by_id(current_job_id)
        if current_job:
            if current_job_id in visited_jobs:
                jobs_holder.remove(current_job_id)
                print("Job already visited, removing from jobs holder: ", current_job_id)
                continue
            visited_jobs.add(current_job_id)
            jobs_holder.remove(current_job_id)
            print("Before processing upstream for this node")
            next_job_ids = get_upstream_job_ids(current_job_id, product_subgraph)
            if next_job_ids and len(next_job_ids) > 0:
                print("Next job IDs found: ", next_job_ids," ... processing into jobs_holder...")
                for next_job_id in next_job_ids:
                    if next_job_id not in jobs_holder and next_job_id not in visited_jobs:
                        print("Adding next job to jobs holder: ", next_job_id)
                        jobs_holder.append(next_job_id)
            else:
                print("No upstream jobs found. Adding to GPT Cache")
                gpt_jobs_cache.append(current_job_id)
        else:
            print("current job in holder is malformed - diagnose this")
    process_gpt_cache(gpt_jobs_cache, product_subgraph,jobs_holder, visited_jobs, current_depth)

    return []

#3. logic to get highest relevance job from jobs_holder: 
# if jobs_holder has any data - returns next highest relevance job, else []
def process_jobs_holder(jobs_holder, product_subgraph):
    print("Processing jobs holder")
    product_id = product_subgraph.get_node_id("product", {})
    if not jobs_holder:
        return []
    if len(jobs_holder) < 1:
        return []
    job_relevance_list = []
    for job_id in jobs_holder:
        print("figuring relevance")
        job = product_subgraph.get_node_by_id(job_id)
        relevance = get_cumulative_relevance_data(product_id, job.get("id"))
        job_relevance_list.append({
            "id": job_id,
            "relevance": relevance
        })
    job_relevance_list.sort(key=lambda x: x.get("relevance", 0), reverse=True)
    if job_relevance_list:
        print("Jobs holder sorted list in this cycle:", job_relevance_list)
        highest_relevance_job_id = job_relevance_list[0]["id"]
        return highest_relevance_job_id
    else:
        return None

#4. logic to get upstream jobs from the next job
def get_upstream_job_ids(next_job_id, product_subgraph):
    upstream_job_nodes = []
    if not next_job_id:
        print("No next job provided.")
        return []
    upstream_pain_ids = product_subgraph.get_target_nodes_by_source_and_type(next_job_id, "impacts")
    if not upstream_pain_ids:
        return []
    for upstream_pain_id in upstream_pain_ids:
        upstream_job_ids = product_subgraph.get_target_nodes_by_source_and_type(upstream_pain_id, "addresses")
        if not upstream_job_ids:
            print(f"No upstream jobs found for pain ID: {upstream_pain_id}")
            continue
        print("Upstream job IDs found:", upstream_job_ids)
        upstream_job_nodes.extend(upstream_job_ids)
    return upstream_job_nodes

#5 logic to process_gpt_cache: 
def process_gpt_cache(gpt_jobs_cache, product_subgraph, jobs_holder, visited_jobs, current_depth = 0):
    if current_depth > 4:
        print("Max depth reached, stopping further processing.")
        return []
    print("Processing GPT cache at depth: ", current_depth)
    product_id = product_subgraph.get_node_id("product", {})
    product_node = product_subgraph.get_node_by_id(product_id)
    summary = product_node.get("summary", "")
    if not gpt_jobs_cache:
        print("No jobs in GPT cache to process.")
        return []
    
    # Here you would typically call the GPT model with the jobs in the cache
    # For now, we will just print them
    print("Processing GPT cache with jobs:", gpt_jobs_cache)
    final_gpt_buffer = []
    for job_id in gpt_jobs_cache:
        personas_list = []
        pains_list = []
        job = product_subgraph.get_node_by_id(job_id)
        linked_personas = product_subgraph.get_target_nodes_by_source_and_type(job_id, "performed_by")
        linked_pains = product_subgraph.get_source_nodes_by_target_and_type(job_id, "addresses")
        if not linked_personas or not linked_pains:
            print(f"Something is wrong - {job_id} has no linked pains/ jobs.")
            continue
        for persona in linked_personas:
            persona_node = product_subgraph.get_node_by_id(persona)
            persona_title = persona_node.get("title")
            persona_department = persona_node.get("department")
            persona_seniority = persona_node.get("seniority")
            personas_list.append({
                "title": persona_title,
                "department": persona_department,
                "seniority": persona_seniority
            })
        for pain in linked_pains:
            pain_node = product_subgraph.get_node_by_id(pain)
            pain_text = pain_node.get("text")
            pains_list.append({
                "pain": pain_text
            })
            
        final_gpt_buffer.append({
            "job_id": job_id,
            "description": job.get("description"),
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
        gpt_output = get_upstream_triplets(summary, batch)
        print("GPT results for batch:", gpt_output)
        if gpt_output:
            gpt_outputs.extend(gpt_output)
    print("GPT results: ", gpt_outputs)

    # Canonicalize & Process
    print("📊 [Graph] Canonicalizing Hop++ map...")
    if not gpt_outputs:
        print("⚠️ No valid data found in OpenAI output. Cannot proceed.")
        return {
            "hop_plus_map": [],
            "source": "empty",
            "error": "No valid data found in OpenAI output."
        }
    hop_plus_new_job_ids = canonicalize_and_create_hop_plus_nodes(gpt_outputs)
    # Clear the cache after processing
    gpt_jobs_cache.clear()

    print("ICP & ZMOT Graph processed. Results follow:", hop_plus_new_job_ids)

    print("Updating cumulative relevance")
    relevance_nodes = product_subgraph.calculate_cumulative_relevance()
    add_or_update_cumulative_relevance_data(product_id, relevance_nodes)
    print("Cumulative relevance json updated successfully.")

    for job_id in hop_plus_new_job_ids:
        if job_id not in visited_jobs:
            if job_id not in jobs_holder:
                print("Adding new job id to jobs holder: ", job_id)
                jobs_holder.append(job_id)

    traverse_upstream_jobs(jobs_holder, product_subgraph, gpt_jobs_cache, visited_jobs, current_depth+1)

    return {
            "pains_map": hop_plus_new_job_ids,
            "ZMOT_ICP_Nodes": hop_plus_new_job_ids
    }