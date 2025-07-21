"""
What's this file?
This should eventually be the primary traversal agent for the entire graph discovery (hop 0, hop++, hop --, blockers)
The output of this will be to build the complete persona graph for a product ID by calling the corresponding GPT helper functions as required.
openai_helper_core.py already exists which builds the hop 0 graph. Eventually all functions from openai_helper_core.py should be moved into this so we have a single agentic flow to build the graph.

"""

"""
Logic flow:
For a given product and capabilities set:
Hop 0 Setup
1. Get first order pains, jobs, personas
--> infer_with_rules_then_fallback(product_id, product_subgraph, force_openai=True)
2. Calculate and update capability centrality. 
--> calculate_capability_centrality(product_id, product_subgraph)
3. Based on capability centrality & capability coreness - identify which capabilities are core functionality and which are likely blockers.
--> set_capabilities_relevance(product_id, product_subgraph)
Traversal logic - identifying terminal nodes that require further exploration:
4. Calculate cumulative relevance for each node
--> update_cumulative_relevance(product_id, product_subgraph)

Hop++ Logic subflow
5. Starting from core capabilites - recurse up to pains-jobs-personas ONLY if the node is relevant.
--> get_all_source_nodes(node_id, product_id)
6. If node is relevant but no further target nodes exist - add its persona-job-pain triplet to gpt cache and run a hop+ gpt query.
--> get_upstream_triplets(persona, job)

Hop-- Logic subflow
7. Starting from core capabilites - recurse down to pains-jobs-personas ONLY if the node is relevant.
--> get_all_target_nodes(node_id, product_id)
8. If node is relevant but no further target nodes exist - add the persona-job-pain triplet to gpt cache and run a hop- gpt query.
--> get_downstream_triplets(persona, job, pain)

Blocker logic
-> Ideally blockers should be identified in the first pass of the graph discovery.

"""

"""
What we should have at the end of this:
- A complete graph of personas, jobs and pains from the product ID to the point where cumulative relevance becomes irrelevant.
- The graph includes:
    hop+ (personas impacted by upstream pains resulting from job failures because this product doesn't exist),
    hop-- (personas impacted by downstream jobs existing just to solve pains caused by this product not existing), and
    blockers (personas unimpacted directly by the product's core capabilities, but impacted by its side effects in legal, complicance, procurement, etc)
- At each pain the graph also includes the pain trigger, which is a real-world metric that must scale for the pain to become intolerable.
"""
"""
One more thing:
- We still will need to come back to this with a minor tweak - currently for a given pain/ persona we are only getting the jobs and pains relevant to the product.
But to be able to assess if our product capabilities actually solve the customers' expectations, we will need to describe "other pains the user typically faces during this job, and the other jobs they are responsible for typically...
This gives us an idea of how much of this user's problems are "met" by the product ==> likelihood of turning a champion/ detractor if they became product aware... 
 


"""