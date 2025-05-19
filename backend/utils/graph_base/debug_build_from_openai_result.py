from backend.utils.graph_base.nodes.persona_nodes import get_or_create_persona_node_from_fields
from backend.utils.graph_base.nodes.job_nodes import get_or_create_job_node
from backend.utils.graph_base.nodes.pain_nodes import get_or_create_pain_node
from backend.utils.graph_base.edges.edge_manager import add_edge


def save_graph_from_openai_result(openai_result_graph):
    for item in openai_result_graph:
        # Step 1: Primary persona
        persona = item["persona"]
        job = item["job"]
        pain = item["pain"]

        persona_node = get_or_create_persona_node_from_fields(
            persona["title"], persona["seniority"], persona["department"]
        )
        job_node = get_or_create_job_node(job)
        pain_node = get_or_create_pain_node(pain)

        add_edge(persona_node["id"], job_node["id"], "has_job")
        add_edge(job_node["id"], pain_node["id"], "has_pain")

        # Step 2: Dependencies (downstream jobs affected)
        for dep in item.get("dependencies", []):
            dep_p = dep["dependent_persona"]
            dep_job = dep["dependent_job"]
            dep_pain = dep["dependent_pain"]

            dep_persona_node = get_or_create_persona_node_from_fields(
                dep_p["title"], dep_p["seniority"], dep_p["department"]
            )
            dep_job_node = get_or_create_job_node(dep_job)
            dep_pain_node = get_or_create_pain_node(dep_pain)

            add_edge(dep_persona_node["id"], dep_job_node["id"], "has_job")
            add_edge(dep_job_node["id"], dep_pain_node["id"], "has_pain")

            # Causal: Primary job → dependent job
            add_edge(job_node["id"], dep_job_node["id"], "impacts")

