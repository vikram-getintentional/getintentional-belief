import json
import traceback
import asyncio
from celery import current_task
from backend.celery_app import celery_app
from backend.utils.inference.discovery_engine.openai_helper_core import infer_with_rules_then_fallback
from backend.utils.graph_base.network_graph import build_product_graph
from backend.utils.inference.discovery_engine.hop_plus_agent import infer_upstream_with_rules



@celery_app.task(bind=True, name="backend.tasks.graph_generation.generate_hop_plus_graph")
def generate_hop_plus_graph(
        self,
        product_id,
        cap_threshold=0.3,
        relevance_threshold=0.6,
        max_depth=3
):
    """
    Celery task for generating Hop0 graph asynchronously.
    
    Args:
        product_id: The product ID to generate graph for
        company_id: The company ID for authentication context
        
    Returns:
        dict: The hop0 results or error information
    """
    try:
        # Update task state to indicate processing has started
        self.update_state(
            state="PROCESSING",
            meta={
                "message": "Starting Hop+ graph generation",
                "product_id": product_id,
                "progress": 10
            }
        )
        
        # Build product subgraph
        self.update_state(
            state="PROCESSING", 
            meta={
                "message": "Building product subgraph",
                "product_id": product_id,
                "progress": 30
            }
        )
        
        print("Starting Hop0 graph load")
        product_subgraph = build_product_graph(product_id)
        print(f"[CELERY] Extracted product subgraph with total nodes: {len(product_subgraph.nodes)}")
        
        # Run inference
        self.update_state(
            state="PROCESSING",
            meta={
                "message": "Running Hop+ inference with OpenAI",
                "product_id": product_id, 
                "progress": 60
            }
        )
        
        print("[CELERY] Starting hop+ inference")
        hop_plus_results = infer_upstream_with_rules(
            product_subgraph=product_subgraph,
            cap_threshold=0.3,
            relevance_threshold=0.6,
            max_depth=3
        )

        
        # Complete task
        self.update_state(
            state="PROCESSING",
            meta={
                "message": "Finalizing results",
                "product_id": product_id,
                "progress": 100
            }
        )
        
        print("results in analyze_hop_plus:", hop_plus_results)
        
        return {
            "status": "SUCCESS",
            "product_id": product_id,
            "hop_plus_results": hop_plus_results,
            "message": "Hop+ graph generation completed successfully"
        }
    
    except Exception as e:
        print(f"[CELERY] ❌ Error in Hop0 graph generation: {e}")
        # Update task state to indicate failure
        self.update_state(
            state="FAILURE",
            meta={
                "message": f"Error in Hop0 graph generation: {str(e)}",
                "product_id": product_id,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        )
        raise
            

@celery_app.task(bind=True, name="backend.tasks.graph_generation.generate_hop0_graph")
def generate_hop0_graph(self, product_id: str, company_id: str):
    """
    Celery task for generating Hop0 graph asynchronously.
    
    Args:
        product_id: The product ID to generate graph for
        company_id: The company ID for authentication context
        
    Returns:
        dict: The hop0 results or error information
    """
    try:
        # Update task state to indicate processing has started
        self.update_state(
            state="PROCESSING",
            meta={
                "message": "Starting Hop0 graph generation",
                "product_id": product_id,
                "progress": 10
            }
        )
        
        # Build product subgraph
        self.update_state(
            state="PROCESSING", 
            meta={
                "message": "Building product subgraph",
                "product_id": product_id,
                "progress": 30
            }
        )
        
        product_subgraph = build_product_graph(product_id)
        print(f"[CELERY] Extracted product subgraph with total nodes: {len(product_subgraph.nodes)}")
        
        # Run inference
        self.update_state(
            state="PROCESSING",
            meta={
                "message": "Running Hop0 inference with OpenAI",
                "product_id": product_id, 
                "progress": 60
            }
        )
        
        print("[CELERY] Starting hop0 inference")
        hop_0_results = infer_with_rules_then_fallback(product_id, product_subgraph)
        
        # Complete task
        self.update_state(
            state="PROCESSING",
            meta={
                "message": "Finalizing results",
                "product_id": product_id,
                "progress": 100
            }
        )
        
        print(f"[CELERY] Hop0 inference completed successfully for product {product_id}")
        
        return {
            "status": "SUCCESS",
            "product_id": product_id,
            "company_id": company_id,
            "hop_0_results": hop_0_results,
            "message": "Hop0 graph generation completed successfully"
        }
    
    except Exception as e:
        print(f"[CELERY] ❌ Error in Hop0 graph generation: {e}")
        # Update task state to indicate failure
        self.update_state(
            state="FAILURE",
            meta={
                "message": f"Error in Hop0 graph generation: {str(e)}",
                "product_id": product_id,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        )
        raise
            

@celery_app.task(name="backend.tasks.graph_generation.get_task_status")
def get_task_status(task_id: str):
    """
    Get status of a running task.
    
    Args:
        task_id: The Celery task ID
        
    Returns:
        dict: Task status information
    """
    task_result = celery_app.AsyncResult(task_id)
    
    if task_result.state == "PENDING":
        response = {
            "state": task_result.state,
            "message": "Task is waiting to be processed"
        }
    elif task_result.state == "PROCESSING":
        response = {
            "state": task_result.state,
            "message": task_result.info.get("message", "Processing..."),
            "progress": task_result.info.get("progress", 0),
            "product_id": task_result.info.get("product_id")
        }
    elif task_result.state == "SUCCESS":
        response = {
            "state": task_result.state,
            "message": "Task completed successfully",
            "result": task_result.result
        }
    elif task_result.state == "FAILURE":
        response = {
            "state": task_result.state,
            "message": task_result.info.get("message", "Task failed"),
            "error": task_result.info.get("error", str(task_result.info))
        }
    else:
        response = {
            "state": task_result.state,
            "message": "Unknown task state"
        }
    
    return response
