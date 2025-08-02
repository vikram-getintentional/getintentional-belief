import os
from celery import Celery
from dotenv import load_dotenv
import multiprocessing

# Add this line at the top of the file
# Force the 'spawn' start method for multiprocessing.
# This MUST be placed at the top level of the script, before the Celery app is instantiated.
# It ensures that any child processes (like Celery workers) are created with a clean state,
# which is required for libraries that use CUDA (like sentence-transformers).
multiprocessing.set_start_method("spawn", force=True)

load_dotenv()

# Redis configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Create Celery instance
celery_app = Celery(
    "getintentional",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["backend.tasks.graph_generation"]
)

celery_app.conf.broker_connection_retry_on_startup = True

# Configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    result_expires=3600,  # Results expire after 1 hour
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    worker_disable_rate_limits=True,
)

# Task routing
celery_app.conf.task_routes = {
    "backend.tasks.graph_generation.*": {"queue": "graph_generation"}
}
