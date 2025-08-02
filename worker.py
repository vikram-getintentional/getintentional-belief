#!/usr/bin/env python3
"""
Celery worker script for GetIntentional backend.
Run this script to start the Celery worker process.

Usage:
    python worker.py

Or with specific options:
    celery -A backend.celery_app worker --loglevel=info --queues=graph_generation
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import and configure the Celery app
from backend.celery_app import celery_app

if __name__ == "__main__":
    # Run the worker
    celery_app.start([
        "worker",
        "--loglevel=info",
        "--queues=graph_generation",
        "--concurrency=2",
        "--without-heartbeat",
        "--without-mingle",
        "--without-gossip"
    ])