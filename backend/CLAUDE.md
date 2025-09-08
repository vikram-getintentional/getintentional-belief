This document provides the necessary context for an AI assistant to effectively contribute to the "GetIntentional" project. The primary goal is to enable the AI to help with adding new features, with a focus on generating correct and runnable backend code.

## 1. Project Overview

**GetIntentional** is a B2B marketing strategy engine. It moves beyond traditional marketing approaches by using a data-driven, causal model to generate prescriptive and dynamic marketing execution plans.

The core philosophy is **"Intentional Marketing,"** which is based on a deep understanding of customer psychology, internal organizational pressures, and the causal relationships between a product's capabilities and a customer's pains.

The application is composed of two main parts:
- A **ReactJS frontend** for the user interface.
- A **FastAPI backend** that handles the core logic, data analysis, and graph-based modeling.

## 2. Tech Stack

- **Backend**:
    - **Framework**: FastAPI
    - **Language**: Python
    - **Database**: SQLite (via SQLAlchemy)
    - **Core Logic**: `networkx` for graph manipulation, `openai` for AI-powered inferences, `sentence-transformers` for embeddings.
    - **Dependencies**: See `backend/requirements.txt`.
- **Frontend**:
    - **Framework**: ReactJS (with Vite)
    - **Language**: JavaScript/TypeScript

## 3. Architecture

### 3.1. Backend

The backend is the heart of the application. It's responsible for:
- **Authentication**: JWT-based authentication (`backend/auth`).
- **API Routes**: Endpoints for scraping, analysis, and data retrieval (`backend/routes`).
- **Graph-Based Data Model**: The core of the application's logic resides in a graph-based representation of marketing concepts.

### 3.2. The Causal Graph

The application's central data structure is a directed graph built with `networkx`. This graph models the causal relationships between different marketing entities:

**Capabilities → Pains → Jobs → Personas**

- **Capabilities**: The features or functionalities of a product.
- **Pains**: The problems or friction points that a capability solves.
- **Jobs**: The tasks or responsibilities that are affected by these pains.
- **Personas**: The individuals or roles within an organization who perform these jobs.

This graph is extended with concepts like:
- **Hop+ / Hop-**: Upstream and downstream dependencies between jobs and personas.
- **ZMOT (Zero Moment of Truth)**: The trigger events that initiate a customer's search for a solution.
- **ICP (Ideal Customer Profile)**: The characteristics of the companies most likely to be a good fit.

### 3.3. The Inference Engine

The `backend/utils/inference/discovery_engine` directory contains the most complex logic in the application. It is responsible for:
- **Hop 0 Graph Generation**: The initial creation of the persona-job-pain graph from a product's capabilities. This is a computationally intensive process that is a prime candidate for asynchronous processing.
- **Hop+ Traversal**: Identifying upstream dependencies and pressure points.
- **ZMOT/ICP Discovery**: Inferring the trigger events and ideal customer profiles from the graph.

## 4. Development Workflow

- **Running the Backend**:
    ```bash
    # Activate the virtual environment
    source ./.venv/bin/activate

    # Run the FastAPI development server
    fastapi dev backend/main.py
    ```
- **Running the Frontend**:
    ```bash
    cd frontend
    npm run dev
    ```

## 5. Key Priorities & Coding Style

1.  **Correctness and Runnability**: All generated code, especially for the backend, must be correct and runnable. It should adhere to the existing patterns and conventions in the codebase.
2.  **Asynchronous Operations**: For long-running tasks like graph generation, the preferred approach is to offload them to a background worker (e.g., Celery).
3.  **Real-time UI Updates**: When a background task completes, the frontend should be updated in real-time. This will likely involve a mechanism like WebSockets or server-sent events (SSE) to notify the client.
4.  **Modularity**: New features should be implemented in a modular way, following the existing project structure.

## 6. Core Concepts

- **Reverse Case Study (RCS)**: A narrative that works backward from a successful customer outcome to the initial trigger event (ZMOT). This is a key conceptual framework for generating marketing strategies.
- **ZMOT (Zero Moment of Truth)**: The critical moment when a customer realizes they have a pain and begins to search for a solution. The system identifies ZMOTs based on internal pressures and external triggers.
- **Intentional Marketing**: The overarching philosophy of the application. It emphasizes a proactive, data-driven approach to marketing that is based on a deep understanding of the customer's world.

