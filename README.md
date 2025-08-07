# Readme

The project contains two components - frontend and backend of the
getintentional webapp

## Frontend

`frontend` folder contains a reactjs app rendering the web UI

To run the vite build server, run `npm run dev` after `cd frontend`

## Backend

`backend` folder contains a FastAPI driven python web server

To run the web server, make sure virtualenv is activated in the shell
with `source ./.venv/bin/activate` - `.venv` is the folder where
virtual environment is setup.

You can find `requirements.txt` in the project root itself.

## Asynchronous Hop0 Graph Generation Setup

This document explains how to set up and run the asynchronous Hop0 graph generation system using Celery and Redis.

### Prerequisites

1. **Redis Server**: Required as a message broker for Celery
2. **Updated Dependencies**: Install the new Python dependencies

### Installation

#### 1. Install Redis (if not already installed)

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install redis-server
sudo systemctl start redis-server
sudo systemctl enable redis-server
```

**macOS:**
```bash
brew install redis
brew services start redis
```

**Windows:**
Download and install Redis from: https://redis.io/download

#### 2. Install Python Dependencies

```bash
# Activate virtual environment
source ./.venv/bin/activate

# Install new dependencies
pip install -r backend/requirements.txt
```

#### 3. Environment Configuration

Add the following to your `.env` file (create if it doesn't exist):

```bash
# Redis configuration
REDIS_URL=redis://localhost:6379/0

# Existing OpenAI configuration
OPENAI_API_KEY=your_openai_api_key_here
```

### Running the System

You need to run **three** processes concurrently:

#### 1. Redis Server
Make sure Redis is running:
```bash
redis-server
```

#### 2. Celery Worker
In a new terminal, start the Celery worker:
```bash
# Navigate to project directory
cd /path/to/getintentional-belief

# Activate virtual environment
source ./.venv/bin/activate

# Start the worker
python worker.py

# Or alternatively:
celery -A backend.celery_app worker --loglevel=info --queues=graph_generation
```

#### 3. FastAPI Backend
In another terminal, start the FastAPI server:
```bash
# Navigate to project directory
cd /path/to/getintentional-belief

# Activate virtual environment
source ./.venv/bin/activate

# Start the backend
fastapi dev backend/main.py
```

#### 4. Frontend (if needed)
In a fourth terminal, start the React frontend:
```bash
cd frontend
npm run dev
```

### How It Works
#### Backend Changes

1. **Celery Integration**: Added Celery configuration in `backend/celery_app.py`
2. **Async Task**: Created `generate_hop0_graph` task in `backend/tasks/graph_generation.py`
3. **API Changes**: Modified `/analyze/deep` endpoint to return task ID instead of blocking
4. **Status Endpoint**: Added `/tasks/{task_id}/status` for polling task progress
5. **WebSocket Support**: Added real-time notifications via WebSocket at `/ws/{company_id}`

#### Frontend Changes

1. **WebSocket Service**: Added `frontend/src/services/websocketService.ts` for real-time updates
2. **Task Service**: Added `frontend/src/services/taskService.ts` for async task management
3. **UI Updates**: Enhanced PersonaBuilder with progress bars and real-time status updates

#### Flow

1. User triggers deep analysis
2. Frontend calls `/analyze/deep` → receives task ID immediately
3. Frontend establishes WebSocket connection for real-time updates
4. Frontend polls task status as fallback
5. Celery worker processes the task asynchronously
6. When complete, worker sends WebSocket notification
7. Frontend receives notification and updates UI with results

### Monitoring
#### Check Celery Worker Status
```bash
celery -A backend.celery_app inspect active
```

#### Check Redis Connection
```bash
redis-cli ping
# Should return: PONG
```

#### View Celery Logs
The worker will output logs showing task execution progress.

### Troubleshooting
#### Common Issues

1. **Redis Connection Error**
   - Ensure Redis server is running
   - Check REDIS_URL in .env file

2. **Celery Worker Not Starting**
   - Verify virtual environment is activated
   - Check that all dependencies are installed
   - Ensure Python path includes project root

3. **WebSocket Connection Failed**
   - Check that FastAPI server is running
   - Verify authentication token is valid
   - Ensure CORS settings allow WebSocket connections

4. **Task Stuck in PENDING**
   - Check if Celery worker is running
   - Verify worker is consuming from correct queue
   - Check Redis connection

#### Debug Commands

```bash
# Check active tasks
celery -A backend.celery_app inspect active

# Check registered tasks
celery -A backend.celery_app inspect registered

# Purge all tasks
celery -A backend.celery_app purge

# Monitor task events
celery -A backend.celery_app events
```

### Production Considerations

1. **Process Management**: Use process managers like `supervisord` or `systemd`
2. **Scaling**: Run multiple worker processes for higher throughput
3. **Monitoring**: Implement proper logging and monitoring solutions
4. **Error Handling**: Add retry mechanisms and dead letter queues
5. **Security**: Secure Redis and implement proper authentication
