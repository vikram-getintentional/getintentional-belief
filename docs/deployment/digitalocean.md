# Deploying GetIntentional on DigitalOcean

This guide assumes you want a reproducible container-based deployment with a managed Postgres database and a durable volume for the journey storage files. We build two images (FastAPI backend + Vite frontend) and orchestrate them with `docker compose`.

## Supported environment variables

| Variable | Purpose | Suggested value |
| --- | --- | --- |
| `DATABASE_URL` | SQLAlchemy connection string (Postgres URI recommended for production). | `postgresql+psycopg2://user:pass@db-host:5432/getintent` |
| `OPENAI_API_KEY` | Your OpenAI API key for GPT-powered features. | `sk-...` |
| `BELIEF_STORAGE_DIR` | Directory where the belief manager reads/writes JSON stats/weights. Mount a persistent block storage path here. | `/data/belief` |
| `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB` | Used by the local Postgres service when running `docker compose`. Override when connecting to a managed database. | align with your Postgres instance |
| `VITE_API_BASE` | API base URL baked into the frontend bundle (e.g., `https://api.getintentional.com`). | `http://backend:8000` for local compose deployments |

Copy `.env.example` → `.env` and update these values (`.env` is already in `.gitignore`, so your secrets stay off Git).

## Local container workflow

1. Build/run the stack:
   ```bash
   docker compose up --build
   ```
2. Containers:
   - `frontend` (port `4173` exposed) — serves the Vite build via nginx.
   - `backend` (port `8000`) — runs `uvicorn` and reads the shared `BELIEF_STORAGE_DIR`.
   - `db` — Postgres 15 that persists to the `db_data` named volume.
3. Override `VITE_API_BASE` if you want the frontend to hit a different address (production domain, reverse proxy, etc.).

## DigitalOcean deployment steps

1. **Provision infrastructure**
   - Create a Droplet (Ubuntu 24.04 or similar) with Docker (`docker` & `docker compose` pre-installed or install manually).
   - Attach a Block Storage volume (e.g., 50 GB) and mount it to `/mnt/belief`.
   - Open ports `80`, `443`, and `8000` via the Droplet firewall if you plan to expose the API directly.

2. **Managed Postgres (recommended)**
   - Spin up a DigitalOcean Managed Database for Postgres.
   - Copy the connection URI and plug it into `DATABASE_URL`/`POSTGRES_*` in `.env`.
   - Allow your Droplet IP in the managed database firewall rules.

3. **Deploy code**
   ```bash
   git clone https://github.com/<your-repo>/getintentional.git
   cd getintentional
   cp .env.example .env
   # edit .env with production values:
   # - set DATABASE_URL to the managed Postgres URI
   # - set BELIEF_STORAGE_DIR to the mounted block storage path (e.g., /mnt/belief)
   # - set OPENAI_API_KEY
   # - optionally set VITE_API_BASE to your frontend domain
   ```

4. **Start containers**
   ```bash
   docker compose up --build -d
   ```
   - The `belief_data` volume is mounted to `/data/belief`; ensure the `/mnt/belief` mountpoint is symlinked or bind-mounted there if you need to attach the block volume.
   - For production, consider replacing the local Postgres service with your managed URL and removing the `db` service from `docker-compose.yml`, leaving only `backend` + `frontend`.

5. **Networking**
   - Point your DNS name to the Droplet IP and use a reverse proxy (Traefik/Caddy/Nginx) or DigitalOcean Load Balancer to forward `/` traffic to port `4173` (frontend) and `/api` traffic to port `8000`.
   - Update `VITE_API_BASE` to the backend URL that your domain will use (e.g., `https://api.getintentional.com`).

6. **Maintenance**
   - Use `docker compose logs -f backend` to monitor the API.
   - Keep the OpenAI key and database credentials rotated via `.env`.
   - Backup `/mnt/belief` (the block storage volume) and your Postgres dumps regularly.
