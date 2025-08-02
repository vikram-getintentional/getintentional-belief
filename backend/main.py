# backend/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.auth import models
from backend.database import engine
from backend.auth.auth_routes import router as auth_router
from backend.routes import scraper
from backend.routes import analyzer
from backend.routes.openai_routes import router as openai_router
from backend.routes.websocket_routes import router as websocket_router
from backend.init_db import init_db

print("Starting db init")
init_db()
print("DB Done")

# Everything autosaves in life
app = FastAPI()
models.Base.metadata.create_all(bind=engine)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(scraper.router)
app.include_router(analyzer.router)
app.include_router(openai_router, prefix="/openai")
app.include_router(websocket_router)
