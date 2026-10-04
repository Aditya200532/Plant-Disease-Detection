from __future__ import annotations

import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.services.predictor import Predictor
from backend.services.database import Database
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.routes.prediction import router as prediction_router
from backend.routes.history import router as history_router
from backend.routes.model import router as model_router

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = PROJECT_ROOT / "artifacts" / "plant_disease_model.keras"
    metadata_path = PROJECT_ROOT / "artifacts" / "metadata.json"
    metrics_path = PROJECT_ROOT / "results" / "metrics.json"

    if not model_path.exists():
        print(f"Model not found at {model_path}. Train the model first.", file=sys.stderr)
        sys.exit(1)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}

    predictor = Predictor(str(model_path), metadata)
    app.state.predictor = predictor
    app.state.metadata = metadata
    app.state.metrics = metrics

    db = Database()
    await db.connect()
    app.state.db = db

    yield

    await db.close()


app = FastAPI(
    title="Plant Disease Detection API",
    description="AI-powered plant leaf disease classification using MobileNetV2",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(prediction_router, prefix="/api")
app.include_router(history_router, prefix="/api")
app.include_router(model_router, prefix="/api")


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.get("/api/results/{filename}")
async def get_result_file(filename: str):
    safe_name = Path(filename).name
    file_path = RESULTS_DIR / safe_name
    if not file_path.exists() or not file_path.is_file():
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)
