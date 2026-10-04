from __future__ import annotations

from pydantic import BaseModel
from typing import Optional


class PredictionResponse(BaseModel):
    plant: str
    disease: str
    status: str
    confidence: float
    class_name: str
    warning: Optional[str] = None
    gradcam: Optional[str] = None
    timestamp: Optional[str] = None


class ModelInfoResponse(BaseModel):
    model_name: str
    number_of_classes: int
    image_size: int
    class_names: list[str]
    test_accuracy: Optional[float] = None
    precision_weighted: Optional[float] = None
    recall_weighted: Optional[float] = None
    f1_weighted: Optional[float] = None


class StatsResponse(BaseModel):
    total: int
    healthy: int
    diseased: int
    top_diseases: list[dict]
    db_available: bool
