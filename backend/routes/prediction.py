from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, File, Request, UploadFile, HTTPException

router = APIRouter()

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/bmp", "image/webp", "image/gif"}
MAX_SIZE = 10 * 1024 * 1024


@router.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail="Invalid image type. Upload JPEG, PNG, BMP, WebP, or GIF.")

    image_bytes = await file.read()
    if len(image_bytes) > MAX_SIZE:
        raise HTTPException(status_code=400, detail="Image too large. Maximum 10 MB.")
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    predictor = request.app.state.predictor
    try:
        result = predictor.predict_with_gradcam(image_bytes)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(exc)}")

    result["timestamp"] = datetime.now(timezone.utc).isoformat()
    result["image_name"] = file.filename or "unknown"

    db = request.app.state.db
    await db.save_prediction(result)

    return result
