"""
ethicura_ai/src/api.py
FastAPI REST API for Ethicura AI deepfake detection.

Endpoints:
    POST /predict       — Analyse an uploaded image or video file
    GET  /health        — Health check
    GET  /model-info    — Model architecture and performance summary

Usage:
    uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
"""

import io
import cv2
import numpy as np
import tempfile
import os
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from src.detector import EthicuraDetector

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Ethicura AI — DeepFake Detection API",
    description=(
        "Unified DeepFake Detection Framework (UDDF). "
        "Detects manipulated faces in images and videos using XceptionNet + MTCNN, "
        "with Grad-CAM visualisations for interpretability."
    ),
    version="1.0.0",
)

# Singleton detector (loaded once, reused across requests)
detector = EthicuraDetector()

# Supported file types
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


# ── Response models ────────────────────────────────────────────────────────────

class PredictionResponse(BaseModel):
    filename: str
    file_type: str           # "image" or "video"
    label: str               # "FAKE", "REAL", or "UNKNOWN"
    confidence: float | None # Fake probability 0–100%
    face_detected: bool
    message: str | None
    # For video:
    fake_frame_count: int | None = None
    total_frames_sampled: int | None = None
    fake_ratio: float | None = None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "Ethicura AI"}


@app.get("/model-info")
def model_info():
    return {
        "classifier": "XceptionNet (fine-tuned on FaceForensics++)",
        "face_detector": "MTCNN (Multi-Task Cascaded CNN)",
        "interpretability": "Grad-CAM (block14_sepconv2_act layer)",
        "input_size": "299x299",
        "performance": {
            "accuracy": "95%",
            "f1_score": "92.5%",
            "auc_roc": 0.97,
            "benchmark": "FaceForensics++"
        },
        "supported_inputs": {
            "images": list(IMAGE_EXTENSIONS),
            "videos": list(VIDEO_EXTENSIONS)
        }
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Upload an image or video file for deepfake analysis.

    Returns a JSON prediction with:
    - label: FAKE / REAL / UNKNOWN
    - confidence: fake probability (%)
    - face_detected: whether a face was found
    - For videos: per-frame stats and overall verdict
    """
    suffix = Path(file.filename).suffix.lower()

    if suffix not in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. "
                   f"Supported: {IMAGE_EXTENSIONS | VIDEO_EXTENSIONS}"
        )

    # Save upload to temp file
    contents = await file.read()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        if suffix in IMAGE_EXTENSIONS:
            result = detector.predict(tmp_path)
            return PredictionResponse(
                filename=file.filename,
                file_type="image",
                label=result["label"],
                confidence=result.get("confidence"),
                face_detected=result["face_detected"],
                message=result.get("message"),
            )

        else:  # video
            result = detector.predict_video(tmp_path)
            return PredictionResponse(
                filename=file.filename,
                file_type="video",
                label=result.get("verdict", "UNKNOWN"),
                confidence=result.get("avg_confidence"),
                face_detected=result.get("total_frames_sampled", 0) > 0,
                message=result.get("message"),
                fake_frame_count=result.get("fake_frame_count"),
                total_frames_sampled=result.get("total_frames_sampled"),
                fake_ratio=result.get("fake_ratio"),
            )

    finally:
        os.unlink(tmp_path)  # Clean up temp file


@app.post("/predict/gradcam")
async def predict_with_gradcam(file: UploadFile = File(...)):
    """
    Same as /predict but also returns the Grad-CAM annotated image as PNG bytes.
    For images only.
    """
    suffix = Path(file.filename).suffix.lower()
    if suffix not in IMAGE_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Grad-CAM endpoint supports images only.")

    contents = await file.read()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        result = detector.predict(tmp_path)
        overlay = result.get("gradcam_overlay")

        if overlay is None:
            raise HTTPException(status_code=422, detail="No face detected — cannot generate Grad-CAM.")

        _, img_encoded = cv2.imencode(".png", overlay)
        return Response(
            content=img_encoded.tobytes(),
            media_type="image/png",
            headers={
                "X-Ethicura-Label": result["label"],
                "X-Ethicura-Confidence": str(result.get("confidence", "")),
            }
        )
    finally:
        os.unlink(tmp_path)
