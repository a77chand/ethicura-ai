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
import base64
import cv2
import numpy as np
import tempfile
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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

# Allow the React frontend (running on a different origin — localhost during
# dev, a vercel.app domain once deployed) to call this API from the browser.
# Wide open ("*") is fine for a hackathon demo API with no auth/user data;
# tighten this to your actual frontend domain before any real production use.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
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
    confidence: Optional[float]  # Fake probability 0-100%
    face_detected: bool
    message: Optional[str]
    # For video:
    fake_frame_count: Optional[int] = None
    total_frames_sampled: Optional[int] = None
    fake_ratio: Optional[float] = None


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


class FramePredictionResponse(BaseModel):
    label: str                # "FAKE", "REAL", or "UNKNOWN"
    confidence: Optional[float]  # Fake probability 0-100%
    face_detected: bool
    overlay_png_base64: Optional[str]  # base64-encoded PNG, or None if no face
    message: Optional[str] = None


@app.post("/predict_frame", response_model=FramePredictionResponse)
async def predict_frame(payload: dict):
    """
    Live-webcam-friendly variant of /predict/gradcam.

    Accepts a base64-encoded JPEG frame (as JSON, not multipart file upload —
    simpler for a browser sending frequent frames from a <canvas>) and returns
    the label, confidence, and Grad-CAM overlay as base64 PNG in one JSON
    response, so the frontend doesn't need to juggle separate header/body
    parsing like /predict/gradcam does.

    Request body: {"image_base64": "<base64 jpeg data, no data: prefix>"}
    """
    image_b64 = payload.get("image_base64")
    if not image_b64:
        raise HTTPException(status_code=400, detail="Missing 'image_base64' in request body.")

    # Strip a data URL prefix if the frontend sent one (e.g. "data:image/jpeg;base64,...")
    if "," in image_b64[:50]:
        image_b64 = image_b64.split(",", 1)[1]

    try:
        image_bytes = base64.b64decode(image_b64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image data.")

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name

    try:
        result = detector.predict(tmp_path)

        overlay_b64 = None
        if result["face_detected"] and result.get("gradcam_overlay") is not None:
            _, img_encoded = cv2.imencode(".png", result["gradcam_overlay"])
            overlay_b64 = base64.b64encode(img_encoded.tobytes()).decode("utf-8")

        return FramePredictionResponse(
            label=result["label"],
            confidence=result.get("confidence"),
            face_detected=result["face_detected"],
            overlay_png_base64=overlay_b64,
            message=result.get("message"),
        )
    finally:
        os.unlink(tmp_path)
