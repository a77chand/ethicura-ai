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

import os

# Disable XLA JIT compilation BEFORE tensorflow is imported anywhere.
# On a weak/shared CPU (e.g. Render's free tier, 0.1 CPU), XLA's first-call
# JIT compile step has been observed taking 8-11 MINUTES (confirmed in
# production logs: "slow_operation_alarm.cc:140] The operation took
# 10m50s"), because XLA is optimizing for repeat-call speed on real hardware,
# not for a CPU this constrained. Disabling it trades a small per-call
# overhead for avoiding that multi-minute one-time compile stall entirely —
# essential for this to be usable on constrained/free hosting tiers.
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"
os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=0"

import io
import base64
import threading
import time
import cv2
import numpy as np
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from src.detector import EthicuraDetector

# ── App setup ─────────────────────────────────────────────────────────────────

# Singleton detector (loaded once, reused across requests)
detector = EthicuraDetector()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load + warm the models in the background so the server binds its port
    # immediately (hosts like Render time out on slow port binding) and the
    # frontend can poll /health for model_ready instead of its first real
    # frame taking a minute.
    threading.Thread(target=detector.warm_up, name="model-warmup", daemon=True).start()
    yield


app = FastAPI(
    lifespan=lifespan,
    title="Ethicura AI — DeepFake Detection API",
    description=(
        "Unified DeepFake Detection Framework (UDDF). "
        "Detects manipulated faces in images and videos using an in-house EfficientNetB0 + MTCNN, "
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
    return {
        "status": "ok",
        "service": "Ethicura AI",
        "model_ready": detector.is_ready,
        "model_error": detector.load_error,
    }


@app.get("/model-info")
def model_info():
    return {
        "classifier": "EfficientNetB0, trained in-house on DeepFakeFace (see README Results)",
        "face_detector": "MTCNN (Multi-Task Cascaded CNN)",
        "interpretability": "Grad-CAM on the EfficientNetB0 feature map",
        "input_size": "224x224",
        "performance": {
            "test_set": "DeepFakeFace, people unseen in training",
            "auc_roc": 0.986,
            "equal_error_rate": 0.056,
            "tpr_at_1pct_fpr": 0.827,
            "balanced_accuracy": 0.946,
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

    # Inference is CPU-bound and blocking — run it in the thread pool so it
    # doesn't stall the event loop (and every other request, /health included).
    try:
        if suffix in IMAGE_EXTENSIONS:
            result = await run_in_threadpool(detector.predict, tmp_path)
            return PredictionResponse(
                filename=file.filename,
                file_type="image",
                label=result["label"],
                confidence=result.get("confidence"),
                face_detected=result["face_detected"],
                message=result.get("message"),
            )

        else:  # video
            result = await run_in_threadpool(detector.predict_video, tmp_path)
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
        result = await run_in_threadpool(detector.predict, tmp_path)
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
    # Face box in the submitted frame's pixel coordinates: {x, y, w, h}
    face_bbox: Optional[dict] = None
    frame_width: Optional[int] = None
    frame_height: Optional[int] = None
    # Raw Grad-CAM grid (e.g. 7x7 for EfficientNetB0), values 0-1, row-major.
    # A few hundred bytes — the browser colours it and draws it over the
    # live video at face_bbox, so the video itself never has to round-trip.
    heatmap: Optional[list[list[float]]] = None
    # Full annotated frame as base64 PNG — only when include_overlay=true.
    overlay_png_base64: Optional[str] = None
    inference_ms: Optional[float] = None
    message: Optional[str] = None


@app.post("/predict_frame", response_model=FramePredictionResponse)
def predict_frame(payload: dict):
    """
    Live-webcam-friendly variant of /predict/gradcam.

    Accepts a base64-encoded JPEG frame (as JSON, not multipart file upload —
    simpler for a browser sending frequent frames from a <canvas>) and returns
    the label, confidence, face box and raw Grad-CAM grid in one small JSON
    response.

    Declared as a plain `def` so FastAPI runs it in the thread pool — the
    model call blocks for the whole inference.

    Request body: {"image_base64": "<base64 jpeg data>", "include_overlay": false}
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

    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image data.")

    include_overlay = bool(payload.get("include_overlay", False))

    started = time.perf_counter()
    result = detector.predict_array(image, with_overlay=include_overlay)
    inference_ms = (time.perf_counter() - started) * 1000

    overlay_b64 = None
    if include_overlay and result["face_detected"] and result.get("gradcam_overlay") is not None:
        _, img_encoded = cv2.imencode(".png", result["gradcam_overlay"])
        overlay_b64 = base64.b64encode(img_encoded.tobytes()).decode("utf-8")

    heatmap = None
    if result.get("heatmap") is not None:
        heatmap = np.round(result["heatmap"].astype(float), 3).tolist()

    bbox = result.get("face_bbox")
    return FramePredictionResponse(
        label=result["label"],
        confidence=result.get("confidence"),
        face_detected=result["face_detected"],
        face_bbox={k: int(v) for k, v in bbox.items()} if bbox else None,
        frame_width=image.shape[1],
        frame_height=image.shape[0],
        heatmap=heatmap,
        overlay_png_base64=overlay_b64,
        inference_ms=round(inference_ms, 1),
        message=result.get("message"),
    )
