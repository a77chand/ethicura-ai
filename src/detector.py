"""
ethicura_ai/src/detector.py
Core detection pipeline: MTCNN face extraction → XceptionNet classification → Grad-CAM overlay.
"""

import threading

import cv2
import numpy as np
from pathlib import Path


class EthicuraDetector:
    """
    Unified DeepFake Detection Framework (UDDF).

    Detects manipulated faces in images and videos using:
    - MTCNN for face region extraction
    - XceptionNet for fake/real classification
    - Grad-CAM for visual interpretability

    Example:
        detector = EthicuraDetector()
        result = detector.predict("elon_musk_clip.jpg")
        print(result['label'], result['confidence'])
    """

    def __init__(self, model_path: str = None, threshold: float = 0.5):
        """
        Args:
            model_path: Path to fine-tuned XceptionNet weights (.h5).
                        If None, uses pretrained ImageNet weights (lower accuracy).
            threshold:  Decision boundary. Confidence > threshold → FAKE.
        """
        self.threshold = threshold
        self.model_path = model_path
        self._model = None
        self._face_detector = None
        self._gradcam = None
        # Serialises model loading and inference: the API serves requests
        # from a thread pool, and on a small CPU running two frames at once
        # only makes both slower.
        self._lock = threading.RLock()
        self.is_ready = False
        self.load_error = None

    # ──────────────────────────────────────────────────────────────────────────
    # Lazy loading — don't import heavy libs until actually needed
    # ──────────────────────────────────────────────────────────────────────────

    # Input size expected by the classifier. EfficientNetB0 backbone uses 224x224
    # (Xception in the original design used 299x299 — kept here as a class attr
    # so the rest of the pipeline can stay agnostic to which backbone is loaded).
    INPUT_SIZE = 224

    def _load_models(self):
        """Initialise MTCNN, classifier, and Grad-CAM on first use.

        Classifier: pretrained EfficientNetB0-based deepfake detector
        (ariyul/deepfake-detection-cnn_v2 on Hugging Face). This replaces the
        original from-scratch XceptionNet head, which had no trained weights
        checked into the repo (weights are gitignored and were never
        committed). Swapping in a real trained model here so the pipeline
        produces meaningful predictions rather than an untrained sigmoid head
        on frozen ImageNet features.
        """
        with self._lock:
            self._load_models_locked()

    def _load_models_locked(self):
        if self._model is not None:
            return  # already loaded

        try:
            from mtcnn import MTCNN
            import os
            os.environ.setdefault("KERAS_BACKEND", "tensorflow")
            import keras
            from huggingface_hub import snapshot_download

            # ── Face detector ──────────────────────────────────────────────
            self._face_detector = MTCNN()

            # ── Classifier ──────────────────────────────────────────────────
            if self.model_path and Path(self.model_path).exists():
                self._model = keras.saving.load_model(self.model_path)
                print(f"[Ethicura] Loaded fine-tuned weights from {self.model_path}")
            else:
                print("[Ethicura] No local model_path given — downloading pretrained "
                      "EfficientNetB0 deepfake classifier from Hugging Face "
                      "(ariyul/deepfake-detection-cnn_v2)...")
                local_dir = snapshot_download(repo_id="ariyul/deepfake-detection-cnn_v2")
                weights_path = os.path.join(local_dir, "keras_model", "best_model.keras")
                self._model = keras.saving.load_model(weights_path)
                print("[Ethicura] Loaded pretrained EfficientNetB0 deepfake classifier.")

            # ── Grad-CAM ───────────────────────────────────────────────────
            # EfficientNetB0 is loaded here as a nested Functional submodel
            # (named "efficientnetb0"), not flat conv layers at the top level,
            # so Grad-CAM targets that submodel's own last layer rather than a
            # named layer string like the original Xception setup used.
            from src.gradcam import GradCAM
            self._gradcam = GradCAM(self._model, layer_name="efficientnetb0")

        except ImportError as e:
            raise ImportError(
                f"Missing dependency: {e}\n"
                "Install with: pip install mtcnn tensorflow keras huggingface_hub opencv-python"
            )

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def warm_up(self):
        """Load models and run one dummy inference so the first real request
        doesn't pay for model download, graph tracing and MTCNN init."""
        try:
            self._load_models()
            with self._lock:
                dummy = np.zeros((1, self.INPUT_SIZE, self.INPUT_SIZE, 3), np.float32)
                self._gradcam.compute_with_prediction(dummy)
                self._face_detector.detect_faces(np.zeros((240, 320, 3), np.uint8))
            self.is_ready = True
            print("[Ethicura] Models warmed up and ready.")
        except Exception as e:
            self.load_error = str(e)
            print(f"[Ethicura] Warm-up failed: {e}")

    def predict(self, image_path: str) -> dict:
        """
        Run deepfake detection on a single image.

        Args:
            image_path: Path to image file (JPG, PNG, etc.)

        Returns:
            dict with keys:
                label           : "FAKE" or "REAL"
                confidence      : float 0–100 (fake probability %)
                face_detected   : bool
                gradcam_overlay : np.ndarray (BGR image with saliency heatmap)
                face_bbox       : dict {x, y, w, h} or None
                heatmap         : np.ndarray raw Grad-CAM grid (values 0-1) or None
        """
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")
        return self.predict_array(image)

    def predict_array(self, image_bgr: np.ndarray, with_overlay: bool = True) -> dict:
        """
        Same as predict(), for an already-decoded BGR image (e.g. a webcam
        frame) — skips the temp-file round trip.

        Args:
            with_overlay: set False to skip rendering gradcam_overlay when the
                          caller only needs the raw heatmap + bbox.
        """
        self._load_models()

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        with self._lock:
            face_crop, bbox = self._extract_face(image_rgb)

            if face_crop is None:
                return {
                    "label": "UNKNOWN",
                    "confidence": None,
                    "face_detected": False,
                    "gradcam_overlay": image_bgr,
                    "face_bbox": None,
                    "heatmap": None,
                    "message": "No face detected in the image."
                }

            face_input = self._preprocess(face_crop)

            # Classify + Grad-CAM in a single forward/backward pass
            cam, fake_prob = self._gradcam.compute_with_prediction(face_input)

        confidence = fake_prob * 100
        label = "FAKE" if fake_prob >= self.threshold else "REAL"

        overlay_bgr = None
        if with_overlay:
            overlay = self._overlay_cam(image_rgb.copy(), cam, bbox)
            overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)

        return {
            "label": label,
            "confidence": round(confidence, 2),
            "face_detected": True,
            "gradcam_overlay": overlay_bgr,
            "face_bbox": bbox,
            "heatmap": cam,
        }

    def predict_video(self, video_path: str, sample_every: int = 10) -> dict:
        """
        Run deepfake detection across video frames.

        Args:
            video_path   : Path to video file (.mp4, .avi, etc.)
            sample_every : Analyse every Nth frame (default: every 10th frame)

        Returns:
            dict with keys:
                verdict          : "FAKE" or "REAL"
                fake_frame_count : int
                total_frames     : int
                avg_confidence   : float
                frame_results    : list of per-frame dicts
        """
        self._load_models()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {video_path}")

        frame_results = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_every == 0:
                # Save temp frame and run image predict
                tmp_path = f"/tmp/ethicura_frame_{frame_idx}.jpg"
                cv2.imwrite(tmp_path, frame)
                result = self.predict(tmp_path)
                result["frame_index"] = frame_idx
                frame_results.append(result)

            frame_idx += 1

        cap.release()

        # Aggregate
        detected = [r for r in frame_results if r["face_detected"]]
        if not detected:
            return {"verdict": "UNKNOWN", "message": "No faces detected in any sampled frame."}

        fake_frames = [r for r in detected if r["label"] == "FAKE"]
        avg_conf = np.mean([r["confidence"] for r in detected])
        fake_ratio = len(fake_frames) / len(detected)

        return {
            "verdict": "FAKE" if fake_ratio >= 0.5 else "REAL",
            "fake_frame_count": len(fake_frames),
            "total_frames_sampled": len(detected),
            "total_frames": frame_idx,
            "avg_confidence": round(float(avg_conf), 2),
            "fake_ratio": round(fake_ratio, 3),
            "frame_results": frame_results,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    # MTCNN's cost grows with image area, so detect on a copy no wider than
    # this, then crop the face from the full-resolution image so the
    # classifier still gets full detail.
    DETECT_MAX_WIDTH = 320

    def _extract_face(self, image_rgb: np.ndarray):
        """Detect and crop the primary face using MTCNN."""
        scale = min(1.0, self.DETECT_MAX_WIDTH / image_rgb.shape[1])
        detect_input = image_rgb
        if scale < 1.0:
            detect_input = cv2.resize(image_rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        try:
            detections = self._face_detector.detect_faces(detect_input)
        except ValueError:
            # mtcnn 1.0 passes an empty batch to its last stage when the
            # earlier stages reject every candidate; newer Keras raises on
            # that instead of returning nothing. It means "no face".
            detections = []
        if not detections:
            return None, None

        # Take the highest-confidence detection
        best = max(detections, key=lambda d: d["confidence"])
        x, y, w, h = (int(round(v / scale)) for v in best["box"])

        # Add 20% margin around the face
        margin = int(0.2 * max(w, h))
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(image_rgb.shape[1], x + w + margin)
        y2 = min(image_rgb.shape[0], y + h + margin)

        face_crop = image_rgb[y1:y2, x1:x2]
        bbox = {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1}
        return face_crop, bbox

    def _preprocess(self, face_rgb: np.ndarray) -> np.ndarray:
        """Resize and normalise face crop for the classifier's expected input.

        The pretrained EfficientNetB0 model was trained with simple 0-1
        rescaling (img / 255.0), per its model card — not Xception's
        preprocess_input, which uses a different normalisation scheme.
        Using the wrong normalisation silently produces garbage predictions,
        so this matters even though the code still "runs" either way.
        """
        face = cv2.resize(face_rgb, (self.INPUT_SIZE, self.INPUT_SIZE))
        face = face.astype("float32") / 255.0
        return np.expand_dims(face, axis=0)  # shape: (1, 224, 224, 3)

    @staticmethod
    def _overlay_cam(image_rgb: np.ndarray, cam: np.ndarray, bbox: dict) -> np.ndarray:
        """Overlay Grad-CAM heatmap onto the face region of the original image."""
        x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]

        # Resize CAM to face bounding box size
        heatmap = cv2.resize(cam, (w, h))
        heatmap = np.uint8(255 * heatmap)
        heatmap_colour = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap_colour = cv2.cvtColor(heatmap_colour, cv2.COLOR_BGR2RGB)

        # Blend heatmap onto face region
        face_region = image_rgb[y:y+h, x:x+w]
        blended = cv2.addWeighted(face_region, 0.6, heatmap_colour, 0.4, 0)
        image_rgb[y:y+h, x:x+w] = blended

        # Draw bounding box
        colour = (255, 60, 60)  # red
        cv2.rectangle(image_rgb, (x, y), (x+w, y+h), colour, 2)

        return image_rgb
