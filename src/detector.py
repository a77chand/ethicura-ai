"""
ethicura_ai/src/detector.py
Core detection pipeline: MTCNN face extraction → XceptionNet classification → Grad-CAM overlay.
"""

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

    # ──────────────────────────────────────────────────────────────────────────
    # Lazy loading — don't import heavy libs until actually needed
    # ──────────────────────────────────────────────────────────────────────────

    def _load_models(self):
        """Initialise MTCNN, XceptionNet, and Grad-CAM on first use."""
        if self._model is not None:
            return  # already loaded

        try:
            from mtcnn import MTCNN
            from tensorflow.keras.applications import Xception
            from tensorflow.keras.models import Model, load_model
            from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
            import tensorflow as tf

            # ── Face detector ──────────────────────────────────────────────
            self._face_detector = MTCNN()

            # ── XceptionNet classifier ─────────────────────────────────────
            if self.model_path and Path(self.model_path).exists():
                self._model = load_model(self.model_path)
                print(f"[Ethicura] Loaded fine-tuned weights from {self.model_path}")
            else:
                # Build architecture (requires fine-tuned weights for production accuracy)
                base = Xception(
                    weights="imagenet",
                    include_top=False,
                    input_shape=(299, 299, 3)
                )
                x = GlobalAveragePooling2D()(base.output)
                x = Dropout(0.5)(x)
                out = Dense(1, activation="sigmoid")(x)
                self._model = Model(inputs=base.input, outputs=out)
                print("[Ethicura] Warning: Using ImageNet weights. Fine-tune on FaceForensics++ for production use.")

            # ── Grad-CAM ───────────────────────────────────────────────────
            from src.gradcam import GradCAM
            self._gradcam = GradCAM(self._model, layer_name="block14_sepconv2_act")

        except ImportError as e:
            raise ImportError(
                f"Missing dependency: {e}\n"
                "Install with: pip install mtcnn tensorflow opencv-python"
            )

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

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
        """
        self._load_models()

        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_crop, bbox = self._extract_face(image_rgb)

        if face_crop is None:
            return {
                "label": "UNKNOWN",
                "confidence": None,
                "face_detected": False,
                "gradcam_overlay": image,
                "face_bbox": None,
                "message": "No face detected in the image."
            }

        # Preprocess for XceptionNet
        face_input = self._preprocess(face_crop)

        # Classify
        import tensorflow as tf
        fake_prob = float(self._model.predict(face_input, verbose=0)[0][0])
        confidence = fake_prob * 100
        label = "FAKE" if fake_prob >= self.threshold else "REAL"

        # Grad-CAM overlay
        cam = self._gradcam.compute(face_input)
        overlay = self._overlay_cam(image_rgb.copy(), cam, bbox)
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)

        return {
            "label": label,
            "confidence": round(confidence, 2),
            "face_detected": True,
            "gradcam_overlay": overlay_bgr,
            "face_bbox": bbox,
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

    def _extract_face(self, image_rgb: np.ndarray):
        """Detect and crop the primary face using MTCNN."""
        detections = self._face_detector.detect_faces(image_rgb)
        if not detections:
            return None, None

        # Take the highest-confidence detection
        best = max(detections, key=lambda d: d["confidence"])
        x, y, w, h = best["box"]

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
        """Resize and normalise face crop for XceptionNet input."""
        import tensorflow as tf
        face = cv2.resize(face_rgb, (299, 299))
        face = face.astype("float32")
        face = tf.keras.applications.xception.preprocess_input(face)
        return np.expand_dims(face, axis=0)  # shape: (1, 299, 299, 3)

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
