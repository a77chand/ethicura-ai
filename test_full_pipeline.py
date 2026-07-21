"""
Run this LOCALLY, from inside your ethicura-ai repo folder, AFTER copying
the updated detector.py and gradcam.py into src/ (replacing the old ones).

This exercises the real, full pipeline end-to-end:
    MTCNN face detection -> EfficientNetB0 classification -> Grad-CAM overlay

using an actual downloaded face photo, not a mock.

Usage:
    python3 test_full_pipeline.py
"""

import os
import urllib.request

# Grab a small public test photo with a clear face (a standard test image
# used in many CV tutorials, hosted on Wikimedia).
TEST_IMAGE_PATH = "test_face.jpg"  # <- whatever you actually named the file

from src.detector import EthicuraDetector

print("\nInitialising detector (this will download the pretrained model on first run)...")
detector = EthicuraDetector()

print("\nRunning prediction...")
result = detector.predict(TEST_IMAGE_PATH)

print("\n--- Result ---")
print("Label:", result["label"])
print("Confidence:", result.get("confidence"))
print("Face detected:", result["face_detected"])
print("Face bbox:", result.get("face_bbox"))

if result["face_detected"]:
    import cv2
    cv2.imwrite("gradcam_output.jpg", result["gradcam_overlay"])
    print("\nSaved Grad-CAM overlay to gradcam_output.jpg — open it and check")
    print("that it shows a heatmap over the face region, not garbage/blank.")
else:
    print("\nNo face detected in the test image — try a different photo of an")
    print("actual person (a selfie works well) and rerun with that path instead.")
