"""
ethicura_ai/demo/run_demo.py

Standalone demo — runs Ethicura AI on a sample image or video
and displays the Grad-CAM annotated result.

Usage:
    python demo/run_demo.py --input path/to/file.jpg
    python demo/run_demo.py --input path/to/video.mp4
    python demo/run_demo.py --demo   # uses the bundled demo video
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from pathlib import Path
from src.detector import EthicuraDetector


def print_banner():
    print("\n" + "═" * 60)
    print("  🛡️  ETHICURA AI — Unified DeepFake Detection Framework")
    print("═" * 60 + "\n")


def run_image_demo(detector, path):
    print(f"  Analysing image: {path}")
    result = detector.predict(path)

    print(f"\n  ┌─────────────────────────────┐")
    print(f"  │  Verdict   : {result['label']:<16} │")
    if result['confidence'] is not None:
        bar_len = int(result['confidence'] / 5)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"  │  Confidence: {result['confidence']:>5.1f}%           │")
        print(f"  │  [{bar}] │")
    print(f"  │  Face found: {'Yes' if result['face_detected'] else 'No':<16} │")
    print(f"  └─────────────────────────────┘\n")

    if result['face_detected'] and result.get('gradcam_overlay') is not None:
        output_path = str(Path(path).stem) + "_ethicura_result.png"
        cv2.imwrite(output_path, result['gradcam_overlay'])
        print(f"  ✅ Grad-CAM overlay saved → {output_path}")

        # Show if display available
        try:
            cv2.imshow("Ethicura AI — Grad-CAM Result", result['gradcam_overlay'])
            print("  (Press any key to close the window)")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception:
            pass  # Headless environment

    return result


def run_video_demo(detector, path):
    print(f"  Analysing video: {path}")
    print("  (Sampling every 10th frame...)\n")

    result = detector.predict_video(path, sample_every=10)

    verdict_icon = "🚨 FAKE" if result.get("verdict") == "FAKE" else "✅ REAL"
    print(f"\n  ┌─────────────────────────────────────┐")
    print(f"  │  Verdict        : {result.get('verdict', 'UNKNOWN'):<18} │")
    print(f"  │  Fake frames    : {result.get('fake_frame_count', 'N/A')!s:<18} │")
    print(f"  │  Frames sampled : {result.get('total_frames_sampled', 'N/A')!s:<18} │")
    print(f"  │  Avg confidence : {result.get('avg_confidence', 'N/A')!s:<18} │")
    print(f"  │  Fake ratio     : {result.get('fake_ratio', 'N/A')!s:<18} │")
    print(f"  └─────────────────────────────────────┘\n")
    print(f"  {verdict_icon}\n")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Ethicura AI deepfake detection demo"
    )
    parser.add_argument("--input", type=str, help="Path to image or video file")
    parser.add_argument("--demo", action="store_true", help="Run on bundled demo video")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Fake confidence threshold (default: 0.5)")
    args = parser.parse_args()

    print_banner()

    if args.demo:
        demo_path = os.path.join(os.path.dirname(__file__), "..", "assets", "demo.mp4")
        if not os.path.exists(demo_path):
            print("  ⚠️  Demo video not found at assets/demo.mp4")
            print("  Run with --input path/to/your/file instead.\n")
            sys.exit(1)
        args.input = demo_path

    if not args.input:
        parser.print_help()
        sys.exit(0)

    if not os.path.exists(args.input):
        print(f"  ❌ File not found: {args.input}\n")
        sys.exit(1)

    detector = EthicuraDetector(threshold=args.threshold)
    ext = Path(args.input).suffix.lower()

    if ext in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        run_image_demo(detector, args.input)
    elif ext in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
        run_video_demo(detector, args.input)
    else:
        print(f"  ❌ Unsupported file type: {ext}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
