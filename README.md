# ‣ Ethicura AI — Unified DeepFake Detection Framework (UDDF)

[![tests](https://github.com/a77chand/ethicura-ai/actions/workflows/tests.yml/badge.svg)](https://github.com/a77chand/ethicura-ai/actions/workflows/tests.yml) [![live demo](https://img.shields.io/badge/demo-live-brightgreen)](https://ethicura-frontend.vercel.app)

> *Empowering digital trust through real-time, interpretable deepfake detection.*

![Project Summary](assets/summary.png)

## ‣ Live Demo:

**[ethicura-frontend.vercel.app](https://ethicura-frontend.vercel.app)**: live webcam deepfake analysis with a heatmap overlay. It runs **entirely in your browser**, and your video never leaves the device:

- **Face tracking:** MediaPipe BlazeFace finds the face on every video frame.
- **Classifier:** an ONNX export of the classifier runs on your GPU through WebGPU, falling back to WebAssembly.
- **Speed:** about 30 analyses per second at about 20 ms each on a laptop, with no server and no network round trip.

### How the browser version matches the Python API
- **Heatmap without a backward pass.** The classifier head is global-average-pool → dropout → dense(sigmoid), so Grad-CAM's channel weights reduce to a positive scalar times the dense weights. After ReLU and normalisation, Grad-CAM equals classic CAM, `ReLU(Σₖ wₖ Aₖ) / max`. The exported graph computes that directly. [`training/export_web.py`](training/export_web.py) checks the result against `src/gradcam.py` numerically before it writes the model.
- **Crop parity.** The model was trained on MTCNN face crops, but the browser uses BlazeFace. Over 73 face frames, BlazeFace boxes are mapped onto MTCNN's with a calibrated offset and scale (median IoU 0.91 with the API's crop).
- **Verified end to end.** On those frames, the browser's fake probability differs from the Python API's by **at most 0.0004**, and every frame gets the same REAL/FAKE label.

## ‣ Demo Video:

[https://github.com/user-attachments/assets/demo.mp4](https://github.com/a77chand/ethicura-ai/raw/main/assets/api_3.mp4)

> The video shows the original XceptionNet version of Ethicura AI analysing a video of Elon Musk. It flags the video as a deepfake with a **57.5% fake confidence score**, and a Grad-CAM saliency overlay highlights the manipulated facial regions.

---

## ‣ The Problem:

Deepfake technology makes it trivially easy to manipulate digital content — spreading misinformation, invading privacy, and damaging reputations. Individuals, businesses, and institutions face growing threats from scams, identity theft, and fake social media content.

We built Ethicura AI to give people a proactive, interpretable tool to verify content authenticity and rebuild trust in digital media.

---

## ‣ Our Solution:

**Ethicura AI** is an end-to-end deepfake detection API and interface that:

- Detects manipulated **images and videos** using a pretrained **EfficientNetB0** classifier ([`ariyul/deepfake-detection-cnn_v2`](https://huggingface.co/ariyul/deepfake-detection-cnn_v2) on Hugging Face) and **MTCNN** for face extraction
- Supports **live webcam analysis in the browser**: a React frontend runs face tracking and an ONNX export of the classifier on-device (WebGPU), with no server and no upload
- Provides **real-time confidence scores** (fake probability 0–100%)
- Generates **Grad-CAM visualisations** to highlight the exact facial regions that triggered the detection — making the model's decision transparent and auditable
- Identifies three types of manipulation:
  - Subtle facial distortions (images)
  - Spatial inconsistencies (videos)
  - Unnatural frame transitions (video sequences)

---

## ‣ Architecture:

```
Input (Image / Video)
        │
        ▼
┌───────────────────┐
│   MTCNN           │  ← Face detection & extraction
│   Face Detector   │     Crops facial region from each frame
└───────────────────┘
        │
        ▼
┌───────────────────┐
│   EfficientNetB0  │  ← Deepfake classification
│   Classifier      │     Pretrained (Hugging Face), 224×224 input
└───────────────────┘
        │
        ▼
┌───────────────────┐
│   Grad-CAM        │  ← Interpretability layer
│   Visualiser      │     Generates saliency map over face
└───────────────────┘
        │
        ▼
Output: Confidence Score + Annotated Frame
```

---

## ‣ Model Performance:

> **Which model these numbers describe:** the results below are from the original research model, an XceptionNet fine-tuned on FaceForensics++. Its trained weights were never committed to this repo, so the current code and the live demo use a third-party pretrained EfficientNetB0 ([`ariyul/deepfake-detection-cnn_v2`](https://huggingface.co/ariyul/deepfake-detection-cnn_v2)). These benchmark figures don't apply to that model.

The XceptionNet model was benchmarked against 10 leading deepfake detection models on FaceForensics++:

| Model | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | AUC-ROC |
|-------|-------------|---------------|------------|--------------|---------|
| **XceptionNet (Ours)** | **95** | **93** | **92** | **92.5** | **0.97** |
| MesoNet | 88 | 85 | 86 | 85.5 | 0.9 |
| VA (SVM) | 80 | 78 | 75 | 76.5 | 0.85 |
| ClassNSeg | 92 | 90 | 89 | 89.5 | 0.93 |
| CapsuleNet | 85 | 82 | 83 | 82.5 | 0.88 |
| FWA | 78 | 76 | 72 | 74 | 0.8 |
| DSSPFWA | 90 | 88 | 86 | 87 | 0.91 |
| Upcov | 82 | 80 | 79 | 79.5 | 0.85 |
| WM (EfficientNet) | 94 | 92 | 91 | 91.5 | 0.96 |
| SeimSafer (EfficientNet) | 92 | 90 | 89 | 89.5 | 0.94 |
| CNNDetection | 91 | 89 | 88 | 88.5 | 0.93 |

**The XceptionNet model achieved the highest performance across all metrics.**

---

## ‣ Training an In-House Model (in progress):

[`training/run.py`](training/run.py) trains and evaluates a replacement for the third-party model end to end. It runs on a free Kaggle T4 GPU through [`training/kaggle/`](training/kaggle/). Results will be added here once the run completes. Until then, no performance claims are made for it.

**Data**
- **[DeepFakeFace](https://huggingface.co/datasets/OpenRL/DeepFakeFace)** (OpenRAIL): 30k real IMDB-WIKI photos. Each has one fake from each of three generators: Stable Diffusion inpainting, InsightFace face swap and Stable Diffusion text-to-image.
- **[SDFVD](https://huggingface.co/datasets/Hemgg/SDFVD-video-dataset):** 53 real and 53 face-swapped videos from a commercial tool. It's used **only as an out-of-domain test set**.

**Shortcut controls.** Inspecting the raw data showed two easy cues that have nothing to do with manipulation:
- **Size and format:** real photos have varied sizes, about 7.5% are grayscale, and some are PNG. Every fake is a 512×512 colour JPEG.
- **Sharpness:** small real faces upscale to blur, while the fakes are sharp.

A model can score well by learning these cues and then fail on real webcam input. The pipeline closes them as follows:
- **Grayscale images are dropped** from every class.
- **Every image goes through the API's own MTCNN face crop** and is re-encoded identically, which removes size and format differences.
- **Faces under 80 px wide are dropped.** Training also applies random blur, downscaling and JPEG recompression (q35–100) to **both** classes, so sharpness and compression carry no label information.
- **Splits are by person (80/10/10):** a celebrity's real photo and all of its fakes always land in the same split.

**Training**
- **Model:** EfficientNetB0 (ImageNet), chosen for CPU latency in the live service and WebGPU in the browser.
- **Schedule:** the head is trained first on the frozen backbone, then the whole network is fine-tuned with AdamW and cosine decay.
- **Batches are class-balanced,** with the fakes split evenly across the generators.
- **Checkpoint selection:** the best epoch is chosen by validation AUC.
- **Calibration:** a temperature is fitted on a 50/50 validation slice and folded into the final layer, so P(fake) is calibrated at no runtime cost.

**Evaluation** (held-out people, compared against the third-party baseline on identical crops)
- **Headline metrics:** AUC with a 95% CI from a person-level bootstrap, EER, the detection rate at 1% and 5% false-positive rates, and balanced accuracy.
- **Calibration:** ECE and reliability diagrams, before and after temperature scaling.
- **Per-generator AUC:** inpainting, face swap and text-to-image reported separately.
- **Robustness:** AUC under JPEG q90/70/50/30, ×0.5 and ×0.25 downscaling, and Gaussian blur.
- **Unseen generators:** three extra models are trained, each with one generator left out, then tested on that generator.
- **Cross-dataset:** frame-level and video-level AUC on SDFVD face-swap videos.

```bash
pip install -r requirements-dev.txt
python -m training.run all --data /tmp/ethicura --out results              # full run (GPU recommended)
python -m training.run all --data /tmp/ethicura --out results --limit 50   # quick smoke test
```

---

## ‣ Project Structure:

```
ethicura-ai/
├── src/
│   ├── detector.py          # Core detection pipeline (MTCNN + EfficientNetB0 + Grad-CAM)
│   ├── gradcam.py           # Grad-CAM visualisation module
│   └── api.py               # FastAPI REST endpoints (incl. webcam /predict_frame)
├── demo/
│   └── run_demo.py          # Standalone demo script
├── assets/
│   ├── summary.png          # Project overview image
│   └── api_3.mp4            # Demo video
├── docs/
│   └── model_card.md        # Model card (bias, limitations, intended use)
├── training/
│   ├── export_web.py        # Keras → ONNX (prob + CAM) for the in-browser demo, with parity checks
│   ├── run.py               # Train + evaluate an in-house model (download → crops → train → evaluate)
│   ├── kaggle/              # Runs run.py on a free Kaggle GPU
│   ├── data.py, model.py, metrics.py
├── tests/                   # pytest suite (model contract, calibration, metrics, data pipeline)
├── fetch_and_test_model.py  # Downloads the pretrained model and runs a smoke test
├── test_full_pipeline.py    # End-to-end test: face detection → classification → Grad-CAM
├── modal_app.py             # Modal deployment (hosts the live API)
├── Dockerfile               # Container image for other hosts
├── render.yaml              # Render deployment config
├── requirements.txt
└── README.md
```

The in-browser frontend (React + Vite, ONNX Runtime Web, MediaPipe) lives in its own repo: **[a77chand/ethicura-frontend](https://github.com/a77chand/ethicura-frontend)**.

---

## ‣ Setup & Usage:

### 1. Clone and install
```bash
git clone https://github.com/a77chand/ethicura-ai.git
cd ethicura-ai
pip install -r requirements.txt
```

### 2. Run on an image
```python
from src.detector import EthicuraDetector

detector = EthicuraDetector()
result = detector.predict("path/to/image.jpg")

print(f"Prediction: {result['label']}")          # FAKE or REAL
print(f"Confidence: {result['confidence']:.1f}%")
# result['gradcam_overlay'] contains the annotated image
```

### 3. Run on a video
```python
result = detector.predict_video("path/to/video.mp4")
print(f"Fake frames detected: {result['fake_frame_count']} / {result['total_frames']}")
print(f"Overall verdict: {result['verdict']}")
```

### 4. Run the API server
```bash
uvicorn src.api:app --reload
# API available at http://localhost:8000
```

On first use the pretrained model is downloaded from Hugging Face and cached locally.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check; `model_ready` turns true once the model is loaded and warmed up |
| `/model-info` | GET | Model and benchmark summary |
| `/predict` | POST | Upload an image or video (multipart `file`) → label, confidence, per-video stats |
| `/predict/gradcam` | POST | Upload an image → Grad-CAM overlay as PNG (label/confidence in `X-Ethicura-*` headers) |
| `/predict_frame` | POST | JSON `{"image_base64": "..."}` → label, confidence, `face_bbox` and the raw 7×7 Grad-CAM `heatmap` (~500 bytes), for clients that stream frames to a server (the live demo runs in the browser instead). Add `"include_overlay": true` to also get the annotated frame as base64 PNG |

The models load and warm up in a background thread at startup, so the port opens immediately and `/health` reports when inference is ready. Classification and Grad-CAM share a single traced forward/backward pass. On an Apple-silicon laptop a webcam frame takes about 130 ms end to end.

### 5. Run the webcam frontend
```bash
git clone https://github.com/a77chand/ethicura-frontend.git && cd ethicura-frontend
npm install
npm run dev
```
The frontend runs the model in the browser and doesn't need this API. To regenerate its model file from a Keras model, run:
```bash
python -m training.export_web --out ../ethicura-frontend/public/models/detector.onnx [--model path/to/model.keras]
```

### 6. Deploy
The live demo is a static site on Vercel and needs no backend: all inference runs in the browser.

To host the FastAPI service anyway (for image and video uploads), use one of these:
- `modal_app.py` deploys it on [Modal](https://modal.com) with `modal deploy modal_app.py`. Expect about a 22 s cold start and about 170 ms per frame.
- The `Dockerfile` builds a self-contained image (model baked in, port 7860) for any container host, including Hugging Face Docker Spaces, which now require a PRO plan.
- `render.yaml` is kept for Render, but the free tier's 512 MB RAM / 0.1 CPU is too small for live webcam use.

XLA JIT is disabled in `src/api.py` because its first-call compile stalled for many minutes on constrained CPUs.

---

## ‣ Future Potential:

- **Education & e-learning**: Verify authenticity of online instructional content, exams, and interviews
- **Professional networking**: Assist platforms like LinkedIn, Glassdoor, and HR systems in verifying user genuineness
- **Multilingual support & voice-guided interfaces** for accessibility and inclusivity
- **Real-time browser extension** for social media verification

---

## ‣ Technical Stack:

| Component | Technology |
|-----------|-----------|
| Face Detection | MTCNN (Multi-Task Cascaded CNN) |
| Classification | EfficientNetB0 (pretrained, `ariyul/deepfake-detection-cnn_v2`) |
| Interpretability | Grad-CAM (Class Activation Mapping) |
| API | FastAPI |
| Frontend | React + Vite, hosted on Vercel |
| In-browser inference | ONNX Runtime Web (WebGPU / WebAssembly), MediaPipe BlazeFace |
| Deep Learning | TensorFlow / Keras |
| Video Processing | OpenCV |
| Optional API hosting | Modal or any container host |

---

## ‣ Ethics & Responsible AI:

Ethicura AI was built with responsible deployment in mind:
- Grad-CAM explanations ensure decisions are **interpretable**, not black-box
- Intended as a **warning tool**, not a definitive judge — users retain agency
- Does not store uploaded content
- See [`docs/model_card.md`](docs/model_card.md) for the bias analysis and limitations of the original XceptionNet model

--- 
