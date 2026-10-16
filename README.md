# ‣ Ethicura AI — Unified DeepFake Detection Framework (UDDF)

> *Empowering digital trust through real-time, interpretable deepfake detection.*

![Project Summary](assets/summary.png)

## ‣ Live Demo:

**[ethicura-frontend.vercel.app](https://ethicura-frontend.vercel.app)**: live webcam deepfake analysis with a Grad-CAM overlay, running in your browser against the hosted API. The API sleeps when idle, so the first visit after a break shows "waking server" for about 20 seconds.

Interactive API docs: [/docs on the hosted API](https://anusha-chandra-007--ethicura-ai-api-api.us-east.modal.run/docs)

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
- Supports **live webcam analysis** via a React frontend that streams frames to the API
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
├── fetch_and_test_model.py  # Downloads the pretrained model and runs a smoke test
├── test_full_pipeline.py    # End-to-end test: face detection → classification → Grad-CAM
├── modal_app.py             # Modal deployment (hosts the live API)
├── Dockerfile               # Container image for other hosts
├── render.yaml              # Render deployment config
├── requirements.txt
└── README.md
```

The React webcam frontend lives in a separate project, `ethicura-frontend/` (Vite + React). It is not part of this repository.

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
| `/predict_frame` | POST | JSON `{"image_base64": "..."}` → label, confidence, `face_bbox` and the raw 7×7 Grad-CAM `heatmap` (~500 bytes); built for live webcam use. Add `"include_overlay": true` to also get the annotated frame as base64 PNG |

The models load and warm up in a background thread at startup, so the port opens immediately and `/health` reports when inference is ready. Classification and Grad-CAM share a single traced forward/backward pass. On an Apple-silicon laptop a webcam frame takes about 130 ms end to end.

### 5. Run the webcam frontend
```bash
cd ../ethicura-frontend
npm install
npm run dev
```
The frontend keeps the live webcam feed on screen and draws the Grad-CAM heatmap and face box over it, gliding the box between results. Two frames are kept in flight, so one uploads while the other is being processed. Each frame is captured at send time, and any out-of-order response is dropped, so analysis always runs on the current frame and requests never pile up. The page polls `/health` on load and shows "waking server" while a cold backend starts up. It points at `http://localhost:8000` by default; set `VITE_API_URL` at build time to use a deployed backend.

### 6. Deploy
The hosted API runs on [Modal](https://modal.com) (2 vCPU / 4 GB, scales to zero when idle):
```bash
pip install modal
modal token new            # one-time browser login
modal deploy modal_app.py
```
A cold start takes about 7 s to answer and about 22 s until `model_ready`. Once warm, inference is about 170 ms per frame and the webcam UI runs at about 5 frames/s.

The frontend is deployed on Vercel. `ethicura-frontend/.env.production` sets `VITE_API_URL` to the Modal endpoint.

Other options:
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
| Frontend | React + Vite (webcam streaming), hosted on Vercel |
| Deep Learning | TensorFlow / Keras |
| Video Processing | OpenCV |
| API Hosting | Modal |

---

## ‣ Ethics & Responsible AI:

Ethicura AI was built with responsible deployment in mind:
- Grad-CAM explanations ensure decisions are **interpretable**, not black-box
- Intended as a **warning tool**, not a definitive judge — users retain agency
- Does not store uploaded content
- See [`docs/model_card.md`](docs/model_card.md) for the bias analysis and limitations of the original XceptionNet model

--- 
