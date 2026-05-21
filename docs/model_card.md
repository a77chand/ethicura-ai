# Model Card — Ethicura AI (XceptionNet Deepfake Detector)

*Following the Model Card framework by Mitchell et al. (2019)*

---

## Model Details

| Field | Details |
|-------|---------|
| Model name | Ethicura AI — XceptionNet Deepfake Classifier |
| Version | 1.0 |
| Architecture | XceptionNet (fine-tuned) + MTCNN face detector + Grad-CAM |
| Input | Face images (299×299 px, RGB) extracted by MTCNN |
| Output | Binary classification: FAKE (1) / REAL (0) + confidence score |
| Developed by | Anusha Chandra & Revanth Christober M |
| Organisation | The Junior Academy, New York Academy of Sciences |
| Date | 2024–2025 |

---

## Intended Use

**Primary use:** Assisting individuals in identifying potentially manipulated images and videos on social media, in educational content, and in professional contexts.

**Out-of-scope uses:**
- Legal evidence or definitive proof of manipulation (model is a warning tool, not a court-grade forensic system)
- Real-time mass surveillance
- Deployment against marginalised communities without consent

---

## Training Data

- **Primary dataset:** FaceForensics++ (FF++) — 1,000 original videos with 4 deepfake generation methods (Deepfakes, Face2Face, FaceSwap, NeuralTextures)
- **Augmentation:** Random horizontal flip, brightness/contrast jitter, JPEG compression simulation
- **Train/val/test split:** 70% / 15% / 15%

---

## Evaluation Results

| Metric | Score |
|--------|-------|
| Accuracy | 95% |
| Precision | 93% |
| Recall | 92% |
| F1-Score | 92.5% |
| AUC-ROC | 0.97 |

Benchmarked against MesoNet, VA (SVM), ClassNSeg, CapsuleNet, FWA, DSSPFWA, Upcov, WM (EfficientNet), SeimSafer, CNNDetection.

---

## Limitations

1. **High-quality deepfakes:** Detection accuracy degrades on state-of-the-art generation methods not seen during training (e.g. very recent diffusion-based face swaps).
2. **Non-frontal faces:** MTCNN face detection performs best on near-frontal faces. Extreme poses or occlusions may prevent detection.
3. **Low resolution:** Very low-resolution inputs (below 64×64) significantly reduce accuracy.
4. **Demographic bias:** Like most deepfake detectors trained on FF++, performance may vary across skin tones and demographic groups. We recommend ongoing evaluation on diverse datasets.
5. **Adversarial attacks:** The model has not been hardened against adversarial perturbations specifically designed to fool it.

---

## Bias & Fairness Considerations

- FF++ skews toward Western, lighter-skinned subjects. Users deploying this system on diverse populations should evaluate and fine-tune accordingly.
- Grad-CAM explanations help surface whether the model is attending to the correct facial features (manipulation artefacts) rather than spurious correlations (e.g. skin tone, background).
- The model outputs a **confidence score**, not a binary verdict — downstream systems should treat outputs probabilistically and involve human review for high-stakes decisions.

---

## Ethical Use Statement

Ethicura AI is built with transparency and human agency at its core. The Grad-CAM layer exists specifically to make the model's reasoning visible — users are not asked to trust a black box. We designed the system to **warn, not decide**. Any deployment must:

- Inform users that content is being analysed
- Present confidence scores rather than hard verdicts
- Provide a path for appeal or human review
- Not be used to target or harass individuals

---

## Citation

If you use this work, please cite:
```
Chandra, A. & Christober, R. (2025). Ethicura AI: Unified DeepFake Detection 
Framework with XceptionNet and Grad-CAM Interpretability. 
The Junior Academy, New York Academy of Sciences.
```
