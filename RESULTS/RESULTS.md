# EEEM068 – Applied Machine Learning
## Human Sentiment Analysis — Results Summary

**Dataset:** MSCTD En-De (Multimodal Sentiment Chat Translation)  
**Classes:** Neutral (0) · Negative (1) · Positive (2)  
**Hardware:** Google Colab T4 GPU  
**Framework:** PyTorch

---

## Part 1 — Face-Based Sentiment (EfficientNet-B2)

| Metric | Value |
|--------|-------|
| Test Accuracy | 36.44% |
| Macro F1 | 0.1781 |

**Setup:** MTCNN face detection → EfficientNet-B2 backbone (last 3 blocks unfrozen) → MLP head (1408→512→128→3). Trained for 15 epochs with AdamW, label smoothing (ε=0.1), cosine LR decay. No-face fallback to class prior; multi-face confidence-weighted averaging.

**Notes:** Neutral was the hardest class — it overlaps visually with both others. The model showed class imbalance bias toward Negative, motivating the weighted loss and robustness experiments in Part 2.

---

## Part 2 — Robustness to Image Degradation

| Model | Clean Acc | Clean F1 | Degraded Acc | Degraded F1 |
|-------|-----------|----------|--------------|-------------|
| Original face model | — | — | — | — |
| Robust (mixed) model | — | — | — | — |

**Degradation pipeline:** FFT-domain noise · JPEG compression (q=50–70) · Gaussian noise · blur · brightness/contrast shifts · random rotation (Albumentations).  
**Robust retraining:** 50/50 mix of clean and degraded training images.

> Run the notebook to populate the degraded accuracy numbers — they are printed by Cell 14 and Cell 15.

---

## Part 3 — Full-Image Feature Extraction (EfficientNet-B3)

| Metric | Value |
|--------|-------|
| Test Accuracy | — |
| Macro F1 | — |

**Setup:** Frozen EfficientNet-B3 backbone → MLP head (1536→512→128→3). Focal loss (γ=2), dual LR (1e-5 backbone / 1e-4 head), 10 epochs, test-time augmentation (TTA × 3).

> Full-image accuracy is printed by Cell 16. It improves on Part 1 especially on the Neutral class, where scene context helps more than facial crops alone.

---

## Part 4 — Attention Fusion Model

| Metric | Value |
|--------|-------|
| Test Accuracy | — |
| Macro F1 | — |

**Setup:** 10-dim input = face softmax (3) + full-image softmax (3) + face-count one-hot (4). Attention gate (sigmoid MLP) learns which stream to trust. Trained for 100 epochs with AdamW and focal loss.

> Fusion accuracy is printed by Cell 19. It gives the best image-only results — the attention gate learns to rely on the full-image stream when no face is detected.

---

## Extra Credit — Multimodal Transformer (CLIP + BERT)

| Metric | Value |
|--------|-------|
| Test Accuracy | **60.31%** |
| Macro F1 | **0.5865** |
| Test samples | 5,067 |

### Per-Class Results

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Neutral | 0.49 | 0.45 | 0.47 | 1,298 |
| Negative | 0.71 | 0.64 | 0.67 | 2,163 |
| Positive | 0.56 | 0.68 | 0.62 | 1,606 |
| **Macro avg** | **0.59** | **0.59** | **0.59** | 5,067 |
| Weighted avg | 0.61 | 0.60 | 0.60 | 5,067 |

### ROC AUC Scores

| Class | AUC |
|-------|-----|
| Neutral | 0.730 |
| Negative | 0.797 |
| Positive | 0.788 |

**Setup:**  
- Vision encoder: CLIP ViT-B/32 (`openai/clip-vit-base-patch32`) — last 2 layers unfrozen  
- Text encoder: BERT-base-uncased — last 2 layers unfrozen  
- Fusion: cross-attention (image queries text) + residual LayerNorm → concat → classifier  
- Training: 10 epochs, linear warmup (2 epochs) + cosine decay, AdamW (backbone lr=1e-5, head lr=3e-4), class-weighted CE + label smoothing (ε=0.1), gradient clipping (max=1.0)

---

## Summary Table

| Model | Test Accuracy | Macro F1 |
|-------|--------------|----------|
| Part 1 — Face model (EfficientNet-B2) | 36.44% | 0.1781 |
| Part 2 — Robust face model (clean) | — | — |
| Part 2 — Robust face model (degraded) | — | — |
| Part 3 — Full image (EfficientNet-B3) | — | — |
| Part 4 — Fusion model | — | — |
| **Extra Credit — CLIP + BERT** | **60.31%** | **0.5865** |

> Fill in the `—` values from your Colab run output (Cell 21 prints the full summary table).

---

## How to Reproduce

1. Open `APPLIED_ML_revised.ipynb` in Google Colab (GPU runtime)
2. Run Cell 1, restart runtime, then run all remaining cells in order
3. Open `extra_credit_multimodal.ipynb` in a separate Colab session (GPU runtime)
4. Run all cells — dataset will be downloaded automatically

**Saved checkpoints** (in `/content/` after training):
```
face_model.pth
face_model_robust.pth
full_image_model.pth
fusion_model.pth
multimodal_model.pth        ← extra credit
ec_eval_test.png            ← extra credit CM + ROC figure
```

---

## References

1. Y. Liang et al., "MSCTD: A Multimodal Sentiment Chat Translation Dataset," ACL 2022. https://doi.org/10.18653/v1/2022.acl-long.186  
2. J. Ren, "Multimodal Sentiment Analysis Based on BERT and ResNet," arXiv 2024. https://arxiv.org/abs/2412.03625  
3. S. Poria et al., "A review of affective computing: From unimodal analysis to multimodal fusion," Information Fusion, vol. 37, 2017.  
4. Y. Cai et al., "Multimodal sentiment analysis based on multi-layer feature fusion," Scientific Reports, 2025.  
5. G. Meena et al., "Sentiment analysis on images using Inception-V3 transfer learning," IJIMDI, 2023.
