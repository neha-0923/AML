# Extra Credit — Multimodal Transformer (CLIP + BERT)

**Module:** EEEM068: Applied Machine Learning | University of Surrey | Spring Term 2026

---

## Overview

This notebook extends the main sentiment analysis pipeline by adding the English dialogue text alongside the scene image. Instead of relying on visual cues alone, it uses two pre-trained Transformer backbones — CLIP for images and BERT for text — fused through a cross-attention mechanism to classify sentiment into **Neutral**, **Negative**, or **Positive**.

---

## File

| File | Description |
|------|-------------|
| `extra_credit_multimodal.ipynb` | Full extra credit notebook (EC1 to EC9) |

---

## How to Run

1. Open `extra_credit_multimodal.ipynb` in Google Colab
2. Set runtime to **GPU** → Runtime → Change runtime type → **T4 GPU**
3. Run **Cell EC1** to install dependencies, then **restart the runtime**
4. Run all remaining cells in order (EC2 → EC9)

> The dataset downloads automatically. No manual setup needed.
> Make sure the main notebook has run first — or run this standalone, Cell EC3 handles its own download.

---

## Model Architecture

- **Vision encoder:** CLIP ViT-B/32 (`openai/clip-vit-base-patch32`) — last 2 encoder layers unfrozen, 768-dim pooled output
- **Text encoder:** BERT-base-uncased — last 2 encoder layers unfrozen, 768-dim [CLS] pooler output
- **Projection heads:** Both streams projected to 256 dims via LayerNorm + GELU
- **Fusion:** Cross-attention where image queries the text, attended output added residually + LayerNorm
- **Classifier:** Concat (256 + 256) → Linear(512→256) → GELU → Dropout → Linear(256→3)

---

## Training Setup

| Setting | Value |
|---------|-------|
| Epochs | 10 |
| Warmup | 2 epochs (linear) |
| LR schedule | Cosine decay after warmup |
| Backbone LR | 1e-5 |
| Head LR | 3e-4 |
| Loss | Cross-entropy + label smoothing (ε=0.1) + class weights |
| Gradient clipping | max norm = 1.0 |
| Batch size | 32 |
| Best checkpoint | Saved by dev accuracy |

---

## Results

**Test Accuracy: 60.31% | Macro F1: 0.5865** (5,067 test samples)

| Class | Precision | Recall | F1 | AUC |
|-------|-----------|--------|----|-----|
| Neutral | 0.49 | 0.45 | 0.47 | 0.730 |
| Negative | 0.71 | 0.64 | 0.67 | 0.797 |
| Positive | 0.56 | 0.68 | 0.62 | 0.788 |

![Confusion Matrix & ROC Curves](results/screenshots/extra_credit_clip_bert_cm_roc.png)

---

## Saved Checkpoints

| File | Description |
|------|-------------|
| `/content/multimodal_model.pth` | Best model weights (by dev accuracy) |
| `/content/ec_eval_test.png` | Confusion matrix + ROC figure |

> Download from Colab Files panel before the session ends.

---

## Dependencies

```
torch torchvision
transformers>=4.37.0
sentencepiece
Pillow
scikit-learn
matplotlib seaborn
tqdm
```

Install via Cell EC1 or manually: `pip install transformers>=4.37.0 sentencepiece`

