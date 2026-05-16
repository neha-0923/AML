# EEEM068 Applied Machine Learning — MSCTD Sentiment Recognition

Three-class sentiment classification (`negative`, `neutral`, `positive`) using the **MSCTD English-German dataset**. The notebook implements a full pipeline: face extraction, image-based models, a fusion model, and an extra-credit multimodal Transformer.

---

## Quick Start

1. Open `FastMode.ipynb` in **Google Colab**.
2. Set runtime to **GPU** (T4 or better).
3. Run all cells in order — the first cell installs all dependencies.
4. By default `FAST_MODE = True` which finishes in around 4 hours.
5. For the full/max-marks run, set `FAST_MODE = False` at the top of Cell 2 and run overnight.

---

## Dataset

The notebook uses the **MSCTD (Multimodal Sentiment Chat Translation Dataset)** — English-German split only.

| Split | Description |
|-------|-------------|
| train | Training samples with English captions, image indices, and sentiment labels |
| dev | Validation split |
| test | Held-out test split |

**Labels:** `0 = neutral`, `1 = negative`, `2 = positive`

### Image Download

Three ZIP files are downloaded automatically from Google Drive:

| File | Drive ID |
|------|----------|
| `train_ende.zip` | `1GAZgPpTUBSfhne-Tp0GDkvSHuq6EMMbj` |
| `dev.zip` | `12HM8uVNjFg-HRZ15ADue4oLGFAYQwvTA` |
| `test.zip` | `1B9ZFmSTqfTMaqJ15nQDrRNLqBvo-B39W` |

If Google Drive quota fails, download the ZIPs manually and place them in:
```
/content/eeem068_msctd_project/data/images/
```

The text/annotation files are cloned from the official repository:
```
https://github.com/XL2248/MSCTD
```

---

## Installation

Dependencies are installed automatically in Cell 1. They can also be installed manually:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install facenet-pytorch gdown transformers timm scikit-learn seaborn accelerate opencv-python pillow
```

---

## Configuration

All main switches are at the top of **Cell 2**. Change these before running:

```python
FAST_MODE = True        # True = 4-hour run. False = full/max-marks run.

# Toggle individual pipeline stages
RUN_FACE_EXTRACTION          = True
RUN_FACE_MODEL               = True
RUN_AUGMENTATION_EXPERIMENT  = True
RUN_FULL_IMAGE_MODEL         = True
RUN_FUSION_MODEL             = True
RUN_MULTIMODAL_EXTRA_CREDIT  = True
RUN_STRONG_MULTIMODAL_FINETUNE = False   # enable for optional deeper fine-tuning
```

### Fast Mode vs Full Mode

| Setting | FAST_MODE = True | FAST_MODE = False |
|---------|-----------------|-------------------|
| Max train samples | 3,500 | All |
| Max val samples | 800 | All |
| Max test samples | 1,000 | All |
| Face model epochs | 3 | 8 |
| Augmented face epochs | 3 | 8 |
| Full-image epochs | 3 | 8 |
| Fusion epochs | 8 | 15 |
| Multimodal epochs | 3 | 5 |

---

## Pipeline Overview

### Part 1 — Face-Based Sentiment Model

- Faces are detected and cropped from each image using **MTCNN**.
- The largest detected face per image is saved to cache.
- A **ResNet-18** classifier (fully fine-tuned) is trained on the cropped faces.
- Images where no face is detected are excluded from the face model training.

### Part 2 — Robustness with Degraded Images

- The trained face model is evaluated on a **degraded** version of the test set.
- Degradations applied randomly per image: brightness shift, Gaussian blur, Gaussian noise, JPEG compression, and small rotation.
- The face model is then **retrained on a 50/50 mix** of original and degraded face crops to improve robustness.

### Part 3 — Full-Image Model (Frozen Backbone)

- Uses the **complete image** (not just the face crop).
- Backbone: **ResNet-50** pretrained on ImageNet, with all backbone weights **frozen**.
- Only the custom classification head is trained (higher learning rate: `1e-3`).

### Part 4 — Fusion Model

Combines the outputs of the face model and full-image model:

```
Feature vector = [face_neg, face_neu, face_pos, img_neg, img_neu, img_pos, n_faces_scaled]
```

A small **MLP** (7 → 64 → 32 → 3) is trained on this 7-dimensional feature vector. The number of detected faces is clipped to 5 and normalised to `[0, 1]`.

### Extra Credit — Multimodal Text + Image Model

- **Text encoder:** DistilBERT (`distilbert-base-uncased`) — uses the `[CLS]` token embedding.
- **Image encoder:** ResNet-18 with the final FC layer replaced by an identity.
- Both encoders are **frozen by default**; only the projection layers and classifier are trained.
- Fusion: text projection (→ 256) and image projection (→ 256) are concatenated (→ 512) and passed through an MLP classifier.

An optional stronger fine-tuning stage (`RUN_STRONG_MULTIMODAL_FINETUNE = True`) unfreezes the last DistilBERT Transformer block and ResNet `layer4` and retrains at `lr=2e-5`.

---

## Output Files

All outputs are saved to `/content/eeem068_msctd_project/`. The notebook zips them at the end.

```
eeem068_msctd_project/
├── results/
│   ├── class_distribution.png
│   ├── sample_images.png
│   ├── face_resnet18_training_curve.png
│   ├── face_resnet18_confusion_matrix.png
│   ├── face_resnet18_on_degraded_confusion_matrix.png
│   ├── full_image_frozen_resnet50_confusion_matrix.png
│   ├── fusion_face_image_nfaces_confusion_matrix.png
│   ├── distilbert_resnet18_fusion_frozen_confusion_matrix.png
│   ├── final_results_summary.csv
│   └── *_history.csv  (training logs per model)
├── models/
│   ├── face_resnet18.pt
│   ├── face_resnet18_augmented.pt
│   ├── full_image_frozen_resnet50.pt
│   ├── fusion_mlp.pt
│   └── distilbert_resnet18_fusion_frozen.pt
└── cache/
    ├── face_meta_train.csv
    ├── face_meta_dev.csv
    └── face_meta_test.csv
```

The final submission ZIP is: `eeem068_results_and_models.zip`

---

## Results Summary

The notebook produces a results table automatically in the final cell. Use it directly in the report.

| Model | Input | Backbone | Accuracy | Macro F1 |
|-------|-------|----------|----------|----------|
| Face ResNet18 | Cropped face | ResNet18 | *generated* | *generated* |
| Face ResNet18 on degraded | Cropped degraded face | ResNet18 | *generated* | *generated* |
| Augmented Face ResNet18 on original | Face + degraded mix | ResNet18 | *generated* | *generated* |
| Augmented Face ResNet18 on degraded | Face + degraded mix | ResNet18 | *generated* | *generated* |
| Full-image frozen ResNet50 | Whole image | ResNet50 (frozen) | *generated* | *generated* |
| Fusion model | Face probs + image probs + face count | MLP | *generated* | *generated* |
| Extra credit: DistilBERT + ResNet18 | English text + image | DistilBERT + ResNet18 | *generated* | *generated* |

---

## How to Maximise Accuracy for the Final Run

Follow this order:

1. Run once with `FAST_MODE = True` to confirm all cells complete without errors.
2. Set `FAST_MODE = False`.
3. Ensure Colab is connected to a GPU runtime.
4. Optionally set `RUN_STRONG_MULTIMODAL_FINETUNE = True` for the deeper fine-tuning stage.
5. Run all cells — this will take several hours.
6. Download `eeem068_results_and_models.zip` from the Colab file browser.

The multimodal model is expected to outperform image-only models because English captions carry clearer and more direct sentiment cues than facial expressions alone.

---

## Notes

- **Face caching:** Face extraction results are saved to CSV. If you re-run the notebook, cached metadata is loaded automatically — face detection does not repeat.
- **Reproducibility:** A fixed random seed (`SEED = 42`) is applied throughout — NumPy, PyTorch, and Python `random` are all seeded.
- **No-face images:** Images where MTCNN detects no face are excluded from Parts 1 and 2 but are still used in Parts 3 and 4.
- **Class imbalance:** A `WeightedRandomSampler` is used in all DataLoaders to address the label imbalance (Negative is the most frequent class).
- **Mixed precision:** AMP (`torch.cuda.amp`) is enabled automatically when a CUDA GPU is available.
