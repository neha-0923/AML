# EEEM068: Applied Machine Learning — Human Sentiment Analysis

*University of Surrey | Institute for People-Centred AI | Spring Term 2026*  
*Module:* EEEM068: Applied Machine Learning  
*Project:* Human Sentiment Analysis on the MSCTD Dataset  

---

## Overview

This project implements a multimodal sentiment analysis system that classifies images into three sentiment categories — *Neutral, **Negative, and **Positive* — using the English-German (En-De) split of the [MSCTD Dataset](https://github.com/XL2248/MSCTD). The pipeline covers four main parts plus an extra-credit multimodal Transformer extension.

| Part | Description |
|------|-------------|
| *Part 1* | Face-based sentiment classification using MTCNN + ResNet-18 |
| *Part 2* | Data augmentation & robustness evaluation (clean vs. degraded images) |
| *Part 3* | Full-image transfer learning with frozen ResNet-50 backbone |
| *Part 4* | Fusion model combining face-branch and full-image-branch predictions |
| *Extra Credit* | Multimodal Transformer fine-tuning (text + image) |

---

## Repository Structure


.
├── Applied_Machine_Learning_EEEM068.ipynb   # Main notebook (all 4 parts)
├── requirements.txt                                    # Python dependencies
├── README.md                                           # This file
└── report/
    └── HAR_15195_Report.pdf                           # 5-page IEEE report (submit separately)


---

## Dataset

The project uses the *MSCTD (Multimodal Sentiment Chat Translation Dataset), specifically the **En-De* split.

- The notebook (Cell 3) *automatically downloads* all images and label files from the official GitHub repository and Google Drive using gdown and wget.
- No manual dataset download is required.
- Labels: 0 = Neutral, 1 = Negative, 2 = Positive

> *Note:* You must have internet access when running the notebook for the first time. Once downloaded, data is cached at /content/msctd/data/ende/.

---

## Setup & Usage

### Option 1: Google Colab (Recommended)

1. Open the notebook in Google Colab.
2. Ensure the runtime is set to *GPU* (Runtime → Change runtime type → T4 GPU).
3. Run *Cell 1* first to install all dependencies, then *restart the runtime*.
4. Run all remaining cells sequentially (Cell 2 onwards).

### Option 2: Local Environment

bash
# Clone the repository
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook Applied_Machine_Learning_EEEM068.ipynb


> A CUDA-capable GPU is strongly recommended for training. CPU-only execution is possible but will be very slow.

---

## Model Architectures

### Part 1 — Face Sentiment Model (FaceSentimentModel)
- *Backbone:* ResNet-18 (ImageNet pre-trained)
- *Head:* Dropout(0.4) → Linear(512 → 3)
- *Face detector:* MTCNN (via mtcnn library)
- *Multi-face strategy:* Average softmax across all detected faces
- *No-face fallback:* Uniform prior [1/3, 1/3, 1/3]

### Part 2 — Robust Face Model
- Same architecture as Part 1, retrained on a *mixed dataset* (clean + degraded images)
- Degradation pipeline: FFT frequency noise + Gaussian noise + JPEG compression + brightness/contrast jitter + rotation

### Part 3 — Full Image Model (FullImageModel)
- *Backbone:* ResNet-50 (ImageNet pre-trained, *fully frozen*)
- *Head:* Flatten → Linear(2048→512) → ReLU → Dropout(0.4) → Linear(512→3)

### Part 4 — Fusion Model (FusionModel)
- *Input:* 10-dim vector = face probs (3) + full-image probs (3) + face-count OHE (4)
- Combines both branches with face-count awareness for the final prediction

---

## Evaluation Metrics

All models are evaluated on the *test split* using:
- *Accuracy*
- *Macro F1-Score*
- *Confusion Matrix*
- *ROC Curves* (per class, with AUC)

A summary table of all results is printed in *Cell 20*.

---

## Results Summary

| Model | Test Accuracy | Macro F1 |
|-------|:---:|:---:|
| Part 1  Face Model | — | — |
| Part 2  Robust Model (Clean) | — | — |
| Part 2  Robust Model (Degraded) | — | — |
| Part 3  Full Image (ResNet-50) | — | — |
| Part 4  Fusion Model | — | — |

> Fill in values after running the notebook.

---

## Saved Model Weights

All trained model weights are saved to /content/ in the Colab environment at the end of training:

| File | Description |
|------|-------------|
| face_model.pth | Part 1 face sentiment model |
| face_model_robust.pth | Part 2 mixed-trained robust model |
| full_image_model.pth | Part 3 frozen ResNet-50 model |
| fusion_model.pth | Part 4 fusion model |

Download these from Colab using the Files panel before the session ends.

---

## Dependencies

See [requirements.txt](requirements.txt) for the full list. Core dependencies:

- torch / torchvision (CUDA 12.1)
- mtcnn — face detection
- albumentations — image augmentation
- scikit-learn — metrics
- Pillow==10.4.0 — image I/O (pinned for compatibility)
- gdown — Google Drive downloads

---

## Reproducibility

All random seeds are fixed at the start of the notebook:
python
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


---

## Group Members

| Name | URN |
|------|-----|
|Likhitha Reddy |6956685 |
|Neha Madhava Sundhram |6957792 |
|Pramodh Rahul Gullipalli |6964218 |
|Harshitha | |
|Malli | |

---

## References

- MSCTD Dataset: [https://github.com/XL2248/MSCTD](https://github.com/XL2248/MSCTD)
- Multimodal Transformers Survey: [https://arxiv.org/pdf/2206.06488.pdf](https://arxiv.org/pdf/2206.06488.pdf)
- LXMERT: [https://github.com/airsplay/lxmert](https://github.com/airsplay/lxmert)
- ClipBERT: [https://github.com/jayleicn/ClipBERT](https://github.com/jayleicn/ClipBERT)

---

## License

This project was developed for academic purposes as part of EEEM068 at the University of Surrey. Not for commercial use.
