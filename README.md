EEEM068: Applied Machine Learning — Human Sentiment Analysis
University of Surrey 

Overview
This project implements a multimodal sentiment analysis system that classifies images into three sentiment categories — Neutral, Negative, and Positive — using the English-German (En-De) split of the MSCTD Dataset. The pipeline covers four main parts plus an extra-credit multimodal Transformer extension.
PartDescriptionPart 1Face-based sentiment classification using MTCNN + EfficientNet-B2Part 2Data augmentation & robustness evaluation (clean vs. degraded images)Part 3Full-image transfer learning with frozen EfficientNet-B3 backbonePart 4Attention fusion model combining face and full-image predictionsExtra CreditMultimodal Transformer fine-tuning (CLIP + BERT)

Repository Structure
├── extra_credit/
│   ├── extra_credit_multimodal.ipynb       # Extra credit notebook (CLIP + BERT)
│   └── weekly_files/
│       └── week7_to_11/                    # Weekly lab files (weeks 7–11)
│
├── main_file/
│   └── APPLIED_ML_revised.ipynb           # Main notebook (all 4 parts)
│
├── individual_parts/
│   ├── part1/
│   │   ├── part1_face_model.py             # Face model code
│   │   ├── part1_notebook.ipynb            # Part 1 notebook
│   │   └── datasets/                       # Part 1 dataset references
│   ├── part2/
│   │   ├── part2_robustness.ipynb          # Part 2 notebook
│   │   └── results_dataset/               # Degraded dataset outputs
│   ├── part3/
│   │   ├── part3_fullimage.py              # Full image model code
│   │   └── part3_notebook.ipynb            # Part 3 notebook
│   └── part4/
│       └── part4_fusion.ipynb              # Fusion model notebook
│
├── results/
│   └── screenshots/                        # Result screenshots (CM, ROC, reports)
│
├── README.md                               # This file
└── requirements.txt                        # Python dependencies

Dataset
The project uses the MSCTD (Multimodal Sentiment Chat Translation Dataset), specifically the En-De split.

The main notebook (Cell 3) automatically downloads all images and label files from the official GitHub repository and Google Drive using gdown and wget
No manual dataset download is required
Labels: 0 = Neutral, 1 = Negative, 2 = Positive


Note: Internet access is required when running the notebook for the first time. Once downloaded, data is cached at /content/msctd/data/ende/


Setup & Usage
Option 1: Google Colab (Recommended)

Open APPLIED_ML_revised.ipynb in Google Colab
Set runtime to GPU → Runtime → Change runtime type → T4 GPU
Run Cell 1 first to install all dependencies, then restart the runtime
Run all remaining cells sequentially from Cell 2 onwards

For the extra credit:

Open extra_credit_multimodal.ipynb in a separate Colab session
Set runtime to GPU
Run all cells — dataset downloads automatically

Option 2: Local Environment
bash# Clone the repository
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook main_file/APPLIED_ML_revised.ipynb

A CUDA-capable GPU is strongly recommended. CPU-only execution is possible but will be very slow.


Model Architectures
Part 1 — Face Sentiment Model

Backbone: EfficientNet-B2 (ImageNet pre-trained, last 3 blocks unfrozen)
Head: MLP (1408 → 512 → 128 → 3) with BatchNorm and Dropout
Face detector: MTCNN
Multi-face strategy: Confidence-weighted softmax averaging
No-face fallback: Training set class prior

Part 2 — Robust Face Model

Same architecture as Part 1, retrained on a 50/50 mix of clean and degraded images
Degradation pipeline: FFT frequency noise + Gaussian noise + JPEG compression + brightness/contrast jitter + random rotation (via Albumentations)

Part 3 — Full Image Model

Backbone: EfficientNet-B3 (ImageNet pre-trained, initially frozen)
Head: MLP (1536 → 512 → 128 → 3) with Dropout
Loss: Focal loss (γ=2) with class weights
Inference: Test-time augmentation (TTA × 3)

Part 4 — Attention Fusion Model

Input: 10-dim vector = face softmax (3) + full-image softmax (3) + face-count one-hot (4)
Fusion: Attention gate (sigmoid MLP mask) learns which stream to trust
Trained for 100 epochs with AdamW and focal loss

Extra Credit — Multimodal Transformer (CLIP + BERT)

Vision: CLIP ViT-B/32 — last 2 encoder layers unfrozen
Text: BERT-base-uncased — last 2 encoder layers unfrozen
Fusion: Cross-attention (image queries text) + residual LayerNorm → classifier
Training: 10 epochs, warmup + cosine decay, class-weighted CE, label smoothing (ε=0.1)


Evaluation Metrics
All models are evaluated on the test split using:

Accuracy
Macro F1-Score
Confusion Matrix
ROC Curves (per class, with AUC)


Results
Part 1 — Face Model (EfficientNet-B2)
Show Image
Show Image

Part 2 — Robustness (Clean vs Degraded)
Show Image

Part 3 — Full Image Model (EfficientNet-B3)
Show Image
Show Image

Part 4 — Fusion Model
Show Image
Show Image

Extra Credit — Multimodal Transformer (CLIP + BERT)
Test Accuracy: 60.31% | Macro F1: 0.5865
ClassPrecisionRecallF1AUCNeutral0.490.450.470.730Negative0.710.640.670.797Positive0.560.680.620.788
Show Image

Saved Model Weights
All trained model weights are saved to /content/ during training on Colab:
FileDescriptionface_model.pthPart 1 — face sentiment modelface_model_robust.pthPart 2 — mixed-trained robust modelfull_image_model.pthPart 3 — EfficientNet-B3 modelfusion_model.pthPart 4 — attention fusion modelmultimodal_model.pthExtra credit — CLIP+BERT model

Download these from Colab via the Files panel before the session ends, or mount Google Drive to save automatically.


Dependencies
See requirements.txt for the full list. Core dependencies:

torch / torchvision (CUDA 12.1)
timm — EfficientNet backbones
transformers — CLIP and BERT (extra credit)
mtcnn — face detection
albumentations — image augmentation
scikit-learn — evaluation metrics
Pillow==10.4.0 — image I/O (pinned for compatibility)
gdown — Google Drive downloads


Reproducibility
All random seeds are fixed at the start of both notebooks:
pythontorch.manual_seed(42)
np.random.seed(42)
random.seed(42)

Group Members
NameURNLikhitha Reddy6956685Neha Madhava Sundhram6957792Pramodh Rahul Gullipalli6964218Harshitha—Malli—

References

MSCTD Dataset — https://github.com/XL2248/MSCTD
Multimodal Transformers Survey — https://arxiv.org/pdf/2206.06488.pdf
LXMERT — https://github.com/airsplay/lxmert
ClipBERT — https://github.com/jayleicn/ClipBERT


Developed for academic purposes as part of EEEM068 at the University of Surrey. Not for commercial use.
