# EEEM068 Applied Machine Learning — Weekly Progress Files

This folder contains the weekly progress documentation for the **Human Sentiment Analysis** project using the MSCTD En-De dataset.

## Contents

| File | Week Focus |
|------|-----------|
| `WEEK_7.docx` | Project setup, environment preparation, dataset download, path structure, and first data inspection |
| `WEEK_8.docx` | Data cleaning, face extraction, face dataset preparation, collate logic, and handling missing/multiple faces |
| `WEEK_9.docx` | Model architectures, training functions, hyper-parameter tuning, and baseline face/full-image training |
| `WEEK_10.docx` | Evaluation pipeline, degraded-image robustness experiments, confusion matrices, ROC/AUC, and classification reports |
| `week_11.docx` | Fusion model, final testing, report discussion, README/docs, and team contribution summary |

## Project Overview

**Module:** EEEM068 Applied Machine Learning  
**Task:** Human Sentiment Analysis using the MSCTD English-German (En-De) dataset  
**Framework:** PyTorch / Google Colab  

The pipeline covers:
- Face-based sentiment classification (EfficientNet-B4 + MTCNN)
- Degraded-image robustness testing (FFT noise + albumentations)
- Full-image transfer learning (EfficientNet-B4)
- Attention-gated fusion model combining face and full-image predictions

## Team Members

| Member | Primary Responsibility |
|--------|----------------------|
| Malli | Setup, imports, transforms, data download, Colab environment |
| Neha | Dataset class, face extraction pipeline, MSCTDFaceDataset, collate function |
| Likhitha | Model architectures and training functions (FaceSentimentModel, FullImageModel, FusionModel) |
| Pramodh | Evaluation pipeline — confusion matrix, ROC/AUC, classification report, final metrics |
| Harshitha | Debug cells, visualisation helpers, main execution, README, and report support |

## How to Use

The main implementation is in `New_main_FINAL_v2.ipynb`. Open it in Google Colab and run cells in order:

1. **Cell 1** — Install packages (restart runtime after)
2. **Cell 2** — Imports and global config
3. **Cell 3** — Download MSCTD En-De dataset (use Cell 3b if Cell 3 fails)
4. **Cells 4–6** — Dataset class, transforms, loaders, class distribution plot
5. **Cell 7** — MTCNN face extractor
6. **Cells 8–10** — Model architectures, loss functions, training/evaluation functions
7. **Cell 11** — Train face model (Part 1)
8. **Cells 13–15** — Degraded dataset and robustness evaluation (Part 2)
9. **Cell 16** — Train full image model (Part 3)
10. **Cells 17–19** — Fusion feature collection and fusion model training/evaluation (Part 4)
11. **Cells 21–22** — Final summary table and save all model weights
