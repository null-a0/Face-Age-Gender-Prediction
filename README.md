# Face Age & Gender Prediction

The task is to predict **age (regression)** and **gender (binary classification)** from facial images using deep learning.

---

## 🎯 Problem Overview

Given a face image, the model predicts:
- **Age**: Continuous value (0–100)
- **Gender**: Binary (0 or 1)

The competition evaluates performance using the **harmonic mean of**:
- Macro **F1-score** for gender
- Normalized **RMSE** for age

---

## 🧠 Models Implemented

### 1. CNN From Scratch
- Custom CNN built using PyTorch
- Multiple convolutional blocks with BatchNorm, ReLU, MaxPooling, and Dropout
- Shared backbone with two heads:
  - Gender classification head
  - Age prediction head

This model serves as the **baseline scratch implementation**.

---

### 2. Fine-Tuned EfficientNet-B4
- Pretrained EfficientNet-B4 backbone (ImageNet)
- Global Average Pooling
- Attention mechanism applied on extracted features
- Separate heads for:
  - Gender classification
  - Age prediction

Age is modeled as a **probability distribution over 101 bins (0–100)** and the final age is computed using the expected value.

This is the **primary model used for submission**.

---

## ⚙️ Training & Setup

- Framework: PyTorch Lightning
- Optimizer: AdamW
- Scheduler: Cosine Annealing
- Loss Functions:
  - Gender → Cross Entropy Loss
  - Age → KL Divergence Loss
- Image Size: 384 × 384
- Batch Size: 16

All experiments are logged using **TrackIO**.

⚠️ As per competition rules, **training code is commented out** in the submission notebook.

---

## 🔮 Inference & Submission

- Trained models are saved and uploaded to **kagglehub**
- The submission notebook:
  - Loads the saved model
  - Performs inference on the test set
  - Generates `submission.csv` with columns:

