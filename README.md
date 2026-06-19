
# PRISMA Fest Image Classifier — CNN
### Image Classification using Convolutional Neural Networks
**PRISMA Fest Scenario**


---

## Objective

This project implements a Convolutional Neural Network (CNN) to automatically classify PRISMA Fest event photographs into 4 categories:

> **comedy · fashion · gaming · music**

---

## Dataset

Images were collected from PRISMA Fest event photographs and organized into 4 categories.

**Folder structure (before preparation):**
```
raw_dataset/
    comedy/       ← comedy/stand-up event photos
    fashion/      ← fashion show photos
    gaming/       ← gaming event photos
    music/        ← music/stage performance photos
```

- Total images used: ~30–100 per category
- Train / Test split: **80% train, 20% test**
- Image format accepted: `.jpg`, `.jpeg`, `.png`

---

## CNN Architecture

Built layer-by-layer from scratch. No deep learning libraries used.

```
Input Image (32 × 32 Grayscale)
        │
┌───────▼──────────────────────────────┐
│  ConvLayer  (8 filters, 3×3) + ReLU  │  ← detects edges & low-level features
│  MaxPoolLayer (2×2, stride 2)         │    32×32 → 16×16
└──────────────────────────────────────┘
        │
┌───────▼──────────────────────────────┐
│  ConvLayer  (16 filters, 3×3) + ReLU │  ← detects textures & patterns
│  MaxPoolLayer (2×2, stride 2)         │    16×16 → 8×8
└──────────────────────────────────────┘
        │
┌───────▼──────────────────────────────┐
│  FlattenLayer                         │    16 × 8 × 8 = 1024 values
│  DenseLayer (1024 → 128) + ReLU      │  ← high-level reasoning
│  DenseLayer (128 → 4)   + Softmax    │  ← one score per class
└──────────────────────────────────────┘
        │
Output: [comedy, fashion, gaming, music]
```

### Layers Explained

| Layer | Purpose |
|---|---|
| **ConvLayer** | Slides small filters (3×3) over the image to detect local features like edges, colours, and textures. Uses `im2col` for efficient matrix multiplication. |
| **ReLULayer** | Activation function: `f(x) = max(0, x)`. Adds non-linearity so the network can learn complex patterns. |
| **MaxPoolLayer** | Takes the maximum value in each 2×2 window. Reduces image size and makes detection robust to small shifts. |
| **FlattenLayer** | Converts 3D feature maps `(16, 8, 8)` into a 1D vector `(1024)` to connect to Dense layers. |
| **DenseLayer** | Fully connected layer — every neuron connects to every neuron in the next layer. Makes the final classification decision. |
| **Softmax + Cross-Entropy Loss** | Converts final scores into probabilities. Loss measures how wrong the prediction is and drives weight updates. |

---

## Project Files

```
prisma_cnn/
├── cnn_scratch.py     ← All CNN layers built from scratch (NumPy only)
├── train.py           ← Loads data, trains model, saves weights, plots curves
├── predict.py         ← Predicts category of any new image
├── dataset_prep.py    ← Splits raw images into train/test folders
└── README.md          ← This file
```

---

## How to Run

### Step 0 — Install requirements
Only 3 libraries needed :
```bash
pip install numpy pillow matplotlib
```

---

### Step 1 — Added images
Dataset Folder-[Dataset Folder Link](https://drive.google.com/drive/folders/1BAnPQh4LwebCNwwWFrz8eTmMutAfGuys?usp=sharing)
Created a `raw_dataset/` folder in the same directory as the `.py` files and added the  images:
```
raw_dataset/
    comedy/        ← add comedy photos here
    fashion/       ← add fashion photos here
    gaming/        ← add gaming photos here
    music/         ← add music photos here
```

---

### Step 2 — Prepare the dataset
```bash
python dataset_prep.py
```
**What it does:** Automatically splits images 80/20 into `dataset/train/` and `dataset/test/` folders.

Expected output:
```
===== Dataset Preparation Complete =====
Category        Train     Test    Total
------------------------------------------
comedy             40       10       50
fashion            40       10       50
gaming             40       10       50
music              40       10       50
```

---

### Step 3 — Train the CNN
```bash
python train.py
```
**What it does:** Builds the CNN, trains it for 25 epochs using mini-batch gradient descent, saves the model, and plots accuracy/loss graphs.

Expected output:
```
Epoch  1/25  |  Loss: 1.3821  |  Train Acc: 28.5%  |  Test Acc: 30.0%
Epoch  2/25  |  Loss: 1.2104  |  Train Acc: 45.0%  |  Test Acc: 42.5%
...
Epoch 25/25  |  Loss: 0.3210  |  Train Acc: 88.0%  |  Test Acc: 76.0%

Model saved → prisma_cnn.pkl
```

> **Note:** Training uses only NumPy (no GPU). Expected time: 3–5 minutes on a normal laptop.

---

### Step 4 — Predict a new image
```bash
python predict.py path/to/image.jpg
```

Example:
```bash
python predict.py test_photo.jpg
```

Expected output:
```
===== Prediction Result =====
Image      : test_photo.jpg
Prediction : FASHION
Confidence : 84.32%

All class probabilities:
  comedy           5.10%  ██
  fashion         84.32%  █████████████████████████████████
  gaming           7.40%  ███
  music            3.18%  █
```

---

## Training Hyperparameters

| Parameter | Value |
|---|---|
| Input image size | 32 × 32 (grayscale) |
| Batch size | 16 |
| Epochs | 25 |
| Learning rate | 0.01 (decays by 0.95 each epoch) |
| Optimiser | Mini-batch Gradient Descent |
| Loss function | Softmax + Cross-Entropy |
| Weight init | He Initialization |

---

## Libraries Used

| Library | Purpose |
|---|---|
| `numpy` | All CNN math — convolution, backpropagation, matrix operations |
| `Pillow` | Loading and resizing images |
| `matplotlib` | Plotting training curves and prediction output |


---

## Submission Contents

- `cnn_scratch.py` — Full CNN implementation from scratch
- `train.py` — Training script
- `predict.py` — Prediction script
- `dataset_prep.py` — Dataset preparation script
- `README.md` — This file
- Dataset link / images folder

---
