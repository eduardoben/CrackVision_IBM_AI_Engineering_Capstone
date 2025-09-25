# CrackVision — IBM AI Engineering Mini‑Capstone

## Abstract
A PyTorch-based **binary image classifier** for crack detection implemented in a single Jupyter Notebook. The notebook covers dataset loading via a custom `Dataset`, `DataLoader`s for training/validation, transfer learning with **ResNet‑18 (pretrained)**, and a short training/evaluation routine.

---

## Introduction
This mini-capstone demonstrates an end‑to‑end supervised learning workflow for **crack detection** using convolutional neural networks in PyTorch. It shows how to adapt a pretrained backbone, define a classification head for two classes, and run a compact training loop with basic evaluation outputs and figures.

---

## Dataset
- Loaded through a **custom `Dataset`** class and consumed with PyTorch **`DataLoader`** objects for training and validation.
- Image handling includes conversion via `transforms.ToPILImage()` within a helper display function.
- Classification is **binary** (final fully connected layer dimensions indicate 2 classes).

> Exact dataset source, size, and class names are not specified in the notebook cells available here.

---

## Methodology
**Model**
- `torchvision.models.resnet18(pretrained=True)` with the final layer replaced:
  - `model.fc = nn.Linear(512, 2)`

**Loss & Optimizer**
- `nn.CrossEntropyLoss()`
- `torch.optim.Adam` with `lr = 0.001`

**Training**
- `n_epochs = 1` (as set in the notebook)
- Batch size observed: `100`
- Uses standard training loop (`for epoch in range(n_epochs):`) with `model.train()` and evaluation with `model.eval()` in `torch.no_grad()` context.

**Data pipeline**
- `DataLoader(dataset=train_dataset, batch_size=100)`
- `DataLoader(dataset=validation_dataset, batch_size=100)`

---

## Findings
Based on the executed cells and saved outputs available in the notebook:
- **Model/backbone:** ResNet‑18 (pretrained), with a binary head (`Linear(512, 2)`).
- **Objective:** Cross‑entropy classification with Adam (`lr=0.001`).
- **Runtime artifacts:** The notebook generates **5 figures** and prints logs; numeric **final metrics** (e.g., accuracy/precision/recall/AUC) are **not explicitly captured** in the saved outputs here.
- The code references common evaluation terms (e.g., *accuracy*, *loss*, *ROC*), but **no specific values** are stored in the visible outputs.

> If you re‑run the notebook end‑to‑end in your environment, additional metrics/plots may appear in the output cells.

---

## Repository Contents
- `Crack detection model.ipynb`

---

## Quickstart
1. Open a Python environment with PyTorch and torchvision installed.
2. Launch Jupyter Lab/Notebook and open **`Crack detection model.ipynb`**.
3. Execute cells sequentially (ensure your dataset paths and environment match the expectations inside the notebook).

---

## Environment (observed imports)
- `torch`, `torchvision`, `numpy`, `pandas`, `matplotlib`, `PIL`
- Utilities: `os`, `glob`, `time`, `h5py`

---

## Limitations / Notes
- The notebook demonstrates the training procedure with **1 epoch**; extend epochs and enable full logging to obtain stable metrics.
- Dataset details (source, size, class names) are not specified in the visible cells; ensure paths and labels are configured before running.
