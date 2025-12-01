# 🤖 Introduction to AI Mini-Project Report  
## Domain Adaptation with DANN: MNIST to MNIST-M

---

## 📋 Project Information

- **Course:** MAT3508 – Introduction to Artificial Intelligence  
- **Term:** Semester 1 – Academic Year 2025–2026  
- **Institution:** VNU-HUS (Vietnam National University, Hanoi – University of Science)  
- **Project Title:** Domain Adaptation with DANN: MNIST to MNIST-M  
- **Submission Date:** 30/11/2025  
- **PDF Report:** 📄 [`dann-m2m.pdf`](https://github.com/kimpro12/VNU-HUS-IntroAI-MiniProject/blob/master/PDF%20Report/dann-m2m.pdf)  
- **Presentation Slides:** 🖥️ [`DANN.pptx`](https://github.com/kimpro12/VNU-HUS-IntroAI-MiniProject/blob/master/Slides/DANN.pptx)  
- **GitHub Repository:** https://github.com/kimpro12/VNU-HUS-IntroAI-MiniProject  

---

## 👥 Team Members

| 👤 Name        | 🆔 Student ID | 🐙 GitHub Username | 🛠️ Contribution |
|---------------|---------------|--------------------|------------------|
| Bui Ngoc Kim  | 25000252      | `kimpro12`         | Full             |

---

## 📝 Project Summary

This project studies **unsupervised domain adaptation** from **MNIST** (grayscale digits) to **MNIST-M** (digits with colored natural-image backgrounds) using **Domain-Adversarial Neural Networks (DANN)**. The goal is to train a classifier that performs well on MNIST-M using **labeled MNIST** and **unlabeled MNIST-M** only.  

DANN combines:
- A shared **feature extractor**,
- A **label classifier** (for digit prediction on MNIST),
- A **domain classifier** trained through a **Gradient Reversal Layer (GRL)** to encourage **domain-invariant features**.  

Training is performed in two phases:
1. **Source-only baseline:** standard supervised training on MNIST.  
2. **Full DANN training:** joint training with labeled MNIST and unlabeled MNIST-M using the adversarial domain loss and a **sigmoid λ schedule**.

---

## ⚙️ Methods & Implementation (Overview)

Key components (see the PDF report for full details):

- **Model architecture**
  - Small CNN feature extractor for 32×32 RGB digits.
  - Label classifier: MLP head with BatchNorm, ReLU, Dropout, LogSoftmax (10 classes).
  - Domain classifier: MLP head with BatchNorm, ReLU, LogSoftmax (2 domains).

- **Gradient Reversal Layer (GRL)**
  - Forward: identity.
  - Backward: multiplies gradient by **−λ**, implementing the adversarial min–max objective.
  - 
- **Lambda schedule**
  - Curriculum-like schedule  

    $$\lambda(p) = \frac{2}{1 + e^{-\gamma p}} - 1,\quad p \in [0,1]$$  

    where $p$ is the normalized training progress and $\gamma = 10$.

- **Losses**
  - Source classification: NLLLoss on label logits (log-softmax).  
  - Domain classification: NLLLoss with domain labels 0 (source) / 1 (target).  
  - Total:  

    $$L_{\text{total}} = L_y + \alpha L_d$$  

    where $\alpha$ is a domain loss weight.

- **t-SNE visualization**
  - 2D t-SNE on intermediate features (100-D).  
  - Plots generated **before** and **after** adaptation to visualize domain alignment.

- **Early stopping**
  - Monitors **source validation loss** with patience and warm-up epochs for both phases.

---

## 📊 Main Results (MNIST → MNIST-M)

| Model                     | MNIST Test Accuracy | MNIST-M Test Accuracy | Domain Accuracy |
|---------------------------|---------------------|------------------------|-----------------|
| Source-Only Baseline      | 99.46%              | 46.71%                 | –               |
| DANN (after adaptation)   | 98.16%              | 82.06%                 | 59.69%          |

**Highlights:**

- **Target accuracy** improves from **46.71% → 82.06%** (+35.35 percentage points).  
- **Source accuracy** drops only slightly (**99.46% → 98.16%**), showing a reasonable trade-off.  
- **Domain accuracy** ≈ 59.69% indicates **partial domain confusion**, consistent with more domain-invariant features.

---

## 💻 How to Run the Code

1. **Prepare MNIST-M dataset** in the following structure:

   ```text
   MNIST_M_ROOT/
     training/
       0/
       1/
       ...
     testing/
       0/
       1/
       ...
