# PRD.md — Product Requirements Document
# L42: AutoEncoder Confusion — Latent Space Distortion Study

**Version:** 1.0  
**Author:** Hadar Wayn  
**Course:** AI Developer Expert — Lesson 42  
**Instructor:** Dr. Yoram Segal  
**Framework:** Keras (TensorFlow backend)  
**GitHub:** https://github.com/hadarwayn/L42-AutoEncoder-Confusion-Keras-Studies.git  
**Last Updated:** April 2026

---

## 1. Project Overview

### 1.1 Project Name
**L42 — AutoEncoder Confusion: Latent Space Distortion Study**

### 1.2 One-Line Description
Train Keras AutoEncoders on one visual domain, then feed images from a *similar but different* domain and observe the fascinating "hallucination" distortions the decoder produces — proving that neural networks only know what they were taught.

### 1.3 Problem Statement
When an AutoEncoder is trained on one group of images (e.g., male faces or sneakers), its decoder learns **only** how to reconstruct that specific world. When we force it to process images from a *related but different* world (female faces or ankle boots), the decoder hallucinates — it projects its training biases onto the unfamiliar input.

Two reference projects (L42 classmates) produced results that were **too blurry and unclear** to demonstrate the hallucination effect. This project fixes that by:
- Using **pre-trained / well-structured CNN architectures** instead of small shallow networks
- Using **sufficient training images** and proper hyperparameters that show gradual learning curves
- Delivering **clear, annotated visual comparisons** that anyone can understand

### 1.4 Why This Matters — Real-World Impact
- **AI Bias Detection:** Understanding how trained networks impose their learned biases on unseen data
- **Anomaly Detection:** Autoencoders trained on "normal" data can detect disease or defects by measuring what they *cannot* reconstruct
- **Generative AI foundations:** Latent space manipulation is the basis of VAEs, GANs, and modern image generation
- **Explainable AI:** Showing *where* and *why* a network fails builds trust in AI systems

---

## 2. The Two Experiments

### Experiment A — Gender Topology Confusion (Human Faces)
| Property | Detail |
|----------|--------|
| **Training Domain** | Male faces only |
| **Test Input** | 20 female face images |
| **Dataset** | Human Faces Dataset (Kaggle: hosseinbadrnezhad/human-faces-dataset-male-female-classification) |
| **Expected Distortion** | Female faces reconstructed with: squared jawlines, hints of facial hair artifacts, broader skull proportions |
| **Why Interesting** | Human faces are structurally identical (eyes/nose/mouth placement) yet morphologically different — perfect for visible hallucination |

### Experiment B — Fashion Topology Confusion (Fashion-MNIST)
| Property | Detail |
|----------|--------|
| **Training Domain** | Sneakers only (class 7 in Fashion-MNIST) |
| **Test Input** | 20 ankle boot images (class 9 in Fashion-MNIST) |
| **Dataset** | Fashion-MNIST (built into Keras — no download needed) |
| **Expected Distortion** | Ankle boots have their upper shaft "chopped off" — decoder compresses them into low sneaker profile |
| **Why Interesting** | Same pixel dimensions (28×28), same grayscale format — only structure differs. Shows pure structural hallucination |

---

## 3. Target Users & Applications

### 3.1 Primary Users
- AI Developer Expert course students (demonstrating L42 assignment)
- Potential employers reviewing portfolio
- AI researchers studying out-of-distribution (OoD) behavior

### 3.2 Use Cases
1. **Portfolio Showcase:** Clear visual proof of understanding AutoEncoders, latent spaces, and OoD behavior
2. **Educational Demo:** A teacher can show this to students: "this is what happens when AI encounters something it hasn't seen"
3. **Research Foundation:** The code pattern (train→confuse→visualize) can be reused for any OoD study
4. **Anomaly Detection Prototype:** The gender experiment's reconstruction error map can flag "unusual" faces in surveillance or medical imaging

---

## 4. Functional Requirements

### FR-1: Dual Experiment Support
- FR-1.1: Implement Experiment A (Gender Faces) as a standalone, runnable pipeline
- FR-1.2: Implement Experiment B (Fashion-MNIST Sneakers vs Boots) as a standalone, runnable pipeline
- FR-1.3: Both experiments share the same AutoEncoder architecture template (configurable depth/filters)

### FR-2: Dataset Handling
- FR-2.1: Experiment A must download/load the Human Faces Dataset from Kaggle OR use a fallback face dataset (CelebA subset, UTKFace, or Flickr-Faces-HQ thumbnails) if Kaggle API is unavailable
- FR-2.2: Experiment A must automatically separate images into male-only training set and female-only test set using filename/folder labels
- FR-2.3: Experiment B must load Fashion-MNIST directly from `keras.datasets` — zero manual download
- FR-2.4: Exactly **20 test images** must be used in each experiment for the "confusion" phase
- FR-2.5: Test images must be selected randomly (with a fixed seed for reproducibility)

### FR-3: AutoEncoder Architecture
- FR-3.1: Use a Convolutional AutoEncoder (CAE) matching the architecture shown in class:  
  `Input → Enc1(512) → Enc2(256) → Enc3(128) → Bottleneck(z) → Dec1(128) → Dec2(256) → Dec3(512) → Output`
- FR-3.2: Architecture must be configurable (image size, number of filters, bottleneck dimension) via a single config dictionary
- FR-3.3: Encoder and Decoder must be separately accessible as sub-models (for latent space extraction)
- FR-3.4: Loss function: Binary Cross-Entropy (for Fashion-MNIST) and MSE (for face images)
- FR-3.5: **BatchNormalization after every Conv2D and Conv2DTranspose layer** — this is mandatory to prevent blurry outputs and accelerate convergence. Without BatchNorm, the network under-trains and produces the muddy gray blur seen in the reference projects.
- FR-3.6: Latent (bottleneck) dimension: **128** (faces) and **64** (Fashion-MNIST). These values were validated to produce visible hallucination effects. Too small = too much information lost; too large = decoder reconstructs OoD images too well (no confusion effect).
- FR-3.7: Encoder path: `Conv2D → BatchNorm → ReLU → MaxPooling2D` (repeated per depth level), then `Flatten → Dense(latent_dim)` to create the latent vector
- FR-3.8: Decoder path: `Dense(latent_dim) → Reshape → UpSampling2D → Conv2DTranspose → BatchNorm → ReLU` (repeated), final layer uses Sigmoid activation

### FR-4: Training
- FR-4.1: Train **only on the in-distribution domain** (males only / sneakers only)
- FR-4.2: Minimum 30 epochs (local), 50 epochs (Colab with GPU)
- FR-4.3: Save best model weights (checkpoint on validation loss)
- FR-4.4: Record training/validation loss history for convergence plot
- FR-4.5: Google Colab: detect GPU at startup with `!nvidia-smi`, and dynamically set batch size (32 for CPU/T4, 128 for A100/V100)

### FR-5: Confusion Phase (OoD Testing)
- FR-5.1: Load the trained model
- FR-5.2: Feed 20 out-of-distribution test images through the full AutoEncoder
- FR-5.3: Capture: original input, reconstructed output, and difference map (|original − reconstructed|)
- FR-5.4: Compute per-image reconstruction error (MSE)

### FR-6: Visualizations (MANDATORY — must be clear and explainable)
- FR-6.1: **Convergence Plot** — training loss vs validation loss over epochs (shows the network learned)
- FR-6.2: **In-Distribution Reconstruction** — 5 training-domain images + their reconstructions (shows network works correctly)
- FR-6.3: **OoD Confusion Grid** — 20 test images arranged as: [Original | Reconstructed | Difference Map] for each image
- FR-6.4: **Distortion Highlight** — top 5 most-distorted images with arrows and annotations explaining *what changed*
- FR-6.5: **Error Distribution Plot** — histogram of reconstruction MSE for in-distribution vs OoD images
- FR-6.6: **Latent Space PCA Plot** *(Optional but impressive)* — encode all training images + 20 OoD test images through the encoder, then reduce to 2D with PCA. Plot in-distribution points (blue) vs OoD points (red). Shows visually that OoD inputs land in the "wrong" region of the latent space — this is the geometric proof of why confusion happens.
- FR-6.7: All plots saved as high-quality PNG (300 DPI) in `results/` directory
- FR-6.8: All plots must include: title, axis labels, legend, and a plain-English subtitle explaining what to observe

### FR-7: Dual Environment — LOCAL FIRST, then Google Colab

#### ⚠️ MANDATORY EXECUTION ORDER — This is not optional

```
STEP 1 ─── Run locally (UV virtual environment)
              │
              ├── Validates that all code works correctly
              ├── Generates all result images (PNG files)
              ├── Reveals actual training times on this machine
              └── Produces the "blueprint" the Colab notebook will mirror
              │
              ▼
STEP 2 ─── Build Google Colab Notebook
              │
              ├── The notebook is built AFTER local runs succeed
              ├── It is a faithful mirror of the local pipeline
              ├── Every local result image is matched by a Colab inline display
              └── The notebook shows the same results — with GPU speed bonus
```

**Why local first?**
Running locally first is not just a technical preference — it is how a professional learns. When you run the code on your own machine, you see every print statement, every epoch, every error message at your own pace. You understand what is happening before handing it to Colab's cloud environment. The Colab notebook is then the *clean, polished presentation* of what you already proved works.

- FR-7.1: **LOCAL** — Full Python project using UV virtual environment (WSL or PowerShell)
  - Entry point: `python main.py` (or `python main.py --experiment b`)
  - All results saved to `results/` folder on local disk
  - Must run to completion before the Colab notebook is created

- FR-7.2: **COLAB** — Complete Google Colab notebook (`.ipynb`), created only after local success
  - The notebook structure mirrors the local pipeline phase by phase
  - Cell 1 (Code): GPU detection — `!nvidia-smi`, hardware report, recommended batch size
  - Cell 2 (Code): Package installation — `!pip install` all required packages
  - Cell 3 (Code): Google Drive mount — saves results to `/content/drive/MyDrive/L42_results/`
  - Cells 4–N: Each pipeline phase gets its own clearly titled code cell
  - **Every code cell is preceded by a Markdown cell** explaining in plain English:
    - What this cell does
    - Why this step exists
    - What you should see in the output
  - All 6 visualizations displayed inline with `plt.show()` — no need to download files
  - Final cell: summary comparison of local vs Colab training times

- FR-7.3: The Colab notebook must be **self-contained** — a new user can open it on Google Colab and run all cells top-to-bottom with zero local setup required

- FR-7.4: The notebook's Markdown cells use the **same plain-English analogies** as the README — a student reading the notebook learns the same concepts as a student reading the README

---

## 5. Technical Requirements

### 5.1 Environment
| Requirement | Specification |
|-------------|--------------|
| Python | 3.10+ |
| Virtual Environment | UV (MANDATORY for local) |
| Primary Framework | Keras 3.x (TensorFlow 2.x backend) |
| GPU Support | Optional locally, auto-detected on Colab |
| OS | Windows (WSL), Linux, macOS, Google Colab |

### 5.2 Core Libraries
| Library | Version | Purpose |
|---------|---------|---------|
| tensorflow | ≥2.13.0 | Keras backend |
| keras | ≥3.0.0 | AutoEncoder construction |
| numpy | ≥1.24.0 | Array operations (NO raw Python loops) |
| matplotlib | ≥3.7.0 | All visualizations |
| scikit-learn | ≥1.3.0 | Train/test split, metrics |
| pillow | ≥10.0.0 | Image loading and resizing |
| kaggle | ≥1.6.0 | Dataset download (optional) |
| tqdm | ≥4.65.0 | Progress bars |

### 5.3 Performance Requirements
| Metric | Local (CPU) | Colab (GPU) |
|--------|-------------|-------------|
| Experiment B training time | < 10 min | < 3 min |
| Experiment A training time | < 30 min | < 8 min |
| Inference (20 images) | < 5 sec | < 1 sec |
| Peak RAM | < 4 GB | < 8 GB |

### 5.4 Code Quality
- **Maximum file length:** 150 lines per Python file
- **NumPy mandatory:** All array/image operations use NumPy vectorization
- **Type hints:** All functions
- **Docstrings:** All functions (explain WHY, not just WHAT)
- **No hardcoded paths:** Use `pathlib.Path(__file__).parent` everywhere

---

## 6. Success Criteria

### 6.1 "The Confusion Test" — Primary Success Metric
The project succeeds if a person with **no AI knowledge** looks at the OoD confusion grid and immediately says:
> *"Oh! The network tried to make that female face look like a man!"*  
> *"It cut off the top of the boot!"*

Both distortions must be **clearly visible** — not blurry noise.

### 6.2 Quantitative Success Criteria
- [ ] Reconstruction loss (in-distribution): < 0.05 MSE for faces, < 0.02 BCE for Fashion-MNIST
- [ ] OoD reconstruction loss: **measurably higher** than in-distribution (proves confusion occurred)
- [ ] Training convergence curve is smooth (no flat curves from day 1)
- [ ] All 20 test images processed and visualized

### 6.3 Quality Checklist
- [ ] All Python files ≤ 150 lines
- [ ] All visualizations have titles, labels, and plain-English explanations
- [ ] README includes embedded images of results
- [ ] Local project runs on fresh UV environment
- [ ] Colab notebook runs top-to-bottom without errors
- [ ] 15-year-old test: README explains AutoEncoders using an analogy anyone can understand

---

## 7. Constraints & Assumptions

### 7.1 Constraints
- No Darknet (requires local GPU + C compiler, not compatible with Colab)
- Fashion-MNIST experiment: images are 28×28 grayscale — small but sufficient for visible structural hallucination
- Face experiment: images resized to 64×64 RGB to balance quality with training speed
- Kaggle API key required for Human Faces Dataset (fallback: use subset of UTKFace or CelebA via direct URL)

### 7.2 Assumptions
- User has internet connection (for dataset download)
- Local machine: at least 8 GB RAM and 2 GB free disk
- Google account available for Colab
- UV is installed (instructions in README)

---

## 8. Directory Structure Plan

```
L42-AutoEncoder-Confusion/
├── README.md
├── main.py                          # Runs both experiments sequentially
├── requirements.txt
├── .gitignore
├── .env.example
│
├── venv/
│   └── .gitkeep
│
├── src/
│   ├── __init__.py
│   ├── autoencoder.py               # Model architecture (build_autoencoder)
│   ├── trainer.py                   # Training loop + checkpoint saving
│   ├── confusion.py                 # OoD inference + error computation
│   ├── visualizer.py                # All 6 visualization functions
│   └── utils/
│       ├── __init__.py
│       ├── data_loader.py           # Dataset loading (faces + Fashion-MNIST)
│       ├── paths.py                 # Relative path utilities
│       └── logger.py                # Ring buffer logging
│
├── experiments/
│   ├── experiment_a_faces.py        # Experiment A standalone runner
│   └── experiment_b_fashion.py      # Experiment B standalone runner
│
├── notebooks/
│   └── L42_AutoEncoder_Colab.ipynb  # Complete Google Colab notebook
│
├── docs/
│   ├── PRD.md                       # This document
│   └── tasks.json
│
├── results/
│   ├── experiment_a/
│   │   ├── model_weights/
│   │   ├── convergence_plot.png
│   │   ├── in_distribution_reconstruction.png
│   │   ├── ood_confusion_grid.png
│   │   ├── distortion_highlight.png
│   │   └── error_distribution.png
│   └── experiment_b/
│       ├── model_weights/
│       ├── convergence_plot.png
│       ├── in_distribution_reconstruction.png
│       ├── ood_confusion_grid.png
│       ├── distortion_highlight.png
│       └── error_distribution.png
│
└── logs/
    ├── config/
    │   └── log_config.json
    └── .gitkeep
```

---

## 9. Learning Objectives

This project demonstrates:
1. **AutoEncoder architecture** — encoder compresses, bottleneck encodes, decoder reconstructs
2. **Latent Space geometry** — the decoder only knows one "world"; foreign inputs get forced into it
3. **Out-of-Distribution (OoD) behavior** — networks hallucinate, they don't fail gracefully
4. **Reconstruction Loss as anomaly signal** — higher loss = more "foreign" the input is
5. **Keras model sub-models** — extracting encoder and decoder as independent models
6. **Google Colab best practices** — GPU detection, Drive mounting, batch size optimization

### The "Aha Moment" Analogy (for 15-year-olds)
> Imagine training an artist to paint only men's faces — thousands of them. Now you show the artist a woman's photo and ask them to redraw it. The artist has **never learned** how to draw smooth female jawlines or delicate features. So they do their best — they draw the general face shape correctly, but the details come out *mannish*. That's exactly what happens inside the AutoEncoder's decoder. It can only paint what it was taught.

---

## 10. Risks & Mitigation

These risks are **based on real failures** observed in the two reference L42 projects. Every mitigation here is a concrete technical fix, not a vague suggestion.

### Risk 1 — Blurry, Featureless Output (Most Likely Failure)
| | Detail |
|---|---|
| **Symptom** | Reconstructed images are smeared gray blobs — no visible structural features at all |
| **Root Cause** | Network under-trained OR architecture too shallow to learn meaningful features |
| **Mitigation A** | Add **BatchNormalization** after every Conv2D — stabilizes gradients, prevents dead neurons |
| **Mitigation B** | Increase **latent dimension**: too small (e.g., 8) forces lossy compression that cannot preserve structure |
| **Mitigation C** | Train for **more epochs** (minimum 30 locally, 50 on Colab with GPU) |
| **Mitigation D** | Verify **loss is actually decreasing** — if loss is flat from epoch 1, the model is not learning (check learning rate) |
| **Mitigation E** | Use **Adam optimizer** with lr=0.001 (not SGD — SGD requires careful tuning) |

### Risk 2 — No Visible Transformation (OoD Images Look Identical to Input)
| | Detail |
|---|---|
| **Symptom** | Reconstructed female faces look exactly like female faces — no masculinization |
| **Root Cause** | Domains are too different OR latent dimension is too large (decoder has enough capacity to "cheat") |
| **Mitigation A** | Confirm training set is **100% pure** — zero OoD contamination. Even 5% female faces in male training = no confusion effect |
| **Mitigation B** | Reduce latent dimension to force stronger compression bias |
| **Mitigation C** | Ensure **consistent normalization**: both training and OoD images must use the same pixel scale [0, 1] |
| **Mitigation D** | For Experiment B: verify label filtering is correct — Fashion-MNIST label 7=Sneaker, label 9=Ankle Boot |

### Risk 3 — Kaggle Dataset Download Fails
| | Detail |
|---|---|
| **Symptom** | `kaggle.api.dataset_download_files()` raises authentication error |
| **Root Cause** | No `kaggle.json` API key configured |
| **Mitigation** | Provide clear fallback instructions in README: manual download + place in `data/` folder. Code must detect and handle both paths gracefully |

### Risk 4 — Colab Session Runs Out of Memory
| | Detail |
|---|---|
| **Symptom** | CUDA out-of-memory error during training |
| **Root Cause** | Batch size too large for allocated GPU VRAM |
| **Mitigation** | GPU detection logic (FR-4.5) automatically adjusts batch size. If OOM still occurs: reduce image size from 64×64 to 32×32 for faces |

---

## 11. Extensions (Optional — for extra portfolio depth)

These are not required for the base assignment but each one significantly strengthens the portfolio value of the project.

| Extension | Description | Difficulty | Wow Factor |
|-----------|-------------|------------|------------|
| **Variational AutoEncoder (VAE)** | Replace standard AE with VAE — adds KL divergence loss, creates a continuous latent space. OoD confusion becomes smoother and more controllable | Medium | ⭐⭐⭐⭐ |
| **Latent Space Interpolation** | Take the latent vector of a male face and the latent vector of a female face, interpolate between them, decode each step → generates a morphing animation | Medium | ⭐⭐⭐⭐⭐ |
| **Cross-Domain Encoder-Decoder Swapping** | Train Encoder A on males, Encoder B on females. Feed female image through Encoder A, then decode with Decoder A → more controlled masculinization | Hard | ⭐⭐⭐⭐⭐ |
| **PCA Latent Space Visualization** | Already included as FR-6.6. Shows the geometric proof of why confusion happens | Easy | ⭐⭐⭐⭐ |
| **Reconstruction Error as Anomaly Detector** | For a medical imaging demo: train on healthy chest X-rays, run pneumonia X-rays through — the high reconstruction error flags anomalies | Medium | ⭐⭐⭐⭐⭐ |

---

## 12. GitHub Repository

**Author:** Hadar Wayn  
**Target repo:** https://github.com/hadarwayn/L42-AutoEncoder-Confusion-Keras-Studies.git

---

*End of PRD.md — Awaiting approval before implementation begins.*
