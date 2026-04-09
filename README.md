# L42 — AutoEncoder Confusion: Latent Space Distortion Study

**Author:** Hadar Wayn  
**Course:** AI Developer Expert — Lesson 42  
**Instructor:** Dr. Yoram Segal  
**Framework:** Keras (TensorFlow backend)  
**GitHub:** [L42-AutoEncoder-Confusion-Keras-Studies](https://github.com/hadarwayn/L42-AutoEncoder-Confusion-Keras-Studies)

---

## What Is This Project?

This project proves a fascinating fact about neural networks: **they can only see the world they were trained on.**

We train AutoEncoders on one type of image (male faces, or sneakers), then force them to process a *similar but different* type (female faces, or ankle boots). The result? The network **hallucinates** — it tries to force unfamiliar images to look like what it knows.

---

## The Artist Analogy (For Everyone)

Imagine hiring an artist who spent their entire career painting **only men's portraits** — thousands of them. They mastered every detail: strong jawlines, facial hair, broad foreheads.

Now you hand them a photo of a **woman** and say: "Please redraw this."

The artist has **never learned** how to draw soft jawlines or delicate features. So they do their best — they get the general face shape right, but the details come out looking *mannish*. The eyes are too deep-set, the jaw is too square, the proportions feel off.

That's exactly what happens inside our AutoEncoder. The decoder can only paint what it was taught.

---

## The Two Experiments

### Experiment A — Gender Face Confusion

| | Detail |
|:--|:--|
| **Trained on** | Male faces only (2,350+ images from LFW dataset) |
| **Confused with** | 20 female face images |
| **What happens** | Female faces get reconstructed with male-like features: broader proportions, rougher textures |
| **Why it works** | Male and female faces share the same structure (eyes, nose, mouth) but differ in subtle details the decoder never learned |

### Experiment B — Fashion Topology Confusion

| | Detail |
|:--|:--|
| **Trained on** | Sneakers only (5,950 images from Fashion-MNIST) |
| **Confused with** | 20 ankle boot images |
| **What happens** | Boot shafts get "chopped off" — the decoder compresses tall boots into low sneaker profiles |
| **Why it works** | Sneakers and boots are both footwear, but boots have a tall shaft the decoder has never seen |

---

## How AutoEncoders Work

```
Original Image  →  [ENCODER]  →  Compressed Code  →  [DECODER]  →  Reconstructed Image
   (input)         (squeezes)     (bottleneck)        (rebuilds)      (output)
```

Think of it like a game of telephone:
1. **Encoder**: Takes a full image and compresses it into a tiny numerical code (like summarizing a whole book into a single tweet)
2. **Bottleneck**: The compressed code — forces the network to remember only the most important patterns
3. **Decoder**: Reads the code and tries to rebuild the original image — but it can only use patterns it learned during training

When the input is something the network was trained on, reconstruction works great. When the input is something new (out-of-distribution), the decoder *hallucinates* its training patterns onto the unfamiliar input.

### Our Architecture

We use a **deep Convolutional AutoEncoder** with these key design choices:

- **Double convolution blocks**: Each depth level has two Conv layers (not one), giving the network more capacity to learn fine details
- **BatchNormalization** after every convolutional layer — prevents the blurry, featureless outputs that plagued the reference projects
- **LeakyReLU activation** (slope 0.2) — prevents "dead neurons" that stop learning
- **3 depth levels** with filters [64, 128, 256] — progressively extracts more complex features
- **Bottleneck of 32 dimensions** — forces strong compression so the decoder must rely on learned patterns

---

## Results

### Experiment B — Fashion-MNIST (Sneakers vs Boots)

**Key Metrics:**
- In-distribution MSE: **0.0062** (sneakers reconstructed accurately)
- Out-of-distribution MSE: **0.0198** (boots reconstructed poorly)
- **Confusion ratio: 3.18x** — boots are 3.2 times harder for the network to reconstruct

#### Learning Curve
*The network successfully learned to reconstruct sneaker images. Both training and validation loss decrease smoothly.*

![Convergence Plot — Experiment B](results/experiment_b/convergence_plot.png)

#### The Network Works on Sneakers (In-Distribution)
*These are sneaker images the network was trained on — it reconstructs them accurately with very low error.*

![In-Distribution Reconstruction — Experiment B](results/experiment_b/in_distribution_reconstruction.png)

#### The Confusion Grid — All 20 Boot Images
*Each row shows: the original boot (left), the network's reconstruction (middle), and a heat map of the distortion (right). Notice how the tall boot shafts get compressed into sneaker-like profiles.*

![OoD Confusion Grid — Experiment B](results/experiment_b/ood_confusion_grid.png)

#### Top 5 Most Confused Images
*The five boots where the network struggled the most. High-heeled boots and tall shaft boots are the hardest — the decoder has no concept of "tall footwear."*

![Distortion Highlights — Experiment B](results/experiment_b/distortion_highlight.png)

#### Error Distribution — Statistical Proof
*Blue = sneaker reconstruction errors (low). Red = boot reconstruction errors (higher). The clear separation proves the network struggles with unfamiliar input.*

![Error Distribution — Experiment B](results/experiment_b/error_distribution.png)

#### Latent Space PCA — Geometric Proof
*Blue dots = sneakers in the compressed space. Red stars = boots. The boots land in a completely different region — the decoder doesn't know how to handle inputs from that area.*

![Latent Space PCA — Experiment B](results/experiment_b/latent_space_pca.png)

---

### Experiment A — Human Faces (Male-Trained vs Female-Tested)

**Key Metrics:**
- In-distribution MSE: **0.0049** (male faces reconstructed accurately)
- Out-of-distribution MSE: **0.0070** (female faces reconstructed with distortion)
- **Confusion ratio: 1.43x** — female faces produce measurably higher reconstruction error

#### Learning Curve
*The network learned to reconstruct male face images. Loss decreases smoothly over 50 epochs.*

![Convergence Plot — Experiment A](results/experiment_a/convergence_plot.png)

#### The Network Works on Male Faces (In-Distribution)
*Male faces are reconstructed accurately — these are the images the network was trained on.*

![In-Distribution Reconstruction — Experiment A](results/experiment_a/in_distribution_reconstruction.png)

#### The Confusion Grid — All 20 Female Face Images
*Each row shows: original female face (left), the network's reconstruction (middle), and distortion heat map (right). The decoder imposes male-learned features onto female faces.*

![OoD Confusion Grid — Experiment A](results/experiment_a/ood_confusion_grid.png)

#### Top 5 Most Confused Images
*The five female faces that confused the network the most. Notice how distinct female features (hair styles, face shapes) get replaced with more generic, male-like reconstructions.*

![Distortion Highlights — Experiment A](results/experiment_a/distortion_highlight.png)

#### Error Distribution — Statistical Proof
*Blue = male face errors (low). Red = female face errors (shifted right). The OoD mean is measurably higher.*

![Error Distribution — Experiment A](results/experiment_a/error_distribution.png)

#### Latent Space PCA — Geometric Proof
*Red stars (female faces) show a different distribution pattern compared to blue dots (male faces) in the compressed latent space.*

![Latent Space PCA — Experiment A](results/experiment_a/latent_space_pca.png)

---

## Applications & Real-World Use Cases

1. **Medical Imaging / Anomaly Detection**: Train an AutoEncoder on healthy X-rays. When a diseased X-ray is processed, the high reconstruction error flags it as abnormal — the same principle used in this project.

2. **Manufacturing Quality Control**: Train on images of good products. Defective products produce higher reconstruction error, automatically catching defects on the production line.

3. **AI Bias Detection**: This project directly demonstrates how AI systems impose their training biases on new data. Understanding this is critical for building fair AI systems.

4. **Cybersecurity**: Train on normal network traffic patterns. Unusual (potentially malicious) traffic produces higher reconstruction error, flagging intrusions.

---

## Project Structure

| File | Description | Lines |
|:-----|:------------|:-----:|
| `main.py` | Entry point — runs both experiments | 77 |
| `src/autoencoder.py` | Deep CNN AutoEncoder with double-conv blocks & BatchNorm | 119 |
| `src/trainer.py` | Training loop, checkpointing, hardware detection | 138 |
| `src/confusion.py` | OoD inference and error computation | 82 |
| `src/visualizer.py` | 5 visualization functions (convergence, grid, etc.) | 150 |
| `src/visualizer_latent.py` | PCA latent space visualization | 96 |
| `src/utils/data_loader.py` | Fashion-MNIST + LFW face dataset loading | 131 |
| `src/utils/paths.py` | Relative path utilities | 49 |
| `src/utils/logger.py` | Ring buffer logging system | 116 |
| `experiments/experiment_a_faces.py` | Experiment A runner | 71 |
| `experiments/experiment_b_fashion.py` | Experiment B runner | 71 |

All Python files are under 150 lines.

---

## Installation & Running

### Prerequisites

- Python 3.10–3.13
- [UV](https://docs.astral.sh/uv/) package manager

### Option A: WSL (Recommended)

```bash
# Navigate to project
cd /mnt/c/Users/YourName/Projects/L42

# Create virtual environment with compatible Python
uv venv --python 3.12

# Activate
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt

# Run both experiments
python main.py

# Or run just one experiment
python main.py --experiment b    # Fashion-MNIST only
python main.py --experiment a    # Faces only
```

### Option B: Windows PowerShell

```powershell
# Navigate to project
cd C:\Users\YourName\Projects\L42

# Create virtual environment
uv venv --python 3.12

# Activate
.venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt

# Run
python main.py
```

### Option C: Google Colab (No Setup Needed)

Open the notebook directly in Google Colab — everything runs in the cloud:

`notebooks/L42_AutoEncoder_Colab.ipynb`

Just click "Runtime > Run all" and watch the results appear.

---

## Command-Line Options

```bash
python main.py                    # Run both experiments (default)
python main.py --experiment a     # Run only face experiment
python main.py --experiment b     # Run only fashion experiment
python main.py --epochs 50        # Train for 50 epochs
python main.py --batch-size 64    # Override batch size
```

---

## What I Learned

### Technical Insights

1. **BatchNormalization is essential** — Without BatchNorm after every convolutional layer, the network produces blurry, featureless gray blobs. This was the main failure in the reference projects.

2. **Latent dimension controls the confusion effect** — Too small = everything is blurry. Too large = the decoder has enough capacity to reconstruct OoD images correctly (no confusion). 32 dimensions hits the sweet spot for visible hallucination.

3. **Reconstruction error is a powerful anomaly detector** — The 3.18x MSE ratio in Experiment B proves that OoD images are *measurably* harder to reconstruct. This same principle powers anomaly detection in medical imaging, cybersecurity, and manufacturing.

### Real-World Connections

1. **AI Bias is Real and Measurable** — Our male-trained face model literally cannot see the world through a female lens. This is the same mechanism behind AI bias in facial recognition, hiring algorithms, and medical diagnosis.

2. **Generative AI Foundations** — The latent space we explored is the same concept behind Stable Diffusion, DALL-E, and other image generators. Understanding AutoEncoders is the first step toward understanding modern generative AI.

3. **The "Unknown Unknowns" Problem** — Our AutoEncoder doesn't know what it doesn't know. It confidently reconstructs boots as sneakers without any error signal. This is a fundamental challenge in deploying AI in safety-critical applications.

---

## GitHub Upload

```bash
git init
git add .
git commit -m "L42: AutoEncoder Confusion — Latent Space Distortion Study"
git remote add origin https://github.com/hadarwayn/L42-AutoEncoder-Confusion-Keras-Studies.git
git push -u origin main
```

---

## Training Summary

| Metric | Experiment B (Fashion) | Experiment A (Faces) |
|:-------|:----------------------:|:--------------------:|
| Training images | 5,950 sneakers | 2,350+ male faces |
| Test images (OoD) | 20 ankle boots | 20 female faces |
| Training epochs | 50 | 50 |
| Training time (CPU) | ~6,630s | ~4,427s |
| Final validation loss | 0.1801 (BCE) | 0.0054 (MSE) |
| In-distribution MSE | 0.0062 | 0.0049 |
| OoD MSE | 0.0198 | 0.0070 |
| Confusion ratio | **3.18x** | **1.43x** |
| Filters | [64, 128, 256] | [64, 128, 256] |
| Latent dimension | 32 | 32 |

---

*Built by Hadar Wayn for the AI Developer Expert course (Lesson 42) — April 2026*
