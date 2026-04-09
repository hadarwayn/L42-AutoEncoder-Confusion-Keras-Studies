"""Visualization functions — convergence, reconstruction, confusion grid, error distribution."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from src.utils.logger import setup_logger

logger = setup_logger("visualizer")

DPI = 300


def _save_and_close(fig: plt.Figure, path: Path) -> None:
    """Save figure at 300 DPI and close it."""
    fig.savefig(str(path), dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved: {path}")


def plot_convergence(history: dict, experiment_name: str, output_path: Path) -> None:
    """Plot training vs validation loss over epochs."""
    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = range(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], "b-", label="Training Loss", linewidth=2)
    ax.plot(epochs, history["val_loss"], "r-", label="Validation Loss", linewidth=2)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title(f"AutoEncoder Learning Curve — {experiment_name}", fontsize=14)
    fig.text(0.5, 0.01, "Loss decreasing = network is learning to reconstruct images",
             ha="center", fontsize=10, style="italic", color="gray")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    _save_and_close(fig, output_path)


def plot_in_distribution(originals: np.ndarray, reconstructed: np.ndarray,
                         experiment_name: str, output_path: Path) -> None:
    """Plot 5 in-distribution pairs: original vs reconstructed."""
    n = min(5, len(originals))
    is_gray = originals.shape[-1] == 1
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6))
    fig.suptitle(f"Network Works Correctly on Training Domain — {experiment_name}",
                 fontsize=13, fontweight="bold")
    fig.text(0.5, 0.92, "These images are from the training domain — the network reconstructs them accurately",
             ha="center", fontsize=10, style="italic", color="gray")

    for i in range(n):
        img_o = originals[i].squeeze() if is_gray else originals[i]
        img_r = reconstructed[i].squeeze() if is_gray else reconstructed[i]
        cmap = "gray" if is_gray else None
        axes[0, i].imshow(img_o, cmap=cmap, vmin=0, vmax=1)
        axes[0, i].set_title("Original", fontsize=9); axes[0, i].axis("off")
        axes[1, i].imshow(img_r, cmap=cmap, vmin=0, vmax=1)
        mse = float(np.mean((originals[i] - reconstructed[i]) ** 2))
        axes[1, i].set_title(f"Reconstructed\nMSE: {mse:.4f}", fontsize=9)
        axes[1, i].axis("off")

    fig.tight_layout(rect=[0, 0.02, 1, 0.90])
    _save_and_close(fig, output_path)


def plot_confusion_grid(originals: np.ndarray, reconstructed: np.ndarray,
                        diff_maps: np.ndarray, mse_list: np.ndarray,
                        experiment_name: str, output_path: Path) -> None:
    """Plot 20-row confusion grid: Original | Reconstructed | Difference Map."""
    n = len(originals)
    is_gray = originals.shape[-1] == 1
    fig, axes = plt.subplots(n, 3, figsize=(12, 2.5 * n))
    fig.suptitle(f"The Confusion Effect — {experiment_name}",
                 fontsize=14, fontweight="bold", y=1.0)

    col_titles = ["Original Input (OoD)", "Reconstructed Output", "Distortion Map"]
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=11, fontweight="bold")

    for i in range(n):
        img_o = originals[i].squeeze() if is_gray else originals[i]
        img_r = reconstructed[i].squeeze() if is_gray else reconstructed[i]
        diff = diff_maps[i].squeeze() if is_gray else np.mean(diff_maps[i], axis=-1)
        cmap = "gray" if is_gray else None
        axes[i, 0].imshow(img_o, cmap=cmap, vmin=0, vmax=1)
        axes[i, 0].set_ylabel(f"#{i+1}", fontsize=9, rotation=0, labelpad=25)
        axes[i, 0].set_xticks([]); axes[i, 0].set_yticks([])
        axes[i, 1].imshow(img_r, cmap=cmap, vmin=0, vmax=1)
        if i > 0: axes[i, 1].set_title(f"MSE: {mse_list[i]:.4f}", fontsize=8)
        axes[i, 1].axis("off")
        axes[i, 2].imshow(diff, cmap="hot", vmin=0, vmax=diff.max() + 1e-8)
        axes[i, 2].axis("off")

    fig.tight_layout()
    _save_and_close(fig, output_path)


def plot_distortion_highlights(originals: np.ndarray, reconstructed: np.ndarray,
                               mse_list: np.ndarray, experiment_name: str,
                               output_path: Path) -> None:
    """Plot top 5 most distorted images with annotations."""
    top5_idx = np.argsort(mse_list)[-5:][::-1]
    is_gray = originals.shape[-1] == 1
    fig, axes = plt.subplots(5, 3, figsize=(12, 14))
    fig.suptitle(f"Top 5 Most Confused Reconstructions — {experiment_name}",
                 fontsize=14, fontweight="bold")

    for row, idx in enumerate(top5_idx):
        img_o = originals[idx].squeeze() if is_gray else originals[idx]
        img_r = reconstructed[idx].squeeze() if is_gray else reconstructed[idx]
        diff = np.abs(originals[idx] - reconstructed[idx])
        diff_show = diff.squeeze() if is_gray else np.mean(diff, axis=-1)
        cmap = "gray" if is_gray else None

        axes[row, 0].imshow(img_o, cmap=cmap, vmin=0, vmax=1)
        axes[row, 0].set_title("Original" if row == 0 else "", fontsize=10)
        axes[row, 0].set_ylabel(f"MSE: {mse_list[idx]:.4f}", fontsize=9)
        axes[row, 0].set_xticks([]); axes[row, 0].set_yticks([])

        axes[row, 1].imshow(img_r, cmap=cmap, vmin=0, vmax=1)
        axes[row, 1].set_title("Reconstructed" if row == 0 else "", fontsize=10)
        axes[row, 1].axis("off")

        axes[row, 2].imshow(diff_show, cmap="hot")
        axes[row, 2].set_title("Distortion Map" if row == 0 else "", fontsize=10)
        axes[row, 2].axis("off")

    fig.tight_layout()
    _save_and_close(fig, output_path)


def plot_error_distribution(in_dist_mse: np.ndarray, ood_mse: np.ndarray,
                            experiment_name: str, output_path: Path) -> None:
    """Plot overlapping histograms of in-distribution vs OoD reconstruction MSE."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(in_dist_mse, bins=15, alpha=0.6, color="blue",
            label=f"In-Distribution (mean={np.mean(in_dist_mse):.4f})")
    ax.hist(ood_mse, bins=15, alpha=0.6, color="red",
            label=f"OoD / Confused (mean={np.mean(ood_mse):.4f})")

    ax.axvline(np.mean(in_dist_mse), color="blue", linestyle="--", linewidth=2)
    ax.axvline(np.mean(ood_mse), color="red", linestyle="--", linewidth=2)

    ax.set_xlabel("Reconstruction MSE", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"Reconstruction Error: Normal vs Confused — {experiment_name}",
                 fontsize=14)
    fig.text(0.5, 0.01, "OoD images have higher error — the network struggles with unfamiliar input",
             ha="center", fontsize=10, style="italic", color="gray")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    _save_and_close(fig, output_path)
