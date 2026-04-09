"""PCA Latent Space Visualization — shows where OoD images land in the latent map."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional

from sklearn.decomposition import PCA
from keras import Model

from src.utils.logger import setup_logger

logger = setup_logger("visualizer_latent")

DPI = 300


def plot_latent_space_pca(encoder: Model, x_in_dist: np.ndarray,
                          x_ood: np.ndarray, experiment_name: str,
                          output_path: Path,
                          max_in_dist: int = 500) -> None:
    """Encode in-distribution + OoD images, project to 2D with PCA, and plot.

    Blue dots = in-distribution (training domain)
    Red stars = OoD (confused domain)

    Args:
        encoder: Encoder sub-model (input=image, output=latent vector)
        x_in_dist: In-distribution images (subset of training data)
        x_ood: OoD test images (20 images)
        experiment_name: For plot title
        output_path: Where to save the PNG
        max_in_dist: Max in-distribution samples to encode (for speed)
    """
    # Subsample in-distribution if too many
    if len(x_in_dist) > max_in_dist:
        rng = np.random.RandomState(42)
        indices = rng.choice(len(x_in_dist), size=max_in_dist, replace=False)
        x_in_dist = x_in_dist[indices]

    # Encode both sets
    latent_in = encoder.predict(x_in_dist, verbose=0)
    latent_ood = encoder.predict(x_ood, verbose=0)

    # Combine and apply PCA
    all_latent = np.concatenate([latent_in, latent_ood], axis=0)
    pca = PCA(n_components=2, random_state=42)
    all_2d = pca.fit_transform(all_latent)

    in_2d = all_2d[:len(latent_in)]
    ood_2d = all_2d[len(latent_in):]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.scatter(in_2d[:, 0], in_2d[:, 1], c="blue", alpha=0.3, s=20,
               label=f"In-Distribution ({len(in_2d)} samples)")
    ax.scatter(ood_2d[:, 0], ood_2d[:, 1], c="red", marker="*", s=150,
               edgecolors="darkred", linewidths=0.5,
               label=f"OoD — Confused ({len(ood_2d)} samples)", zorder=5)

    # Draw convex hulls if enough points
    _draw_hull(ax, in_2d, color="blue", alpha=0.08)
    _draw_hull(ax, ood_2d, color="red", alpha=0.15)

    ax.set_xlabel(f"PCA Component 1 ({pca.explained_variance_ratio_[0]:.1%} variance)",
                  fontsize=11)
    ax.set_ylabel(f"PCA Component 2 ({pca.explained_variance_ratio_[1]:.1%} variance)",
                  fontsize=11)
    ax.set_title(f"Latent Space: Where OoD Images Land — {experiment_name}",
                 fontsize=14, fontweight="bold")
    fig.text(0.5, 0.01,
             "Red stars = OoD inputs forced into the wrong region of the latent space",
             ha="center", fontsize=10, style="italic", color="gray")
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.savefig(str(output_path), dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved latent space PCA plot: {output_path}")


def _draw_hull(ax: plt.Axes, points: np.ndarray, color: str,
               alpha: float = 0.1) -> None:
    """Draw a convex hull around a set of 2D points."""
    if len(points) < 3:
        return
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(points)
        hull_points = np.append(hull.vertices, hull.vertices[0])
        ax.fill(points[hull_points, 0], points[hull_points, 1],
                color=color, alpha=alpha)
    except Exception:
        pass  # Skip hull if scipy unavailable or points are collinear
