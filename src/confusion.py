"""OoD confusion inference — feed out-of-distribution images through trained autoencoder."""
import numpy as np
from typing import Dict
from keras import Model

from src.utils.logger import setup_logger

logger = setup_logger("confusion")


def run_confusion(autoencoder: Model, x_ood: np.ndarray) -> Dict[str, np.ndarray]:
    """Feed OoD images through the autoencoder and measure reconstruction error.

    Args:
        autoencoder: Trained autoencoder model
        x_ood: OoD test images, shape (N, H, W, C), normalized [0, 1]

    Returns:
        Dict with originals, reconstructed, diff_maps, mse_per_image, mean_mse
    """
    reconstructed = autoencoder.predict(x_ood, verbose=0)

    # Clip to [0, 1] for safety
    reconstructed = np.clip(reconstructed, 0.0, 1.0)

    # Difference maps and per-image MSE (pure NumPy, no loops)
    diff_maps = np.abs(x_ood - reconstructed)
    mse_per_image = np.mean((x_ood - reconstructed) ** 2, axis=(1, 2, 3))
    mean_mse = float(np.mean(mse_per_image))

    logger.info(f"OoD confusion — {len(x_ood)} images | "
                f"Mean MSE: {mean_mse:.6f} | "
                f"Min: {mse_per_image.min():.6f} | Max: {mse_per_image.max():.6f}")

    return {
        "originals": x_ood,
        "reconstructed": reconstructed,
        "diff_maps": diff_maps,
        "mse_per_image": mse_per_image,
        "mean_mse": mean_mse,
    }


def run_in_distribution(autoencoder: Model,
                        x_in_dist: np.ndarray,
                        n_samples: int = 20) -> Dict[str, np.ndarray]:
    """Run in-distribution images through the autoencoder for baseline comparison.

    Args:
        autoencoder: Trained autoencoder model
        x_in_dist: In-distribution images (from training/val set)
        n_samples: Number of samples to use for comparison

    Returns:
        Same structure as run_confusion()
    """
    rng = np.random.RandomState(42)
    indices = rng.choice(len(x_in_dist), size=min(n_samples, len(x_in_dist)),
                         replace=False)
    subset = x_in_dist[indices]

    result = run_confusion(autoencoder, subset)
    logger.info(f"In-distribution baseline — Mean MSE: {result['mean_mse']:.6f}")
    return result


def compare_distributions(in_dist_result: Dict[str, np.ndarray],
                          ood_result: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Compare in-distribution vs OoD reconstruction errors.

    Returns:
        Dict with in_dist_mean_mse, ood_mean_mse, ratio
    """
    in_mse = float(in_dist_result["mean_mse"])
    ood_mse = float(ood_result["mean_mse"])
    ratio = ood_mse / in_mse if in_mse > 0 else float("inf")

    logger.info(
        f"Confusion comparison — In-dist MSE: {in_mse:.6f} | "
        f"OoD MSE: {ood_mse:.6f} | Ratio: {ratio:.2f}x"
    )
    return {"in_dist_mean_mse": in_mse, "ood_mean_mse": ood_mse, "ratio": ratio}
