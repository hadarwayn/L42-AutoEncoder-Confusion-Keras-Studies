"""Experiment B — Fashion-MNIST: Train on sneakers, confuse with ankle boots."""
from typing import Dict, Any

from src.autoencoder import build_autoencoder, DEFAULT_CONFIG_FASHION
from src.trainer import train_autoencoder, save_models, detect_hardware
from src.confusion import run_confusion, run_in_distribution, compare_distributions
from src.visualizer import (plot_convergence, plot_in_distribution,
                            plot_confusion_grid, plot_distortion_highlights,
                            plot_error_distribution)
from src.visualizer_latent import plot_latent_space_pca
from src.utils.data_loader import load_fashion_mnist, verify_data
from src.utils.paths import get_results_dir
from src.utils.logger import setup_logger

logger = setup_logger("experiment_b")

EXPERIMENT = "experiment_b"
NAME = "Fashion-MNIST (Sneakers vs Boots)"


def run(epochs: int = 30, batch_size: int = None) -> Dict[str, Any]:
    """Run the full Experiment B pipeline: load, train, confuse, visualize."""
    results_dir = get_results_dir(EXPERIMENT)
    hw = detect_hardware()
    if batch_size is None:
        batch_size = hw["recommended_batch_size"]

    # 1. Load data
    logger.info("=" * 60)
    logger.info("EXPERIMENT B — Fashion-MNIST: Sneakers vs Ankle Boots")
    logger.info("=" * 60)
    x_train, x_val, x_test = load_fashion_mnist()
    verify_data(x_train, x_val, x_test, NAME)

    # 2. Build model
    models = build_autoencoder(DEFAULT_CONFIG_FASHION)

    # 3. Train
    train_config = {"experiment": EXPERIMENT, "epochs": epochs, "batch_size": batch_size}
    history = train_autoencoder(models["autoencoder"], x_train, x_val, train_config)

    # 4. Save models
    save_models(models, EXPERIMENT)

    # 5. Confusion phase
    ood_result = run_confusion(models["autoencoder"], x_test)
    in_dist_result = run_in_distribution(models["autoencoder"], x_val)
    comparison = compare_distributions(in_dist_result, ood_result)

    # 6. Visualizations
    plot_convergence(history, NAME, results_dir / "convergence_plot.png")
    plot_in_distribution(
        in_dist_result["originals"], in_dist_result["reconstructed"],
        NAME, results_dir / "in_distribution_reconstruction.png")
    plot_confusion_grid(
        ood_result["originals"], ood_result["reconstructed"],
        ood_result["diff_maps"], ood_result["mse_per_image"],
        NAME, results_dir / "ood_confusion_grid.png")
    plot_distortion_highlights(
        ood_result["originals"], ood_result["reconstructed"],
        ood_result["mse_per_image"], NAME,
        results_dir / "distortion_highlight.png")
    plot_error_distribution(
        in_dist_result["mse_per_image"], ood_result["mse_per_image"],
        NAME, results_dir / "error_distribution.png")
    plot_latent_space_pca(
        models["encoder"], x_val, x_test,
        NAME, results_dir / "latent_space_pca.png")

    logger.info(f"Experiment B complete — all results in {results_dir}")
    return {"history": history, "comparison": comparison, "hardware": hw}
