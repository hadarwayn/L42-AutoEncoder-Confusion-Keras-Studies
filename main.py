"""L42 AutoEncoder Confusion — Main entry point for both experiments."""
import argparse
import sys
import time

from src.utils.logger import setup_logger

logger = setup_logger("main")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="L42 AutoEncoder Confusion: Latent Space Distortion Study"
    )
    parser.add_argument(
        "--experiment", type=str, choices=["a", "b", "both"], default="both",
        help="Which experiment to run: 'a' (faces), 'b' (fashion), or 'both' (default)"
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Number of training epochs (default: 50)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Batch size (auto-detected from hardware if not specified)"
    )
    return parser.parse_args()


def main() -> None:
    """Run the selected experiment(s) end-to-end."""
    args = parse_args()
    start = time.time()

    logger.info("=" * 70)
    logger.info("L42 — AutoEncoder Confusion: Latent Space Distortion Study")
    logger.info("Author: Hadar Wayn | Course: AI Developer Expert — Lesson 42")
    logger.info("=" * 70)

    results = {}

    if args.experiment in ("b", "both"):
        logger.info("\n>>> Starting Experiment B — Fashion-MNIST <<<\n")
        from experiments.experiment_b_fashion import run as run_b
        results["experiment_b"] = run_b(epochs=args.epochs, batch_size=args.batch_size)

    if args.experiment in ("a", "both"):
        logger.info("\n>>> Starting Experiment A — Human Faces <<<\n")
        from experiments.experiment_a_faces import run as run_a
        results["experiment_a"] = run_a(epochs=args.epochs, batch_size=args.batch_size)

    elapsed = time.time() - start

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)

    for name, res in results.items():
        comp = res["comparison"]
        hist = res["history"]
        logger.info(
            f"  {name}: "
            f"val_loss={hist['val_loss'][-1]:.6f} | "
            f"in_dist_MSE={comp['in_dist_mean_mse']:.6f} | "
            f"ood_MSE={comp['ood_mean_mse']:.6f} | "
            f"ratio={comp['ratio']:.2f}x | "
            f"train_time={hist['training_time_sec']:.1f}s"
        )

    logger.info(f"\nTotal time: {elapsed:.1f}s")
    logger.info("All results saved to results/ directory")


if __name__ == "__main__":
    main()
