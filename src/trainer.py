"""Training pipeline with checkpointing, early stopping, and hardware detection."""
import time
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import keras
from keras import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from src.utils.logger import setup_logger
from src.utils.paths import get_weights_dir

logger = setup_logger("trainer")


def detect_hardware() -> Dict[str, Any]:
    """Detect available hardware and recommend batch size.

    Returns:
        Dict with device, gpu_name, vram_gb, recommended_batch_size
    """
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            details = tf.config.experimental.get_device_details(gpus[0])
            gpu_name = details.get("device_name", "Unknown GPU")
        except Exception:
            gpu_name = "GPU detected"

        # Recommend batch size based on GPU type
        batch_size = 64
        for high_end in ["A100", "V100", "P100"]:
            if high_end in gpu_name:
                batch_size = 128
                break

        info = {"device": "GPU", "gpu_name": gpu_name,
                "recommended_batch_size": batch_size}
    else:
        info = {"device": "CPU", "gpu_name": "N/A",
                "recommended_batch_size": 32}

    logger.info(f"Hardware: {info['device']} ({info['gpu_name']}) | "
                f"Recommended batch size: {info['recommended_batch_size']}")
    return info


def train_autoencoder(autoencoder: Model, x_train: np.ndarray,
                      x_val: np.ndarray, config: Dict[str, Any]) -> Dict[str, Any]:
    """Train the autoencoder with checkpointing and early stopping.

    AutoEncoder training target = input (reconstruct itself).

    Args:
        autoencoder: Compiled Keras autoencoder model
        x_train: Training images (normalized [0, 1])
        x_val: Validation images (normalized [0, 1])
        config: Dict with experiment, epochs, batch_size

    Returns:
        Dict with train_loss, val_loss, epochs_run, training_time_sec
    """
    experiment = config["experiment"]
    epochs = config.get("epochs", 30)
    batch_size = config.get("batch_size", 32)

    weights_dir = get_weights_dir(experiment)
    best_weights_path = weights_dir / "best_weights.keras"

    callbacks = [
        ModelCheckpoint(
            str(best_weights_path), monitor="val_loss",
            save_best_only=True, verbose=1
        ),
        EarlyStopping(
            monitor="val_loss", patience=10,
            restore_best_weights=True, verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=5, min_lr=1e-6, verbose=1
        ),
    ]

    logger.info(f"Training [{experiment}] — {epochs} epochs, batch_size={batch_size}, "
                f"train={len(x_train)}, val={len(x_val)}")

    start_time = time.time()
    history = autoencoder.fit(
        x_train, x_train,
        validation_data=(x_val, x_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )
    training_time = time.time() - start_time

    result = {
        "train_loss": history.history["loss"],
        "val_loss": history.history["val_loss"],
        "epochs_run": len(history.history["loss"]),
        "training_time_sec": training_time,
    }

    logger.info(
        f"Training complete — Final val_loss: {result['val_loss'][-1]:.6f} | "
        f"Epochs: {result['epochs_run']} | Time: {training_time:.1f}s"
    )
    return result


def save_models(models: Dict[str, Model], experiment: str) -> None:
    """Save autoencoder and encoder sub-model to disk."""
    weights_dir = get_weights_dir(experiment)

    autoencoder_path = weights_dir / "autoencoder_final.keras"
    encoder_path = weights_dir / "encoder_final.keras"

    models["autoencoder"].save(str(autoencoder_path))
    models["encoder"].save(str(encoder_path))

    logger.info(f"Models saved — {autoencoder_path} | {encoder_path}")


def load_trained_autoencoder(experiment: str) -> Optional[Model]:
    """Load a previously trained autoencoder from disk."""
    weights_dir = get_weights_dir(experiment)
    path = weights_dir / "autoencoder_final.keras"
    if path.exists():
        model = keras.models.load_model(str(path))
        logger.info(f"Loaded trained model from {path}")
        return model
    logger.warning(f"No trained model found at {path}")
    return None
