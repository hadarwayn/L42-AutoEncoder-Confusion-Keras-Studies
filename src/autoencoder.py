"""Convolutional AutoEncoder with BatchNorm — configurable for faces and Fashion-MNIST."""
import numpy as np
from typing import Dict, Any

import keras
from keras import layers, Model

from src.utils.logger import setup_logger

logger = setup_logger("autoencoder")

DEFAULT_CONFIG_FACES = {
    "input_shape": (64, 64, 3),
    "filters": [64, 128, 256],
    "latent_dim": 32,
    "loss": "mse",
    "learning_rate": 0.001,
}

DEFAULT_CONFIG_FASHION = {
    "input_shape": (28, 28, 1),
    "filters": [64, 128, 256],
    "latent_dim": 32,
    "loss": "binary_crossentropy",
    "learning_rate": 0.001,
}


def _enc_block(x, filters: int, idx: int):
    """Encoder block: Conv -> BN -> LeakyReLU -> Conv -> BN -> LeakyReLU -> MaxPool."""
    x = layers.Conv2D(filters, (3, 3), padding="same", name=f"enc_conv_{idx}a")(x)
    x = layers.BatchNormalization(name=f"enc_bn_{idx}a")(x)
    x = layers.LeakyReLU(0.2, name=f"enc_lrelu_{idx}a")(x)
    x = layers.Conv2D(filters, (3, 3), padding="same", name=f"enc_conv_{idx}b")(x)
    x = layers.BatchNormalization(name=f"enc_bn_{idx}b")(x)
    x = layers.LeakyReLU(0.2, name=f"enc_lrelu_{idx}b")(x)
    x = layers.MaxPooling2D((2, 2), name=f"enc_pool_{idx}")(x)
    return x


def _dec_block(x, filters: int, idx: int):
    """Decoder block: UpSample -> ConvT -> BN -> LeakyReLU -> ConvT -> BN -> LeakyReLU."""
    x = layers.UpSampling2D((2, 2), name=f"dec_up_{idx}")(x)
    x = layers.Conv2DTranspose(filters, (3, 3), padding="same", name=f"dec_conv_{idx}a")(x)
    x = layers.BatchNormalization(name=f"dec_bn_{idx}a")(x)
    x = layers.LeakyReLU(0.2, name=f"dec_lrelu_{idx}a")(x)
    x = layers.Conv2DTranspose(filters, (3, 3), padding="same", name=f"dec_conv_{idx}b")(x)
    x = layers.BatchNormalization(name=f"dec_bn_{idx}b")(x)
    x = layers.LeakyReLU(0.2, name=f"dec_lrelu_{idx}b")(x)
    return x


def build_autoencoder(config: Dict[str, Any]) -> Dict[str, Model]:
    """Build a deep convolutional autoencoder with double-conv blocks and BatchNorm.

    Args:
        config: Dictionary with input_shape, filters, latent_dim, loss, learning_rate

    Returns:
        Dict with keys 'autoencoder', 'encoder', 'decoder'
    """
    input_shape = tuple(config["input_shape"])
    filters = config["filters"]
    latent_dim = config["latent_dim"]

    # --- Encoder ---
    encoder_input = layers.Input(shape=input_shape, name="encoder_input")
    x = encoder_input
    for i, f in enumerate(filters):
        x = _enc_block(x, f, i)

    shape_before_flatten = x.shape[1:]
    x = layers.Flatten(name="enc_flatten")(x)
    latent = layers.Dense(latent_dim, name="latent_vector")(x)
    encoder = Model(encoder_input, latent, name="encoder")

    # --- Decoder ---
    flat_dim = int(np.prod(shape_before_flatten))
    decoder_input = layers.Input(shape=(latent_dim,), name="decoder_input")
    x = layers.Dense(flat_dim, name="dec_dense")(decoder_input)
    x = layers.Reshape(shape_before_flatten, name="dec_reshape")(x)
    for i, f in enumerate(reversed(filters)):
        x = _dec_block(x, f, i)

    x = layers.Conv2DTranspose(
        input_shape[-1], (3, 3), padding="same", activation="sigmoid", name="dec_output"
    )(x)
    x = layers.Resizing(input_shape[0], input_shape[1], name="dec_resize")(x)
    decoder = Model(decoder_input, x, name="decoder")

    # --- Full AutoEncoder ---
    encoded = encoder(encoder_input)
    decoded = decoder(encoded)
    autoencoder = Model(encoder_input, decoded, name="autoencoder")
    autoencoder.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config["learning_rate"]),
        loss=config["loss"],
    )

    total_params = autoencoder.count_params()
    logger.info(
        f"AutoEncoder built — Input: {input_shape} | Latent: {latent_dim} | "
        f"Filters: {filters} | Params: {total_params:,} | Loss: {config['loss']}"
    )
    return {"autoencoder": autoencoder, "encoder": encoder, "decoder": decoder}


def validate_architecture(config: Dict[str, Any]) -> bool:
    """Validate that a dummy input passes through the autoencoder without errors."""
    models = build_autoencoder(config)
    dummy = np.zeros((1, *config["input_shape"]), dtype=np.float32)
    encoded = models["encoder"].predict(dummy, verbose=0)
    assert encoded.shape == (1, config["latent_dim"])
    decoded = models["decoder"].predict(encoded, verbose=0)
    assert decoded.shape == (1, *config["input_shape"])
    full = models["autoencoder"].predict(dummy, verbose=0)
    assert full.shape == (1, *config["input_shape"])
    logger.info(f"Architecture validated — Input/Output: {(1, *config['input_shape'])}")
    return True
