"""Dataset loading — Fashion-MNIST and Human Faces with verification."""
import numpy as np
from pathlib import Path
from typing import Tuple
from src.utils.paths import get_data_dir
from src.utils.logger import setup_logger

logger = setup_logger("data_loader")
SEED = 42


def load_fashion_mnist() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load Fashion-MNIST: sneakers for training, boots for OoD test."""
    from keras.datasets import fashion_mnist
    from sklearn.model_selection import train_test_split

    (x_tr, y_tr), (x_te, y_te) = fashion_mnist.load_data()
    x_all = np.concatenate([x_tr, x_te], axis=0)
    y_all = np.concatenate([y_tr, y_te], axis=0)

    sneakers = (x_all[y_all == 7].astype(np.float32) / 255.0)[..., np.newaxis]
    boots = (x_all[y_all == 9].astype(np.float32) / 255.0)[..., np.newaxis]

    x_train, x_val = train_test_split(sneakers, test_size=0.15, random_state=SEED)
    x_test = boots[np.random.RandomState(SEED).choice(len(boots), 20, replace=False)]

    logger.info(f"Fashion-MNIST — Train: {len(x_train)} sneakers | "
                f"Val: {len(x_val)} | Test: 20 boots")
    return x_train, x_val, x_test


def load_human_faces() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load face dataset: males for training, females for OoD test.

    Tries local data/ folder, then Kaggle, then LFW fallback (auto-download).
    """
    from sklearn.model_selection import train_test_split
    data_dir = get_data_dir()

    try:
        male_images, female_images = _load_local_or_kaggle(data_dir)
    except (FileNotFoundError, Exception) as e:
        logger.info(f"Local/Kaggle unavailable ({e}). Using LFW fallback.")
        male_images, female_images = _load_lfw_faces()

    x_train, x_val = train_test_split(male_images, test_size=0.15, random_state=SEED)
    rng = np.random.RandomState(SEED)
    x_test = female_images[rng.choice(len(female_images), 20, replace=False)]

    logger.info(f"Faces — Train: {len(x_train)} males | "
                f"Val: {len(x_val)} | Test: 20 females")
    return x_train, x_val, x_test


def _load_local_or_kaggle(data_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Try local directory or Kaggle download for face images."""
    for pat in [("male", "female"), ("Male", "Female"), ("males", "females")]:
        m_dir, f_dir = data_dir / pat[0], data_dir / pat[1]
        if m_dir.exists() and f_dir.exists():
            return _load_images_from_dir(m_dir, (64, 64)), _load_images_from_dir(f_dir, (64, 64))

    if data_dir.exists():
        for sub in data_dir.iterdir():
            if sub.is_dir():
                for pat in [("male", "female"), ("Male", "Female")]:
                    m, f = sub / pat[0], sub / pat[1]
                    if m.exists() and f.exists():
                        return _load_images_from_dir(m, (64, 64)), _load_images_from_dir(f, (64, 64))

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi(); api.authenticate()
        api.dataset_download_files("ashwingupta3012/human-faces", path=str(data_dir), unzip=True)
        return _load_local_or_kaggle(data_dir)
    except BaseException:
        pass
    raise FileNotFoundError("No local face data and Kaggle unavailable")


def _load_lfw_faces() -> Tuple[np.ndarray, np.ndarray]:
    """Load LFW dataset, split by gender using known public figure names."""
    from sklearn.datasets import fetch_lfw_people
    logger.info("Downloading LFW face dataset (sklearn)...")
    lfw = fetch_lfw_people(min_faces_per_person=20, resize=1.0)

    female_names = {
        "Amelie Mauresmo", "Angelina Jolie", "Gloria Macapagal Arroyo",
        "Jennifer Aniston", "Jennifer Capriati", "Jennifer Lopez",
        "Laura Bush", "Lindsay Davenport", "Megawati Sukarnoputri",
        "Naomi Watts", "Serena Williams", "Winona Ryder",
    }
    female_idx, male_idx = [], []
    for i, t in enumerate(lfw.target):
        (female_idx if lfw.target_names[t] in female_names else male_idx).append(i)

    logger.info(f"LFW split — Males: {len(male_idx)} | Females: {len(female_idx)}")
    return _resize_lfw_batch(lfw.images[male_idx]), _resize_lfw_batch(lfw.images[female_idx])


def _resize_lfw_batch(images: np.ndarray, size: Tuple[int, int] = (64, 64)) -> np.ndarray:
    """Resize LFW grayscale images to 64x64 RGB, normalized [0, 1]."""
    from PIL import Image
    result = []
    for img in images:
        pil = Image.fromarray((img * 255).astype(np.uint8), mode="L")
        result.append(np.array(pil.convert("RGB").resize(size, Image.LANCZOS)))
    return np.stack(result).astype(np.float32) / 255.0


def _load_images_from_dir(img_dir: Path, target_size: Tuple[int, int]) -> np.ndarray:
    """Load all images from directory, resize to target_size RGB, normalize [0, 1]."""
    from PIL import Image
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in extensions)
    if not paths:
        raise FileNotFoundError(f"No images in {img_dir}")
    images = [np.array(Image.open(p).convert("RGB").resize(target_size, Image.LANCZOS)) for p in paths]
    result = np.stack(images).astype(np.float32) / 255.0
    logger.info(f"Loaded {len(result)} images from {img_dir.name}")
    return result


def verify_data(x_train: np.ndarray, x_val: np.ndarray,
                x_test: np.ndarray, name: str) -> None:
    """Verify dataset integrity: shapes, value range, sample counts."""
    assert x_train.ndim == 4 and x_val.ndim == 4, "Data must be 4D"
    assert x_test.shape[0] == 20, f"Test must have 20 samples, got {x_test.shape[0]}"
    for label, arr in [("train", x_train), ("val", x_val), ("test", x_test)]:
        assert 0.0 <= arr.min() and arr.max() <= 1.0, f"{label} pixels not in [0,1]"
    logger.info(f"[{name}] Verified — Train: {x_train.shape} | Val: {x_val.shape} | "
                f"Test: {x_test.shape}")
