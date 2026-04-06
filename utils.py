import os
from typing import List, Tuple, Union

import numpy as np
from PIL import Image, UnidentifiedImageError


DEFAULT_CLASS_NAMES = ["Bellpepper", "carrot", "Cucumber", "potato", "tomato"]
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _one_hot(index: int, num_classes: int = 5) -> np.ndarray:
    vec = np.zeros(num_classes, dtype=np.float32)
    vec[index] = 1.0
    return vec


def _preprocess_image(image_path: str, image_size: Tuple[int, int] = (32, 32)) -> np.ndarray:
    """Load image, resize to 32x32 and normalize to [0, 1] in CHW format."""
    image = Image.open(image_path).convert("RGB")
    image = image.resize(image_size, Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return np.transpose(array, (2, 0, 1))


def load_data(path: str, class_names: Union[List[str], None] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load dataset from disk expecting exactly 5 class folders.

    Returns:
        X: np.ndarray with shape (N, 3, 32, 32), dtype float32
        y: np.ndarray with shape (N, C), one-hot labels, dtype float32
        class_names: ordered list of classes used to encode labels
    """
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Dataset path not found: {path}")

    if class_names is None:
        detected = [d for d in sorted(os.listdir(path)) if os.path.isdir(os.path.join(path, d))]
        class_names = detected if detected else DEFAULT_CLASS_NAMES
    else:
        class_names = list(class_names)

    if len(class_names) != 5:
        raise ValueError(
            f"Expected exactly 5 class folders, got {len(class_names)}: {class_names}"
        )

    missing = [name for name in class_names if not os.path.isdir(os.path.join(path, name))]
    if missing:
        raise ValueError(
            "Missing required class folders: "
            + ", ".join(missing)
            + f". Expected folders are: {', '.join(class_names)}"
        )

    class_to_index = {name: idx for idx, name in enumerate(class_names)}

    X_data: List[np.ndarray] = []
    y_data: List[np.ndarray] = []
    corrupted_count = 0

    for class_name in class_names:
        class_idx = class_to_index[class_name]
        class_dir = os.path.join(path, class_name)

        for file_name in os.listdir(class_dir):
            file_path = os.path.join(class_dir, file_name)

            if not os.path.isfile(file_path):
                continue

            ext = os.path.splitext(file_name)[1].lower()
            if ext not in VALID_EXTENSIONS:
                continue

            try:
                image = _preprocess_image(file_path, image_size=(32, 32))
                X_data.append(image)
                y_data.append(_one_hot(class_idx, num_classes=len(class_names)))
            except (UnidentifiedImageError, OSError, ValueError) as exc:
                corrupted_count += 1
                print(f"[WARN] Skipping corrupted/invalid image: {file_path} ({exc})")

    if not X_data:
        raise ValueError("No valid images found after preprocessing.")

    X = np.array(X_data, dtype=np.float32)
    y = np.array(y_data, dtype=np.float32)

    if corrupted_count > 0:
        print(f"[INFO] Corrupted/invalid images skipped: {corrupted_count}")

    return X, y, class_names


def train_test_split_manual(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: Union[int, None] = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Shuffle and split without sklearn."""
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must contain the same number of samples.")

    if not (0.0 < test_size < 1.0):
        raise ValueError("test_size must be a float between 0 and 1.")

    n_samples = X.shape[0]
    rng = np.random.default_rng(random_state)
    indices = np.arange(n_samples)
    rng.shuffle(indices)

    split_idx = int(n_samples * (1.0 - test_size))
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]

    X_train = X[train_idx]
    y_train = y[train_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]

    return X_train, X_test, y_train, y_test


def get_label_name(index: int) -> str:
    class_names = DEFAULT_CLASS_NAMES
    if not isinstance(index, (int, np.integer)):
        raise TypeError("index must be an integer.")

    if index < 0 or index >= len(CLASS_NAMES):
        raise ValueError(f"index out of range [0, {len(CLASS_NAMES) - 1}].")

    return CLASS_NAMES[int(index)]


def predict_with_threshold(
    probabilities: np.ndarray,
    threshold: float = 0.5,
    class_names: Union[List[str], None] = None,
) -> Union[str, List[str]]:
    """
    Apply confidence threshold to softmax outputs.

    If max probability < threshold -> 'Ninguna de las anteriores'.
    Else -> class name among avion/barco/bicicleta/carro/moto.
    """
    if not (0.0 <= threshold <= 1.0):
        raise ValueError("threshold must be between 0 and 1.")

    probs = np.asarray(probabilities, dtype=np.float32)
    class_names = DEFAULT_CLASS_NAMES if class_names is None else list(class_names)

    if probs.ndim == 1:
        if probs.shape[0] != len(class_names):
            raise ValueError(f"Expected vector of length {len(class_names)}.")
        max_idx = int(np.argmax(probs))
        max_prob = float(probs[max_idx])
        if max_prob < threshold:
            return "Ninguna de las anteriores"
        return class_names[max_idx]

    if probs.ndim == 2:
        if probs.shape[1] != len(class_names):
            raise ValueError(f"Expected matrix with {len(class_names)} columns.")

        results: List[str] = []
        for row in probs:
            max_idx = int(np.argmax(row))
            max_prob = float(row[max_idx])
            if max_prob < threshold:
                results.append("Ninguna de las anteriores")
            else:
                results.append(class_names[max_idx])
        return results

    raise ValueError("probabilities must be a 1D or 2D array.")
