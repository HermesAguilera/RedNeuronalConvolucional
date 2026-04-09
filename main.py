import os

import numpy as np
from PIL import Image

from activations import ReLU, Softmax
from layers import ConvolutionalLayer, PoolingLayer, DenseLayer
from model import CNN


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
CONFIDENCE_THRESHOLD = 0.20
IMAGE_SIZE = (64, 64)
EXPECTED_NUM_CLASSES = 5


def preprocess_image(image_path, image_size=IMAGE_SIZE):
    img = Image.open(image_path).convert("RGB")
    img = img.resize(image_size, Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    return arr


def load_images_from_directory(data_dir, image_size=IMAGE_SIZE, class_names=None):
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"No existe la carpeta de datos: {data_dir}")

    if class_names is None:
        class_names = [
            d for d in sorted(os.listdir(data_dir)) if os.path.isdir(os.path.join(data_dir, d))
        ]
    else:
        class_names = list(class_names)

    if len(class_names) != EXPECTED_NUM_CLASSES:
        raise ValueError(
            f"Se esperaban exactamente {EXPECTED_NUM_CLASSES} clases, pero se encontraron "
            f"{len(class_names)}: {class_names}"
        )

    x_data = []
    y_data = []

    for idx, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            raise FileNotFoundError(f"No existe la carpeta de clase: {class_dir}")

        for file_name in os.listdir(class_dir):
            ext = os.path.splitext(file_name)[1].lower()
            if ext not in VALID_EXTENSIONS:
                continue

            file_path = os.path.join(class_dir, file_name)
            img_arr = preprocess_image(file_path, image_size=image_size)
            x_data.append(img_arr)
            y_data.append(idx)

    if not x_data:
        raise ValueError(f"No se encontraron imagenes validas en: {data_dir}")

    x = np.array(x_data, dtype=np.float32)
    y = np.array(y_data, dtype=np.int64)
    x = np.transpose(x, (0, 3, 1, 2))

    return x, y, class_names


def train_test_split(x, y, test_ratio=0.2):
    n = x.shape[0]
    split = int(n * (1.0 - test_ratio))
    return x[:split], y[:split], x[split:], y[split:]


def random_horizontal_flip_batch(x, flip_prob=0.5):
    x_aug = x.copy()
    mask = np.random.rand(x_aug.shape[0]) < flip_prob
    x_aug[mask] = x_aug[mask, :, :, ::-1]
    return x_aug


def accuracy_with_threshold(pred_outputs, y_true):
    valid_idx = [i for i, p in enumerate(pred_outputs) if p != "Ninguna de las anteriores"]
    if not valid_idx:
        return 0.0, 1.0

    y_pred_valid = np.array([pred_outputs[i] for i in valid_idx], dtype=int)
    y_true_valid = y_true[valid_idx]

    acc = np.mean(y_pred_valid == y_true_valid)
    reject_rate = 1.0 - (len(valid_idx) / len(pred_outputs))
    return float(acc), float(reject_rate)


def build_cnn(num_classes=EXPECTED_NUM_CLASSES, learning_rate=0.1, confidence_threshold=CONFIDENCE_THRESHOLD):
    model = CNN(num_classes=num_classes, confidence_threshold=confidence_threshold)

    model.add(ConvolutionalLayer(input_shape=(3, 64, 64), num_filters=8, filter_size=3, stride=2, padding=1, learning_rate=learning_rate))
    model.add(ReLU())
    model.add(PoolingLayer(pool_size=2, stride=2))

    model.add(ConvolutionalLayer(input_shape=(8, 16, 16), num_filters=16, filter_size=3, stride=2, padding=1, learning_rate=learning_rate))
    model.add(ReLU())
    model.add(PoolingLayer(pool_size=2, stride=2))

    model.add(ConvolutionalLayer(input_shape=(16, 4, 4), num_filters=32, filter_size=3, stride=1, padding=1, learning_rate=learning_rate))
    model.add(ReLU())
    model.add(PoolingLayer(pool_size=2, stride=2))

    model.add(DenseLayer(input_size=32 * 2 * 2, output_size=32, learning_rate=learning_rate))
    model.add(ReLU())
    model.add(DenseLayer(input_size=32, output_size=num_classes, learning_rate=learning_rate))
    model.add(Softmax())

    return model


def predict_image_file(model, image_path, class_names, confidence_threshold=CONFIDENCE_THRESHOLD, image_size=IMAGE_SIZE):
    img_arr = preprocess_image(image_path, image_size=image_size)
    x = np.transpose(img_arr, (2, 0, 1))[np.newaxis, :, :, :]
    preds, probs = model.predict(x, confidence_threshold=confidence_threshold)
    return preds[0], probs[0]


def main():
    dataset_root = "dataset"
    train_dir = os.path.join(dataset_root, "train")
    test_dir = os.path.join(dataset_root, "test")

    if os.path.isdir(train_dir):
        detected_classes = [
            d for d in sorted(os.listdir(train_dir)) if os.path.isdir(os.path.join(train_dir, d))
        ]
        x_train, y_train, class_names = load_images_from_directory(
            train_dir,
            image_size=IMAGE_SIZE,
            class_names=detected_classes,
        )
        if os.path.isdir(test_dir):
            x_test, y_test, _ = load_images_from_directory(
                test_dir,
                image_size=IMAGE_SIZE,
                class_names=class_names,
            )
        else:
            x_all, y_all, class_names = load_images_from_directory(
                dataset_root,
                image_size=IMAGE_SIZE,
                class_names=detected_classes,
            )
            x_train, y_train, x_test, y_test = train_test_split(x_all, y_all, test_ratio=0.2)
    else:
        x_all, y_all, class_names = load_images_from_directory(
            dataset_root,
            image_size=IMAGE_SIZE,
            class_names=None,
        )
        x_train, y_train, x_test, y_test = train_test_split(x_all, y_all, test_ratio=0.2)

    print(f"Clases detectadas: {class_names}")
    print(f"Train: {len(y_train)} imagenes | Test: {len(y_test)} imagenes")

    x_train_aug = random_horizontal_flip_batch(x_train, flip_prob=0.5)

    model = build_cnn(num_classes=len(class_names), learning_rate=0.1, confidence_threshold=CONFIDENCE_THRESHOLD)
    model.fit(x_train_aug, y_train, epochs=35, batch_size=16, verbose=True, shuffle=True)

    preds, probs = model.predict(x_test, confidence_threshold=CONFIDENCE_THRESHOLD)
    acc_valid, reject_rate = accuracy_with_threshold(preds, y_test)

    print("\nEvaluacion con umbral de confianza:")
    print(f"Umbral usado: {CONFIDENCE_THRESHOLD:.2f}")
    print(f"Muestras test: {len(y_test)}")
    print(f"Accuracy en muestras aceptadas: {acc_valid:.4f}")
    print(f"Tasa de rechazo ('Ninguna de las anteriores'): {reject_rate:.4f}")

    print("\nEjemplos de prediccion (primeras 10):")
    for i in range(min(10, len(preds))):
        print(
            f"Real={y_test[i]} | Pred={preds[i]} | ConfianzaMax={np.max(probs[i]):.4f}"
        )


if __name__ == "__main__":
    main()
