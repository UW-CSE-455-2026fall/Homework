from __future__ import annotations

from dataclasses import dataclass
import numpy as np

try:
    from PIL import Image as PILImage
except ImportError:
    PILImage = None

import os

import src.matrix as matrix

# visionlib (the prebuilt C/stb backend) has been removed; image I/O is
# Pillow-only. Kept as a constant so the test helpers that select the Pillow
# ground-truth baselines continue to work.
_vision = None


@dataclass
class Image:
    """
    CHW layout like the C code: data shape = (c, h, w), float32 in [0,1]
    """
    w: int
    h: int
    c: int
    data: np.ndarray  # (c,h,w) float32


def _need_pillow():
    if PILImage is None:
        raise ImportError("Pillow not installed. Run: pip install pillow")


def make_image(w: int, h: int, c: int) -> Image:
    w, h, c = int(w), int(h), int(c)
    data = np.zeros((c, h, w), dtype=np.float32)
    return Image(w=w, h=h, c=c, data=data)


def free_image(im: Image) -> None:
    # For API compatibility; Python GC handles it automatically.
    return


def load_image(f: str, channels: int = 0) -> Image:
    """
    channels:
      0 = auto (drop alpha if present; default to RGB)
      1 = force grayscale
      3 = force RGB
    """
    _need_pillow()
    pil = PILImage.open(f)

    if channels == 1:
        pil = pil.convert("L")
    elif channels == 3:
        pil = pil.convert("RGB")
    elif channels == 0:
        if pil.mode in ("RGBA", "LA", "P"):
            pil = pil.convert("RGB")
        elif pil.mode == "L":
            pass
        else:
            pil = pil.convert("RGB")
    else:
        raise ValueError("channels must be 0, 1, or 3")

    arr = np.array(pil)
    if arr.ndim == 2:
        arr = arr[:, :, None]

    arr = arr.astype(np.float32) / 255.0
    chw = np.transpose(arr, (2, 0, 1)).astype(np.float32, copy=False)
    chw = np.ascontiguousarray(chw)

    c, h, w = chw.shape
    if c == 4:
        chw = chw[:3]
        c = 3

    return Image(w=w, h=h, c=c, data=chw)


def _save_any(im: Image, name_no_ext: str, ext: str) -> None:
    _need_pillow()
    hwc = np.transpose(im.data, (1, 2, 0))
    hwc = np.clip(hwc, 0.0, 1.0)
    u8 = (hwc * 255.0 + 0.5).astype(np.uint8)

    if u8.shape[2] == 1:
        out = PILImage.fromarray(u8[:, :, 0], mode="L")
    else:
        out = PILImage.fromarray(u8, mode="RGB")

    out.save(f"{name_no_ext}.{ext}")


def save_image(im: Image, f: str) -> None:
    _save_any(im, f, "jpg")


def save_png(im: Image, f: str) -> None:
    _save_any(im, f, "png")


# ====================================================================
# HW4 Neural Network Bindings (Pure Python implementation)
# ====================================================================

# Activation Enums (Mapped to strings to match our python classifier.py)
LINEAR = 'LINEAR'
LOGISTIC = 'LOGISTIC'
RELU = 'RELU'
LRELU = 'LRELU'
SOFTMAX = 'SOFTMAX'

def load_classification_data(images_list_path: str, label_file_path: str, k: int):
    """
    Reads a list of image paths, loads the images, flattens them into rows of a matrix,
    and extracts the corresponding one-hot encoded labels.
    """
    with open(images_list_path, 'r') as f:
        image_paths = [line.strip() for line in f.readlines() if line.strip()]

    n = len(image_paths)
    if n == 0:
        raise ValueError(f"No image paths found in {images_list_path}")

    # Class label = parent directory name of each image (e.g. "train/3/x.png" -> "3").
    # Sorted for deterministic class-index ordering across train/test splits.
    path_classes = [os.path.basename(os.path.dirname(p)) for p in image_paths]
    class_names = sorted(set(path_classes))
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    num_classes = len(class_names)

    # Load the first image to dynamically determine the feature size (cols)
    test_im = load_image(image_paths[0])
    cols = test_im.w * test_im.h * test_im.c

    X = matrix.make_matrix(n, cols)
    y = matrix.make_matrix(n, num_classes)

    for i, (path, cls) in enumerate(zip(image_paths, path_classes)):
        im = load_image(path)
        X.data[i] = im.data.flatten()
        y.data[i][class_to_idx[cls]] = 1.0

    return matrix.data(X, y)

def make_model(layers):
    """
    Constructs a model holding the provided list of layers.
    This replaces the C-struct version and hooks into our Python implementation.
    """
    # We dynamically import the python model class to avoid circular imports
    from src.hw4.classifier import model

    m = model()
    m.n = len(layers)
    m.layers = layers
    return m