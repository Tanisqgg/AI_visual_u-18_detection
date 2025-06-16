"""Evaluate the age estimation model on the local WIKI dataset.

This script iterates over the images in ``wiki_crop/wiki_crop`` and uses the
Vision Transformer age estimation model to predict ages. It then computes
standard regression metrics such as MAE, RMSE, Cumulative Score (CS) and R².
"""

from __future__ import annotations

import math
from pathlib import Path
from datetime import datetime, date

import numpy as np
from PIL import Image
from transformers import pipeline
import torch

DATA_DIR = Path("wiki_crop/wiki_crop")
MODEL_DIR = Path("local_model")


def parse_age_from_filename(path: Path) -> int:
    """Return the ground-truth age extracted from the WIKI filename."""
    parts = path.stem.split("_")
    if len(parts) != 3:
        raise ValueError(f"Unexpected filename format: {path.name}")
    _, dob_str, photo_year_str = parts
    dob = datetime.strptime(dob_str, "%Y-%m-%d").date()
    photo_date = date(int(photo_year_str), 1, 1)
    age = photo_date.year - dob.year - (
        (photo_date.month, photo_date.day) < (dob.month, dob.day)
    )
    return age


def parse_predicted_age(label: str) -> float:
    """Convert a model label like ``"16-20"`` or ``"34"`` into a numeric age."""
    if "-" in label:
        start, end = label.split("-")
        return (float(start) + float(end)) / 2
    try:
        return float(label)
    except ValueError:
        return 0.0


def evaluate_dataset() -> dict[str, float]:
    predictor = pipeline(
        "image-classification",
        model=str(MODEL_DIR),
        device=0 if torch.cuda.is_available() else -1,
    )

    image_paths = sorted(DATA_DIR.rglob("*.jpg"))
    y_true: list[float] = []
    y_pred: list[float] = []

    for img_path in image_paths:
        gt_age = parse_age_from_filename(img_path)
        prediction = predictor(Image.open(img_path))[0]["label"]
        pred_age = parse_predicted_age(prediction)
        y_true.append(gt_age)
        y_pred.append(pred_age)

    y_true_np = np.array(y_true, dtype=float)
    y_pred_np = np.array(y_pred, dtype=float)
    errors = y_pred_np - y_true_np

    mae = float(np.mean(np.abs(errors)))
    rmse = float(math.sqrt(np.mean(errors ** 2)))
    cs1 = float(np.mean(np.abs(errors) <= 1) * 100)
    cs3 = float(np.mean(np.abs(errors) <= 3) * 100)
    cs5 = float(np.mean(np.abs(errors) <= 5) * 100)
    r2 = float(
        1.0 - np.sum(errors ** 2) / np.sum((y_true_np - y_true_np.mean()) ** 2)
    )
    std_err = float(np.std(errors))

    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "CS_1": cs1,
        "CS_3": cs3,
        "CS_5": cs5,
        "StdError": std_err,
        "R2": r2,
    }
    return metrics


if __name__ == "__main__":
    results = evaluate_dataset()
    for key, value in results.items():
        print(f"{key}: {value:.3f}")
