import os
import argparse
import math
from datetime import datetime, date
from typing import List

import numpy as np
from PIL import Image
from transformers import pipeline


def parse_age_from_filename(path: str) -> int:
    """Extract the ground truth age from a wiki_crop filename."""
    name = os.path.basename(path)
    parts = name.split("_")
    if len(parts) != 3:
        raise ValueError(f"Unexpected filename: {name}")
    dob_str = parts[1]
    year_str = parts[2].split(".")[0]
    dob = datetime.strptime(dob_str, "%Y-%m-%d").date()
    photo_date = date(int(year_str), 1, 1)
    age = photo_date.year - dob.year - (
        (photo_date.month, photo_date.day) < (dob.month, dob.day)
    )
    return age


def parse_age_label(label: str) -> int:
    """Convert a model age label like '16-20' or '55+' to an integer age."""
    label = label.strip()
    if "-" in label:
        start, end = label.split("-")
        return (int(start) + int(end)) // 2
    if label.endswith("+"):
        return int(label[:-1])
    return int(label)


def evaluate(dataset_dir: str, model_path: str, max_images: int | None = None):
    age_pipe = pipeline("image-classification", model=model_path)

    y_true: List[int] = []
    y_pred: List[int] = []
    processed = 0

    for root, _, files in os.walk(dataset_dir):
        for f in files:
            if not f.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img_path = os.path.join(root, f)
            true_age = parse_age_from_filename(img_path)
            img = Image.open(img_path).convert("RGB")
            pred_label = age_pipe(img)[0]["label"]
            pred_age = parse_age_label(pred_label)

            y_true.append(true_age)
            y_pred.append(pred_age)
            processed += 1
            if max_images and processed >= max_images:
                break
        if max_images and processed >= max_images:
            break

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    errors = y_pred_arr - y_true_arr

    mae = np.mean(np.abs(errors))
    rmse = math.sqrt(np.mean(errors ** 2))
    r2 = 1 - np.sum(errors ** 2) / np.sum((y_true_arr - np.mean(y_true_arr)) ** 2)
    cs1 = np.mean(np.abs(errors) <= 1) * 100
    cs3 = np.mean(np.abs(errors) <= 3) * 100
    cs5 = np.mean(np.abs(errors) <= 5) * 100

    print(f"Processed {processed} images")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R^2: {r2:.3f}")
    print(f"CS@1: {cs1:.2f}%")
    print(f"CS@3: {cs3:.2f}%")
    print(f"CS@5: {cs5:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate age model on wiki dataset")
    parser.add_argument("--data", default="wiki_crop/wiki_crop", help="Path to wiki_crop dataset")
    parser.add_argument("--model", default="local_model", help="Path to local model directory")
    parser.add_argument("--max-images", type=int, default=None, help="Limit number of images for quick evaluation")
    args = parser.parse_args()

    evaluate(args.data, args.model, args.max_images)
