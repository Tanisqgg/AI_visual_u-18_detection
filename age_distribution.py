import os
from pathlib import Path
import pandas as pd
from PIL import Image
from datetime import datetime, date

# 1. point to your base “wiki_crop” folder
base_dir = Path(r"C:\Users\tanis\OneDrive\Desktop\tan priv\collegeTrade\AI_visual_u-18_detection\wiki_crop\wiki_crop")

# collect (age, filepath) pairs

records = []
for sub in sorted(base_dir.iterdir()):
    if not sub.is_dir():
        continue

    for img in sub.iterdir():
        if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue

        parts = img.stem.split("_")
        if len(parts) != 3:
            continue  # skip anything unexpected

        # parse
        _, dob_str, photo_year_str = parts
        try:
            dob = datetime.strptime(dob_str, "%Y-%m-%d").date()
            photo_year = int(photo_year_str)
        except ValueError:
            continue  # malformed—skip

        # assume photo taken Jan 1 of photo_year
        photo_date = date(photo_year, 1, 1)

        # compute age at photo_date
        age = photo_date.year - dob.year - ((photo_date.month, photo_date.day) < (dob.month, dob.day))

        records.append({
            "age": age,
            "path": str(img)
        })

df = pd.DataFrame(records)

# anything that isn’t 0–100
outliers = df[(df.age < 0) | (df.age > 100)]
print("Found", len(outliers), "outliers. Examples:")
print(outliers.head()[["age","path"]])

# keep only 0–100
df_clean = df[(df.age >= 0) & (df.age <= 100)].copy()

# count per age
age_counts = df_clean.age.value_counts().sort_index()
print(age_counts)

import matplotlib.pyplot as plt

plt.figure(figsize=(12,4))
age_counts.plot(kind="bar", width=1)
plt.xlabel("Age")
plt.ylabel("Number of Faces")
plt.title("Wikipedia Faces: Count by Age")
plt.tight_layout()
plt.show()

