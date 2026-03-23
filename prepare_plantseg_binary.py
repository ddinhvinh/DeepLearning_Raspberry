import shutil
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(".")
RAW_ROOT = PROJECT_ROOT / "raw" / "plantseg_raw"
IMG_DIR = RAW_ROOT / "images"
ANN_DIR = RAW_ROOT / "annotations"

meta_candidates = [
    RAW_ROOT / "Metadata.csv",
    RAW_ROOT / "Metadata",
    RAW_ROOT / "metadata.csv",
    RAW_ROOT / "metadata",
]

META_CSV = None
for p in meta_candidates:
    if p.exists():
        META_CSV = p
        break

if META_CSV is None:
    raise FileNotFoundError(f"Không tìm thấy file metadata trong: {RAW_ROOT}")

OUT_ROOT = PROJECT_ROOT / "processed_binary"
SPLIT_DIR = PROJECT_ROOT / "splits"

for split in ["train", "val", "test"]:
    (OUT_ROOT / split / "images").mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / split / "masks").mkdir(parents=True, exist_ok=True)

SPLIT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(META_CSV)

print("Metadata file:", META_CSV.resolve())
print("Columns:", df.columns.tolist())

def normalize_split(x):
    x = str(x).strip().lower()
    if x == "training":
        return "train"
    if x == "validation":
        return "val"
    if x == "test":
        return "test"
    return None

print("Indexing image files recursively...")
image_index = {}
for p in IMG_DIR.rglob("*"):
    if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png"]:
        image_index[p.name.lower()] = p
        image_index[p.stem.lower()] = p

print("Indexing annotation files recursively...")
mask_index = {}
for p in ANN_DIR.rglob("*"):
    if p.is_file() and p.suffix.lower() in [".png", ".jpg", ".jpeg"]:
        mask_index[p.name.lower()] = p
        mask_index[p.stem.lower()] = p

print(f"Indexed images: {len(image_index)}")
print(f"Indexed masks: {len(mask_index)}")

train_names, val_names, test_names = [], [], []
missing_count = 0
ok_count = 0

for _, row in df.iterrows():
    split = normalize_split(row["Split"])
    if split is None:
        continue

    image_name = str(row["Name"]).strip()
    label_name = str(row["Label file"]).strip()

    image_key1 = image_name.lower()
    image_key2 = Path(image_name).stem.lower()

    label_key1 = label_name.lower()
    label_key2 = Path(label_name).stem.lower()

    src_img = image_index.get(image_key1) or image_index.get(image_key2)
    src_mask = (
        mask_index.get(label_key1)
        or mask_index.get(label_key2)
        or mask_index.get(image_key1)
        or mask_index.get(image_key2)
    )

    if src_img is None or src_mask is None:
        print(f"[WARN] Missing file for image={image_name}, label={label_name}")
        missing_count += 1
        continue

    mask = np.array(Image.open(src_mask))
    binary_mask = (mask > 0).astype(np.uint8) * 255

    image_stem = src_img.stem
    dst_img = OUT_ROOT / split / "images" / src_img.name
    dst_mask = OUT_ROOT / split / "masks" / f"{image_stem}.png"

    shutil.copy2(src_img, dst_img)
    Image.fromarray(binary_mask).save(dst_mask)

    if split == "train":
        train_names.append(image_stem)
    elif split == "val":
        val_names.append(image_stem)
    else:
        test_names.append(image_stem)

    ok_count += 1

(SPLIT_DIR / "train.txt").write_text("\n".join(train_names), encoding="utf-8")
(SPLIT_DIR / "val.txt").write_text("\n".join(val_names), encoding="utf-8")
(SPLIT_DIR / "test.txt").write_text("\n".join(test_names), encoding="utf-8")

print("\nDone preparing PlantSeg binary dataset.")
print("Copied:", ok_count)
print("Missing:", missing_count)
print("Train:", len(train_names))
print("Val:  ", len(val_names))
print("Test: ", len(test_names))
