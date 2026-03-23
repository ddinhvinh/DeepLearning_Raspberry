from pathlib import Path
from PIL import Image
import numpy as np

ROOT = Path("processed_binary")

def check_split(split):
    img_dir = ROOT / split / "images"
    mask_dir = ROOT / split / "masks"

    img_files = sorted([p for p in img_dir.iterdir() if p.is_file()])
    mask_files = sorted([p for p in mask_dir.iterdir() if p.is_file()])

    print(f"\n=== {split.upper()} ===")
    print("Images:", len(img_files))
    print("Masks :", len(mask_files))

    img_stems = {p.stem for p in img_files}
    mask_stems = {p.stem for p in mask_files}

    missing_masks = sorted(img_stems - mask_stems)
    missing_images = sorted(mask_stems - img_stems)

    print("Missing masks :", len(missing_masks))
    print("Missing images:", len(missing_images))

    sample_masks = mask_files[:10]
    for p in sample_masks:
        arr = np.array(Image.open(p))
        uniq = np.unique(arr)
        print(f"{p.name}: unique={uniq[:10]}")

for split in ["train", "val", "test"]:
    check_split(split)
