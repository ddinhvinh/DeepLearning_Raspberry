# infer_image.py
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import tensorflow as tf


def load_image(img_path, image_size=(256, 256)):
    img = Image.open(img_path).convert("RGB")
    orig_size = img.size
    img_resized = img.resize(image_size, Image.BILINEAR)
    img_arr = np.array(img_resized).astype(np.float32) / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)
    return img, img_arr, orig_size


def load_mask(mask_path, image_size=(256, 256)):
    mask = Image.open(mask_path).convert("L")
    mask_resized = mask.resize(image_size, Image.NEAREST)
    mask_arr = np.array(mask_resized)
    mask_bin = (mask_arr > 127).astype(np.uint8) * 255
    return Image.fromarray(mask_bin)


def predict_mask(model, img_arr, threshold=0.5):
    pred = model.predict(img_arr, verbose=0)[0]
    if pred.shape[-1] == 1:
        pred = pred[..., 0]
    pred_bin = (pred > threshold).astype(np.uint8) * 255
    return Image.fromarray(pred_bin)


def create_overlay(orig_img, pred_mask, color=(255, 0, 0), alpha=0.4):
    orig = np.array(orig_img).astype(np.float32)
    mask = np.array(pred_mask.resize(orig_img.size, Image.NEAREST))

    overlay = orig.copy()
    overlay[mask > 0] = (
        (1 - alpha) * overlay[mask > 0] + alpha * np.array(color, dtype=np.float32)
    )

    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return Image.fromarray(overlay)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--mask_path", type=str, default=None)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output_dir", type=str, default="outputs/infer")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_size = (args.img_size, args.img_size)

    model = tf.keras.models.load_model(args.model_path, compile=False)

    orig_img, img_arr, orig_size = load_image(args.image_path, image_size=image_size)
    pred_mask = predict_mask(model, img_arr, threshold=args.threshold)
    pred_mask = pred_mask.resize(orig_size, Image.NEAREST)

    overlay = create_overlay(orig_img, pred_mask)

    stem = Path(args.image_path).stem

    orig_img.save(output_dir / f"{stem}_image.png")
    pred_mask.save(output_dir / f"{stem}_pred_mask.png")
    overlay.save(output_dir / f"{stem}_overlay.png")

    if args.mask_path:
        gt_mask = load_mask(args.mask_path, image_size=image_size)
        gt_mask = gt_mask.resize(orig_size, Image.NEAREST)
        gt_mask.save(output_dir / f"{stem}_gt_mask.png")

    print("Saved results to:", output_dir.resolve())


if __name__ == "__main__":
    main()
