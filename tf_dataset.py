# tf_dataset.py
from pathlib import Path
import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE


def get_pairs(root, split):
    root = Path(root)
    img_dir = root / split / "images"
    mask_dir = root / split / "masks"

    img_paths = sorted([str(p) for p in img_dir.glob("*") if p.is_file()])
    mask_paths = []

    for img_path in img_paths:
        stem = Path(img_path).stem
        mask_path = mask_dir / f"{stem}.png"
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask for image: {img_path}")
        mask_paths.append(str(mask_path))

    return img_paths, mask_paths


def decode_image(img_path):
    img_bytes = tf.io.read_file(img_path)
    img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    return img


def decode_mask(mask_path):
    mask_bytes = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask_bytes, channels=1)
    mask.set_shape([None, None, 1])
    return mask


def preprocess(img_path, mask_path, image_size=(256, 256), normalize=True):
    img = decode_image(img_path)
    mask = decode_mask(mask_path)

    img = tf.image.resize(img, image_size, method="bilinear")
    mask = tf.image.resize(mask, image_size, method="nearest")

    img = tf.cast(img, tf.float32)
    if normalize:
        img = img / 255.0

    mask = tf.cast(mask > 127, tf.float32)

    return img, mask


def augment(img, mask):
    flip_lr = tf.random.uniform(()) > 0.5
    flip_ud = tf.random.uniform(()) > 0.5

    if flip_lr:
        img = tf.image.flip_left_right(img)
        mask = tf.image.flip_left_right(mask)

    if flip_ud:
        img = tf.image.flip_up_down(img)
        mask = tf.image.flip_up_down(mask)

    if tf.random.uniform(()) > 0.5:
        img = tf.image.random_brightness(img, max_delta=0.08)
        img = tf.clip_by_value(img, 0.0, 1.0)

    return img, mask


def build_dataset(
    root,
    split,
    image_size=(256, 256),
    batch_size=8,
    training=False,
    shuffle_buffer=1024
):
    img_paths, mask_paths = get_pairs(root, split)

    ds = tf.data.Dataset.from_tensor_slices((img_paths, mask_paths))

    if training:
        ds = ds.shuffle(min(len(img_paths), shuffle_buffer), reshuffle_each_iteration=True)

    ds = ds.map(
        lambda x, y: preprocess(x, y, image_size=image_size),
        num_parallel_calls=AUTOTUNE
    )

    if training:
        ds = ds.map(augment, num_parallel_calls=AUTOTUNE)

    ds = ds.batch(batch_size)
    ds = ds.prefetch(AUTOTUNE)

    return ds, len(img_paths)


def build_all_datasets(root, image_size=(256, 256), batch_size=8):
    train_ds, train_count = build_dataset(
        root=root,
        split="train",
        image_size=image_size,
        batch_size=batch_size,
        training=True
    )

    val_ds, val_count = build_dataset(
        root=root,
        split="val",
        image_size=image_size,
        batch_size=batch_size,
        training=False
    )

    test_ds, test_count = build_dataset(
        root=root,
        split="test",
        image_size=image_size,
        batch_size=batch_size,
        training=False
    )

    return {
        "train": train_ds,
        "val": val_ds,
        "test": test_ds,
        "train_count": train_count,
        "val_count": val_count,
        "test_count": test_count,
    }


if __name__ == "__main__":
    data_root = r"D:\dataset\processed_binary"

    data = build_all_datasets(
        root=data_root,
        image_size=(256, 256),
        batch_size=4
    )

    print("Train count:", data["train_count"])
    print("Val count:", data["val_count"])
    print("Test count:", data["test_count"])

    for images, masks in data["train"].take(1):
        print("Images shape:", images.shape)
        print("Masks shape :", masks.shape)
        print("Images dtype:", images.dtype)
        print("Masks dtype :", masks.dtype)
        print("Mask min/max:", tf.reduce_min(masks).numpy(), tf.reduce_max(masks).numpy())
