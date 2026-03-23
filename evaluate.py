# evaluate.py
import argparse
from pathlib import Path
import tensorflow as tf

from tf_dataset import build_all_datasets


def dice_coef(y_true, y_pred, smooth=1e-6):
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    y_true = tf.cast(y_true, tf.float32)

    inter = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    denom = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3])
    dice = (2.0 * inter + smooth) / (denom + smooth)
    return tf.reduce_mean(dice)


def iou_score(y_true, y_pred, smooth=1e-6):
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    y_true = tf.cast(y_true, tf.float32)

    inter = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3]) - inter
    iou = (inter + smooth) / (union + smooth)
    return tf.reduce_mean(iou)


def dice_loss_from_probs(y_true, y_prob, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    inter = tf.reduce_sum(y_true * y_prob, axis=[1, 2, 3])
    denom = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_prob, axis=[1, 2, 3])
    dice = (2.0 * inter + smooth) / (denom + smooth)
    return 1.0 - tf.reduce_mean(dice)


def bce_dice_loss(y_true, y_pred):
    bce = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred))
    dloss = dice_loss_from_probs(y_true, y_pred)
    return 0.5 * bce + 0.5 * dloss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=r"D:\dataset\processed_binary")
    parser.add_argument("--model_path", type=str, default=r"D:\dataset\outputs\fast_scnn\saved_model\final_model.keras")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    image_size = (args.img_size, args.img_size)

    data = build_all_datasets(
        root=args.data_root,
        image_size=image_size,
        batch_size=args.batch_size
    )
    test_ds = data["test"]

    model_path = Path(args.model_path)

    custom_objects = {
        "bce_dice_loss": bce_dice_loss,
        "dice_coef": dice_coef,
        "iou_score": iou_score
    }

    if model_path.suffix == ".keras":
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    else:
        model = tf.keras.models.load_model(model_path)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=bce_dice_loss,
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="bin_acc"),
            dice_coef,
            iou_score
        ]
    )

    results = model.evaluate(test_ds, return_dict=True)

    print("\n===== TEST RESULTS =====")
    for k, v in results.items():
        print(f"{k}: {v:.6f}")


if __name__ == "__main__":
    main()
