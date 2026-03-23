# train.py
import os
import argparse
from pathlib import Path

import tensorflow as tf

from tf_dataset import build_all_datasets
from models.fast_scnn import build_fast_scnn
from models.enet import build_enet


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


def build_model(model_name, input_shape=(256, 256, 3)):
    model_name = model_name.lower()
    if model_name == "fast_scnn":
        return build_fast_scnn(input_shape=input_shape, num_classes=1)
    elif model_name == "enet":
        return build_enet(input_shape=input_shape, num_classes=1)
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def make_save_dirs(save_root, model_name):
    save_root = Path(save_root)
    ckpt_dir = save_root / model_name / "checkpoints"
    log_dir = save_root / model_name / "logs"
    export_dir = save_root / model_name / "saved_model"

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    export_dir.mkdir(parents=True, exist_ok=True)

    return ckpt_dir, log_dir, export_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=r"D:\dataset\processed_binary")
    parser.add_argument("--model", type=str, default="fast_scnn", choices=["fast_scnn", "enet"])
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save_root", type=str, default="outputs")
    args = parser.parse_args()

    image_size = (args.img_size, args.img_size)

    data = build_all_datasets(
        root=args.data_root,
        image_size=image_size,
        batch_size=args.batch_size
    )

    train_ds = data["train"]
    val_ds = data["val"]
    test_ds = data["test"]

    print("Train count:", data["train_count"])
    print("Val count  :", data["val_count"])
    print("Test count :", data["test_count"])

    model = build_model(
        model_name=args.model,
        input_shape=(args.img_size, args.img_size, 3)
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss=bce_dice_loss,
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="bin_acc"),
            dice_coef,
            iou_score
        ]
    )

    ckpt_dir, log_dir, export_dir = make_save_dirs(args.save_root, args.model)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_dir / "best.keras"),
            monitor="val_iou_score",
            mode="max",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_iou_score",
            mode="max",
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_iou_score",
            mode="max",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger(
            filename=str(log_dir / "train_log.csv"),
            append=False
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=str(log_dir)
        )
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks
    )

    print("\nEvaluating on test set...")
    results = model.evaluate(test_ds, return_dict=True)
    print(results)

    model.save(export_dir / "final_model.keras")
    model.export(str(export_dir / "saved_model"))

    print("\nSaved final model to:", export_dir)


if __name__ == "__main__":
    main()
