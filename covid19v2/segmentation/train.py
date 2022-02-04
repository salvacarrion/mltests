import os
from pathlib import Path
import random

import segmentation_models as sm
import tensorflow as tf
from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score
from segmentation_models.utils import set_trainable
from tensorflow.keras.optimizers import SGD, Adam

from covid19v2.segmentation.dataset import DatasetMasks, DataloaderMasks
from covid19v2.segmentation.da import da_tr_fn, da_ts_fn
from covid19v2.segmentation import plot
from covid19v2.segmentation import utils

# Fix sm
sm.set_framework('tf.keras')
sm.framework()

# Variables
TARGET_SIZE = 256
BATCH_SIZE = 128
EPOCHS_STAGE1 = 2000
EPOCHS_STAGE2 = 2000
BACKBONE = "resnet34"
RUN_NAME = "v2"

BASE_PATH = "/home/scarrion/datasets/nn/vision/lungs_masks"
print(BASE_PATH)

strategy = tf.distribute.MirroredStrategy()


def train(model, train_dataset, val_dataset, use_multiprocessing=False, workers=1):
    # Outputs
    checkpoints_path = os.path.join(BASE_PATH, "runs", RUN_NAME, "models")
    logs_path = os.path.join(BASE_PATH, "runs", RUN_NAME, "logs")
    plots_path = os.path.join(BASE_PATH, "runs", RUN_NAME, "plots")

    # Create folders
    for dir_i in [checkpoints_path, logs_path, plots_path]:
        Path(dir_i).mkdir(parents=True, exist_ok=True)

    # Build dataloaders
    train_dataloader = DataloaderMasks(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataloaderMasks(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Callbacks
    model_callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=1e-5),
        tf.keras.callbacks.EarlyStopping(patience=20),
        # tf.keras.callbacks.ModelCheckpoint(filepath=checkpoints_path, save_best_only=True, mode='min'),  # It can make the end of an epoch extremely slow
        tf.keras.callbacks.TensorBoard(log_dir=logs_path),
        # WandbCallback(),
    ]

    # define model
    model.summary()

    with strategy.scope():
        # Compile the model
        model.compile(optimizer=Adam(learning_rate=1e-3), loss=bce_jaccard_loss, metrics=[iou_score])

    # train the model on the new data for a few epochs
    print("Training decoder first...")
    history1 = model.fit(train_dataloader, validation_data=val_dataloader, epochs=EPOCHS_STAGE1, callbacks=model_callbacks,
                         use_multiprocessing=use_multiprocessing, workers=workers)
    print("Initial training results:")
    print(history1)
    if plots_path:
        plot.plot_hist(history1, title="Training decoder", savepath=plots_path, suffix="_initial")

    with strategy.scope():
        # we need to recompile the model for these modifications to take effect
        # we use SGD with a low learning rate
        print("Fine-tuning model...")
        set_trainable(model, recompile=False)
        model.compile(optimizer=SGD(learning_rate=1e-4, momentum=0.9), loss=bce_jaccard_loss, metrics=[iou_score])

    # we train our model again (this time fine-tuning the top 2 inception blocks
    # alongside the top Dense layers
    history2 = model.fit(train_dataloader, validation_data=val_dataloader, epochs=EPOCHS_STAGE2, callbacks=model_callbacks,
                         use_multiprocessing=use_multiprocessing, workers=workers)
    print("Fine-tuning results:")
    print(history2)
    if plots_path:
        plot.plot_hist(history2, title="Fine-tuning full model", savepath=plots_path, suffix="_finetuning")

    # Save model
    print("Saving last model...")
    last_checkpoint_path = os.path.join(checkpoints_path, "checkpoint_last")
    model.save(last_checkpoint_path)
    print(f"Model saved at: {last_checkpoint_path}")


def main():
    # Vars
    images_dir = os.path.join(BASE_PATH, "images", "train")
    masks_dir = os.path.join(BASE_PATH, "masks", "train")

    # Get data by matching the filename (no basepath)
    images_files = set([f for f in os.listdir(images_dir)])
    masks_files = set([f for f in os.listdir(masks_dir)])
    files = list(masks_files.intersection(images_files))
    random.shuffle(files)  # Randomize
    print(f"Total Images+Masks: {len(files)}")

    # Get model + auxiliar functions
    with strategy.scope():
        model, preprocess_fn = utils.get_model(backbone=BACKBONE)

    # Datasets
    tr_size = int(len(files)*0.8)
    train_dataset = DatasetMasks(BASE_PATH, folder="train", files=files[:tr_size], da_fn=da_tr_fn(TARGET_SIZE, TARGET_SIZE), preprocess_fn=preprocess_fn)
    val_dataset = DatasetMasks(BASE_PATH, folder="train", files=files[tr_size:], da_fn=da_ts_fn(TARGET_SIZE, TARGET_SIZE), preprocess_fn=preprocess_fn)

    # Visualize
    # for i in range(10):
    #     plot.plot_dataset(train_dataset, img_i=i, num_aug=1, mode="all")  # img+img_da, img+mask, all

    # Train
    train(model, train_dataset, val_dataset)


if __name__ == "__main__":
    main()
    print("Done!")
