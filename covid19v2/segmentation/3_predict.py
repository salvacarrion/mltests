import os
from pathlib import Path
import random
import tqdm

import segmentation_models as sm
import tensorflow as tf
from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score

import numpy as np
from PIL import Image

from covid19v2.segmentation.dataset import DatasetImages, DataloaderImages
from covid19v2.segmentation.da import da_ts_fn
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
SAVE_OVERLAY = True
RUN_NAME = "v2"

BASE_PATH = "/home/scarrion/datasets/nn/vision/lungs_masks"
print(BASE_PATH)

strategy = tf.distribute.MirroredStrategy()


def predict(model, dataset, use_multiprocessing=False, workers=1):
    # Outputs
    output_path = os.path.join(BASE_PATH, "masks", "pred")
    output_overlay_path = os.path.join(BASE_PATH, "masks", "pred_overlay")

    # Create folders
    for dir_i in [output_path, output_overlay_path]:
        Path(dir_i).mkdir(parents=True, exist_ok=True)

    # Build dataloaders
    dataloader = DataloaderImages(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Predicting images
    print("Predicting images...")
    for file_ids, batch in tqdm.tqdm(dataloader, total=len(dataloader)):
        predictions = model.predict(batch, use_multiprocessing=use_multiprocessing, workers=workers)

        # Save images in batch
        for file_id, img, img_pred in zip(file_ids, batch, predictions):  # ids, inputs, outputs
            # From 0-1 to 0-255
            img255 = (img.squeeze()).astype(np.uint8)
            pred_img255 = ((img_pred.squeeze() > 0.5) * 255).astype(np.uint8)

            # Convert to PIL
            pil_img = Image.fromarray(img255)
            pil_img_pred = Image.fromarray(pred_img255)

            # Save images
            pil_img_pred.save(os.path.join(output_path, file_id))
            # print(f"Image save! {img_id}")

            # Save overlay
            if SAVE_OVERLAY:
                # Convert to RGBA
                pil_img = pil_img.convert('RGBA')
                pil_img_pred = pil_img_pred.convert('RGBA')

                # Make the background transparent
                pil_img_pred_trans = utils.make_transparent(pil_img_pred, color=(0, 0, 0))
                pil_img_pred_trans.putalpha(75)
                overlaid_img = Image.alpha_composite(pil_img, pil_img_pred_trans)

                # Save overlaid image
                overlaid_img.save(os.path.join(output_overlay_path, file_id))
                # print(f"Overlaid image save! {img_id}")

                # Show images
                # if True:
                #     imshow(overlaid_img)
                #     plt.show()
                asd = 3


def main():
    # Vars
    images_dir = os.path.join(BASE_PATH, "images", "train")
    masks_dir = os.path.join(BASE_PATH, "masks", "train")

    # Get data by matching the filename (no basepath)
    images_files = set([f for f in os.listdir(images_dir)])
    masks_files = set([f for f in os.listdir(masks_dir)])
    files = list(images_files.difference(masks_files))
    print(f"Total images (to predict): {len(files)}")

    # Get model + auxiliar functions
    _, preprocess_fn = utils.get_model(backbone=BACKBONE)

    with strategy.scope():
        # Load model
        print("Loading model...")
        checkpoints_path = os.path.join(BASE_PATH, "runs", RUN_NAME, "models", "checkpoint_last")
        model = tf.keras.models.load_model(filepath=checkpoints_path, compile=False)
        model.compile(loss=bce_jaccard_loss, metrics=[iou_score])
        model.summary()

    # Datasets
    dataset = DatasetImages(BASE_PATH, folder="train", files=files, da_fn=da_ts_fn(TARGET_SIZE, TARGET_SIZE), preprocess_fn=preprocess_fn)

    # Train
    predict(model, dataset)


if __name__ == "__main__":
    main()
    print("Done!")
