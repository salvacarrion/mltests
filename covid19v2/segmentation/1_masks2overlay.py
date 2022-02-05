import os
from pathlib import Path
import random
import tqdm

import segmentation_models as sm
import tensorflow as tf
from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score

from covid19v2.segmentation.dataset import DatasetMasks
from covid19v2.segmentation.da import da_ts_fn
from covid19v2.segmentation import utils, plot

# Variables
TARGET_SIZE = 256
BATCH_SIZE = 128
BACKBONE = "resnet34"
RUN_NAME = "v2"

BASE_PATH = "/home/scarrion/datasets/nn/vision/lungs_masks"
print(BASE_PATH)


def main():
    # Vars
    images_dir = os.path.join(BASE_PATH, "images", "train")
    masks_dir = os.path.join(BASE_PATH, "masks", "train")
    output_path = os.path.join(BASE_PATH, "masks", "masks_overlay")

    # Create folders
    for dir_i in [output_path]:
        Path(dir_i).mkdir(parents=True, exist_ok=True)

    # Get data by matching the filename (no basepath)
    images_files = set([f for f in os.listdir(images_dir)])
    masks_files = set([f for f in os.listdir(masks_dir)])
    files = list(images_files.intersection(masks_files))
    print(f"Total images+masks: {len(files)}")

    # Datasets
    dataset = DatasetMasks(BASE_PATH, folder="train", files=files, da_fn=da_ts_fn(TARGET_SIZE, TARGET_SIZE), preprocess_fn=None)

    # Predicting images
    print("Overlaying images...")
    for i, (image, mask, _, _) in tqdm.tqdm(enumerate(dataset), total=len(dataset)):
        plot.masks2overlay(image=image, mask=mask, output_path=output_path, file_id=dataset.file_ids[i])


if __name__ == "__main__":
    main()
    print("Done!")
