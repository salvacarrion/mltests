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

from covid19v2.segmentation.dataset import DatasetMasks
from covid19v2.segmentation.da import da_ts_fn
from covid19v2.segmentation import utils

# Variables
TARGET_SIZE = 256
BATCH_SIZE = 128
BACKBONE = "resnet34"
RUN_NAME = "v2"

BASE_PATH = "/home/scarrion/datasets/nn/vision/lungs_masks"
print(BASE_PATH)


def masks2overlay(image, mask, output_path, file_id):
    assert image.shape[:2] == mask.shape[:2]

    # From 0-1 to 0-255
    image = image.astype(np.uint8)
    mask = ((mask > 0) * 255).astype(np.uint8)

    # Convert to PIL
    pil_img = Image.fromarray(image)
    pil_mask = Image.fromarray(mask)

    # Convert to RGBA
    pil_img = pil_img.convert('RGBA')
    pil_mask = pil_mask.convert('RGBA')

    # Make the background transparent
    pil_mask_alpha = utils.make_transparent(pil_mask, color=(0, 0, 0))
    pil_mask_alpha.putalpha(75)

    # Overlay mask and save image
    overlaid_img = Image.alpha_composite(pil_img, pil_mask_alpha)
    overlaid_img.save(os.path.join(output_path, file_id))


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
        masks2overlay(image=image, mask=mask, output_path=output_path, file_id=dataset.file_ids[i])


if __name__ == "__main__":
    main()
    print("Done!")
