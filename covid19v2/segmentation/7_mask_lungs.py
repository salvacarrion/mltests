import os
import shutil
from pathlib import Path

import tqdm

import glob
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tqdm
from matplotlib import pyplot as plt
from covid19v2.preprocess.da import da_resize_pad_fn
from PIL import Image
import PIL
from covid19v2.segmentation.utils import force_2d


TARGET_SIZE = 512

BASE_PATH = "/home/scarrion/datasets/nn/vision/lungs_masks"
print(BASE_PATH)


def resize_pad(image, max_size=None):
    # Resize and padding
    max_size = max_size if max_size else max(*image.shape)
    da_fn = da_resize_pad_fn(max_size, max_size)
    image = da_fn(image=image)['image']
    assert image.shape[0] == image.shape[1]
    return image


def mask_lungs(mask_files, images_dir, output_path):

    # Process images
    for mask_filename in tqdm.tqdm(mask_files, total=len(mask_files)):
        file_id = os.path.split(mask_filename)[1]

        # Load image
        image = Image.open(os.path.join(images_dir, file_id))
        image = np.array(image)
        # image = resize_pad(image)

        # Load mask
        mask = Image.open(mask_filename)
        mask = mask.resize((TARGET_SIZE, TARGET_SIZE), PIL.Image.NEAREST)
        mask = (force_2d(np.array(mask)) > 0).astype(np.uint8)
        # mask = resize_pad(mask, max_size=image.shape[0])
        assert mask.shape[:2] == image.shape[:2]

        # Mask image
        mask_image = np.where(mask.astype(np.bool), image, 0)
        mask_image = Image.fromarray(mask_image)
        mask_image.save(os.path.join(output_path, file_id))


def main():
    # Vars
    images_dir = os.path.join(BASE_PATH, "images", str(TARGET_SIZE))
    masks_dir = os.path.join(BASE_PATH, "masks", "train")
    masks_pred_dir = os.path.join(BASE_PATH, "masks", "pred")
    output_path = os.path.join(BASE_PATH, "images", str(TARGET_SIZE) + "_mask_pos")

    # Create folders
    for dir_i in [output_path]:
        Path(dir_i).mkdir(parents=True, exist_ok=True)

    # Get data by matching the filename (no basepath)
    masks_files = set([os.path.join(masks_dir, f) for f in os.listdir(masks_dir)])
    masks_pred_files = set([os.path.join(masks_pred_dir, f) for f in os.listdir(masks_pred_dir)])
    masks_files = [p for p in list(masks_files.union(masks_pred_files)) if "sub-" not in p]
    print(f"Total masks: {len(masks_files)}")

    # Process masks
    mask_lungs(masks_files, images_dir=images_dir, output_path=output_path)


if __name__ == "__main__":
    main()
    print("Done!")
