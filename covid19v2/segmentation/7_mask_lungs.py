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


TARGET_SIZE = 256
PARTITION = "test"
BASE_PATH = "/home/scarrion/datasets/nn/vision/lungs_masks"
print(BASE_PATH)

ALIGNED = True
NEGATIVE_MASK = True


def resize_pad(image, max_size=None):
    # Resize and padding
    max_size = max_size if max_size else max(*image.shape)
    da_fn = da_resize_pad_fn(max_size, max_size)
    image = da_fn(image=image)['image']
    assert image.shape[0] == image.shape[1]
    return image


def expand_bboxes(bboxes, margin_factor):
    new_bboxes = []
    for (x, y, w, h) in bboxes:
        x, y = x * (1.0 - margin_factor), y * (1.0 - margin_factor)
        w, h = w * (1.0 + margin_factor), h * (1.0 + margin_factor)
        new_bboxes.append([x, y, w, h])
    return new_bboxes


def mask_lungs(mask_files, images_dir, output_path, interest_regions):

    # Process images
    for mask_filename in tqdm.tqdm(mask_files, total=len(mask_files)):
        file_id = os.path.split(mask_filename)[1]

        # Load image
        image = Image.open(os.path.join(images_dir, file_id))
        image = np.array(image)

        # Load mask (unaligned always)
        mask = Image.open(mask_filename)
        mask = (force_2d(np.array(mask)) > 0).astype(np.uint8)

        # Align mask (if needed)
        if ALIGNED:
            # Add margin
            interest_region = interest_regions[file_id]
            interest_region = expand_bboxes(interest_region, margin_factor=0.05)

            # image = resize_pad(image)
            x, y, w, h = [int(v * mask.shape[0]) for v in interest_region[0]]
            mask = mask[y:y + h, x:x + w]

        # Resize and pad mask
        mask = resize_pad(mask)
        mask = Image.fromarray(mask)
        mask = mask.resize((TARGET_SIZE, TARGET_SIZE), PIL.Image.LANCZOS)
        mask = (force_2d(np.array(mask)) > 0).astype(np.uint8)
        assert mask.shape[:2] == image.shape[:2]

        # Mask image
        if NEGATIVE_MASK:
            mask_image = np.where(mask.astype(np.bool), 0, image)
        else:
            mask_image = np.where(mask.astype(np.bool), image, 0)

        # Save image
        mask_image = Image.fromarray(mask_image)
        mask_image.save(os.path.join(output_path, file_id))


def main():
    aligned_text = "_aligned" if ALIGNED else ""
    mask_type_text = "_neg" if NEGATIVE_MASK else "_pos"

    # Vars
    images_dir = os.path.join(BASE_PATH, "images", PARTITION, str(TARGET_SIZE) + aligned_text)
    masks_dir = os.path.join(BASE_PATH, "masks", PARTITION, "bucket")  # Unaligned always
    masks_pred_dir = os.path.join(BASE_PATH, "masks", PARTITION, "pred")  # Unaligned always
    output_path = os.path.join(BASE_PATH, "images", PARTITION, str(TARGET_SIZE) + "_mask" + mask_type_text + aligned_text)

    # Create folders
    for dir_i in [output_path]:
        Path(dir_i).mkdir(parents=True, exist_ok=True)

    # Get data by matching the filename (no basepath)
    masks_files = set([os.path.join(masks_dir, f) for f in os.listdir(masks_dir)])
    masks_pred_files = set([os.path.join(masks_pred_dir, f) for f in os.listdir(masks_pred_dir)])
    masks_files = [p for p in list(masks_files.union(masks_pred_files)) if "sub-" not in p]
    print(f"Total masks: {len(masks_files)}")

    # bboxes
    import json
    bboxes_dir = os.path.join(BASE_PATH, "masks", PARTITION, "bboxes")
    df = pd.read_csv(os.path.join(bboxes_dir, "bboxes.csv"))
    interest_regions = {row["filepath"]: json.loads(row["interest_region"]) for i, row in df.iterrows()}

    # Process masks
    mask_lungs(masks_files, images_dir=images_dir, output_path=output_path, interest_regions=interest_regions)


if __name__ == "__main__":
    main()
    print("Done!")
