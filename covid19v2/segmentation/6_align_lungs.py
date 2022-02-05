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
import json
from PIL import Image

from covid19v2.segmentation.da import da_ts_fn


# Vars
TARGET_SIZE = 256

BASE_PATH = "/home/scarrion/datasets/nn/vision/lungs_masks"
print(BASE_PATH)



def read_img(filename):
    img = cv2.imread(filename, 0)
    img = img.astype(np.uint8)
    return img


def show_img(img, cmap="gray", title=""):
    plt.imshow(img, cmap=cmap)
    plt.title(title)
    plt.show()


def binarize(img):
    thr_val, img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
    return img


def get_bounding_boxes(img, threshold_area=1, scaling=1.0):
    # Find contours
    ctrs, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Clean areas
    cleaned_ctrs = []
    for ctr in ctrs:
        area = cv2.contourArea(ctr)
        if area > threshold_area:
            cleaned_ctrs.append(ctr)

    # Bboxes
    bboxes = []
    ctrs = cleaned_ctrs
    for ctr in ctrs:
        x, y, w, h = cv2.boundingRect(ctr)
        values = [v / scaling for v in [x, y, w, h]]
        bboxes.append(values)

    # Concat bboxes
    ctrs = np.concatenate(ctrs)
    x, y, w, h = cv2.boundingRect(ctrs)
    concat_bboxes = [[v / scaling for v in [x, y, w, h]]]

    return bboxes, concat_bboxes


def draw_bounding_boxes(img, boxes, scaling=1.0, width=2):  # 256=>w=2; 512=>w=5; large=>w=10
    new_img = np.array(img)
    new_img = cv2.cvtColor(new_img, cv2.COLOR_GRAY2RGB)

    # Draw bounding boxes
    for box in boxes:
        top_left = (int(box[0] * scaling), int(box[1] * scaling))
        bottom_right = (int(box[0] * scaling + box[2] * scaling), int(box[1] * scaling + box[3] * scaling))
        cv2.rectangle(new_img, top_left, bottom_right, (0, 255, 0), width)

    return new_img


def pad_img(img):
    max_side = max(*img.shape)
    img_padded = np.zeros((max_side, max_side), np.uint8)
    ax, ay = (max_side - img.shape[1]) // 2, (max_side - img.shape[0]) // 2
    img_padded[ay:img.shape[0] + ay, ax:ax + img.shape[1]] = img
    return img_padded


def expand_bboxes(bboxes, margin_factor):
    new_bboxes = []
    for (x, y, w, h) in bboxes:
        x, y = x * (1.0 - margin_factor), y * (1.0 - margin_factor)
        w, h = w * (1.0 + margin_factor), h * (1.0 + margin_factor)
        new_bboxes.append([x, y, w, h])
    return new_bboxes


def get_all_masks_files(masks_dir, pred_masks_dir):
    masks_files = [file for file in
                   glob.glob(os.path.join(masks_dir, "*.png"))]
    pred_masks_files = [file for file in
                        glob.glob(os.path.join(pred_masks_dir, "*.png"))]
    return masks_files, pred_masks_files


def align_lungs(interest_regions, files, images_dir, output_path, margin_factor=0.05):
    # Process images
    for file_id in tqdm.tqdm(files, total=len(files)):
        # Add margin
        interest_region = interest_regions[file_id]
        interest_region = expand_bboxes(interest_region, margin_factor=margin_factor)

        # Read image
        image = read_img(os.path.join(images_dir, file_id))
        assert image.shape[0] == image.shape[1]

        # Crop
        x, y, w, h = [int(v * image.shape[0]) for v in interest_region[0]]
        image = image[y:y + h, x:x + w]

        # Resize + pad
        da_fn = da_ts_fn(TARGET_SIZE, TARGET_SIZE)
        image = da_fn(image=image)['image']

        # Save image
        pil_img_pred = Image.fromarray(image)
        pil_img_pred.save(os.path.join(output_path, file_id))


def main():
    # Vars
    images_dir = os.path.join(BASE_PATH, "images", "2048")
    bboxes_dir = os.path.join(BASE_PATH, "masks", "bboxes")
    output_path = os.path.join(BASE_PATH, "images", f"aligned{TARGET_SIZE}")

    # Get images
    files = set([f for f in os.listdir(images_dir)])

    # Create folders
    for dir_i in [output_path]:
        Path(dir_i).mkdir(parents=True, exist_ok=True)

    # bboxes
    df = pd.read_csv(os.path.join(bboxes_dir, "bboxes.csv"))
    interest_regions = {row["filepath"]: json.loads(row["interest_region"]) for i, row in df.iterrows()}

    # Align lungs
    align_lungs(interest_regions, files, images_dir, output_path)


if __name__ == "__main__":
    main()
    print("Done!")
