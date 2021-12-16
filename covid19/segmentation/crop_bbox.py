import glob
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tqdm
from matplotlib import pyplot as plt


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


def main(base_path=".", target_size=512, margin_factor=0.05):
    # path = os.path.join(base_path, "masks256_pred", "sub-S10880_ses-E18932_run-1.png")  # Testing

    # Create output not exists
    images_output_path = os.path.join(base_path, f"images{target_size}_crop")
    Path(images_output_path).mkdir(parents=True, exist_ok=True)

    # Get files
    raw_images_dir = os.path.join(base_path, "images_raw")
    df = pd.read_csv(os.path.join(base_path, "bboxes.csv"))

    # Process masks
    for i, row in tqdm.tqdm(df.iterrows(), total=len(df)):
        bboxes, interest_region = json.loads(row["bboxes"]), json.loads(row["interest_region"])
        file = row["filepath"]
        fname = os.path.split(file)[1]

        # Read image
        img = read_img(os.path.join(raw_images_dir, file))
        # show_img(img)

        # Pad image
        max_side = max(*img.shape)
        img = pad_img(img)
        # show_img(img)

        # Expand bboxes
        bboxes = expand_bboxes(bboxes, margin_factor=margin_factor)
        interest_region = expand_bboxes(interest_region, margin_factor=margin_factor)

        # # Draw bboxes
        # img_bboxes = draw_bounding_boxes(img, bboxes, scaling=max_side)
        # show_img(img_bboxes)

        # Crop
        x, y, w, h = [int(v * max_side) for v in interest_region[0]]
        img = img[y:y + h, x:x + w]
        # show_img(img)

        # Pad image (again)
        img = pad_img(img)
        # show_img(img)

        # Resize
        img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)
        # show_img(img)

        # Save image
        cv2.imwrite(os.path.join(images_output_path, fname), img)
        asdasd = 3

    print("Done!")


if __name__ == "__main__":
    BASE_PATH = "/home/scarrion/datasets/covid19/front"

    # Run script
    main(base_path=BASE_PATH)
