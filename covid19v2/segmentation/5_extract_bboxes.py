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


def draw_bounding_boxes(img, boxes, scaling=1.0):
    new_img = np.array(img)
    new_img = cv2.cvtColor(new_img, cv2.COLOR_GRAY2RGB)

    # Draw bounding boxes
    for box in boxes:
        top_left = (int(box[0] * scaling), int(box[1] * scaling))
        bottom_right = (int(box[0] * scaling + box[2] * scaling), int(box[1] * scaling + box[3] * scaling))
        cv2.rectangle(new_img, top_left, bottom_right, (0, 255, 0), 2)

    return new_img


def get_all_masks_files(masks_dir, pred_masks_dir):
    masks_files = [file for file in
                   glob.glob(os.path.join(masks_dir, "*.png"))]
    pred_masks_files = [file for file in
                        glob.glob(os.path.join(pred_masks_dir, "*.png"))]
    return masks_files, pred_masks_files


def process_masks(files, output_path, apply_otsu=True, draw_contours=True, threshold_area=50, scaling=256):
    rows = []
    for i, file in tqdm.tqdm(enumerate(files), total=len(files)):
        fname = os.path.split(file)[1]

        # Read mask
        mask = read_img(file)

        # # Show image (debug)
        # show_img(img, cmap="gray")

        # Binarize
        if apply_otsu:
            mask = binarize(mask)

        # Get bounding boxes (relative bboxes)
        bboxes, concat_bboxes = get_bounding_boxes(mask, threshold_area=threshold_area, scaling=scaling)

        # Store bboxes
        rows.append({"filepath": fname, "bboxes": bboxes, "interest_region": concat_bboxes})

        # Draw bounding boxes
        if draw_contours:
            painted_img = draw_bounding_boxes(mask, concat_bboxes, scaling=scaling)

            # Save image
            cv2.imwrite(os.path.join(output_path, fname), painted_img)

    # Save contours
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_path, "bboxes.csv"), index=False)
    print("CSV saved!")


def main():
    # Vars
    masks_dir = os.path.join(BASE_PATH, "masks", "train")
    masks_pred_dir = os.path.join(BASE_PATH, "masks", "pred")
    output_path = os.path.join(BASE_PATH, "masks", "bboxes")

    # Create folders
    for dir_i in [output_path]:
        Path(dir_i).mkdir(parents=True, exist_ok=True)

    # Get data by matching the filename (no basepath)
    masks_files = set([os.path.join(masks_dir, f) for f in os.listdir(masks_dir)])
    masks_pred_files = set([os.path.join(masks_pred_dir, f) for f in os.listdir(masks_pred_dir)])
    files = list(masks_files.union(masks_pred_files))
    print(f"Total samples: {len(files)}")

    # Process masks
    process_masks(files, output_path)


if __name__ == "__main__":
    main()
    print("Done!")
