import collections
import os
from pathlib import Path

import PIL.Image
import tqdm

import pandas as pd
from covid19v2.preprocess.da import da_resize_pad_fn
import glob
import os

import numpy as np
import seaborn as sns
import tqdm
from PIL import Image
from skimage import exposure

import cv2
import numpy as np
from skimage.exposure import equalize_adapthist

# Variables
IMAGE_SIZES = [256, 512]
PARTITION = "test"
BASE_PATH = "/home/scarrion/datasets/nn/vision/lungs_masks"
print(BASE_PATH)


def main():
    # Get data
    df = pd.read_excel(os.path.join(BASE_PATH, "data", f"{PARTITION}.xls"))
    print(f"Total images: {len(df)}")

    for i, row in tqdm.tqdm(df.iterrows(), total=len(df)):
        # print(f"Augmenting image: {row['filepath']}")
        filename = row["ImageFile"]

        # Open ori image
        ori_img_path = os.path.join(BASE_PATH, "images", PARTITION, "raw_aligned", filename)
        pil_img = Image.open(ori_img_path)
        img = np.array(pil_img)

        # Resize, padding and save
        for image_size in IMAGE_SIZES:
            savepath = os.path.join(BASE_PATH, "images", PARTITION, str(image_size) + "_aligned")
            Path(savepath).mkdir(parents=True, exist_ok=True)

            # Resize image
            pil_img = Image.fromarray(img)
            pil_img = pil_img.resize((image_size, image_size), PIL.Image.LANCZOS)

            # Save image
            pil_img.save(os.path.join(savepath, filename))


if __name__ == "__main__":
    main()
    print("Done!")

