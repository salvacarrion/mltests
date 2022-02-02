import collections
import os
import tqdm

import pandas as pd
from covid19v2.preprocess.da import da_resize_pad_fn
import glob
import os

import numpy as np
import seaborn as sns
import tqdm
from PIL import Image

import cv2
import numpy as np
from skimage.exposure import equalize_adapthist

# Variables
BASE_PATH = "/home/scarrion/datasets/nn/vision/covid19v2"
print(BASE_PATH)


def main():
    # Get data
    df = pd.read_excel(os.path.join(BASE_PATH, "data", "data.xls"))
    print(f"Total images: {len(df)}")

    for i, row in tqdm.tqdm(df.iterrows(), total=len(df)):
        # print(f"Augmenting image: {row['filepath']}")
        filename = row["ImageFile"]
        fname, ext = os.path.splitext(filename)  # ext includes "."

        # Open ori image
        ori_img_path = os.path.join(BASE_PATH, "images", "raw", filename)
        ori_image = np.array(Image.open(ori_img_path))

        # Specific preprocessing
        img = equalize_adapthist(ori_image)
        img = img * 255
        img = img.astype(np.uint8)

        # Resize, padding and save
        for image_size in [256, 512]:
            # Augmentation
            da_fn = da_resize_pad_fn(image_size, image_size)

            # Resize and padding
            sample = da_fn(image=img)
            new_image = sample['image']

            # Save image
            new_img_path = os.path.join(BASE_PATH, "images", str(image_size), filename)
            new_image = Image.fromarray(new_image)
            new_image.save(new_img_path)


if __name__ == "__main__":
    main()
    print("Done!")

