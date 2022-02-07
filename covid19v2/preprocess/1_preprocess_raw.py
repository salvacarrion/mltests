import collections
import os

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
BASE_PATH = "/home/scarrion/datasets/nn/vision/lungs_masks"
print(BASE_PATH)


def main():
    # Get data
    df = pd.read_excel(os.path.join(BASE_PATH, "data", "data.xls"))
    print(f"Total images: {len(df)}")

    for i, row in tqdm.tqdm(df.iterrows(), total=len(df)):
        # print(f"Augmenting image: {row['filepath']}")
        filename = row["ImageFile"]

        # Open ori image
        ori_img_path = os.path.join(BASE_PATH, "images", "raw", filename)
        pil_img = Image.open(ori_img_path)

        # Specific preprocessing
        img = np.array(pil_img).astype(np.float)
        img = exposure.rescale_intensity(img)  # x => [0.0, 1.0]
        img = exposure.equalize_adapthist(img)
        img = (img*255).astype(np.uint8)

        # Resize and padding
        max_size = max(*img.shape)
        da_fn = da_resize_pad_fn(max_size, max_size)
        img = da_fn(image=img)['image']

        # Resize, padding and save
        for image_size in [256, 512]:
            pil_img = Image.fromarray(img)
            pil_img = pil_img.resize((image_size, image_size), PIL.Image.LANCZOS)

            # Save image
            new_img_path = os.path.join(BASE_PATH, "images", str(image_size), filename)
            pil_img.save(new_img_path)


if __name__ == "__main__":
    main()
    print("Done!")

