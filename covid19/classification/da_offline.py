import glob
import os
import pandas as pd
import numpy as np
import tqdm

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
sns.set()

from covid19.classification.da import offline_da_fn

BASE_PATH = "/Users/salvacarrion/Documents/Programming/datasets/nn/vision/covid19/front-cropped"
SAVE_PATH = "/Users/salvacarrion/Documents/Programming/datasets/nn/vision/covid19/512x512"


def process_splits(repeats, image_size):
    # Augmentation
    da_fn_noaugment = offline_da_fn(image_size, image_size, augment=False)
    da_fn_augment = offline_da_fn(image_size, image_size, augment=True)

    # Get data
    df_train = pd.read_csv(os.path.join(BASE_PATH, "data", "train.csv"))
    df_val = pd.read_csv(os.path.join(BASE_PATH, "data", "val.csv"))
    df_test = pd.read_csv(os.path.join(BASE_PATH, "data", "test.csv"))

    # Name splits
    splits = [(df_train, "train"), (df_val, "val"), (df_test, "test")]

    # Print total augmentations
    total_augmentations = (len(df_train)+len(df_val)+len(df_test))*repeats
    print(f"Total augmentation: {total_augmentations}")

    # Walk through splits
    for split, split_name in splits:
        print(f"Augmenting split: {split_name}")

        rows = []
        for i, row in tqdm.tqdm(split.iterrows(), total=len(split)):
            # print(f"Augmenting image: {row['filepath']}")
            filename = row["filepath"]
            fname, ext = os.path.splitext(filename)  # ext includes "."

            # Augment n times
            for j in range(repeats):
                # print(f"\t- Augmentation #{j+1}")
                new_row = dict(row)  # Get row

                # Form new name
                new_name = f"{fname}__r{j}{ext}"
                new_row["filepath"] = new_name

                # Open image
                img_path = os.path.join(BASE_PATH, "images", f"images512", filename)
                ori_image = np.array(Image.open(img_path))
                # if j == 0 and i <= 5:  # Preview (debugging)
                #     Image.fromarray(ori_image).show()
                if j==0:
                    sample = da_fn_noaugment(image=ori_image)
                else:
                    # Perform augmentation
                    sample = da_fn_augment(image=ori_image)
                image = sample['image']

                # Save image
                img_savepath = os.path.join(SAVE_PATH, "images", f"images{image_size}", new_name)
                image = Image.fromarray(image)
                image.save(img_savepath)

                # Add new row
                rows.append(new_row)

        # Create new CSV
        new_split = pd.DataFrame(rows)  # columns=split.columns from dict

        # Save split
        savepath = os.path.join(SAVE_PATH, "data", f"{split_name}.csv")
        new_split.to_csv(savepath, index=False)
        print("File saved!")

    print("Done!")


def process_masks(image_size):
    # Augmentation
    da_fn = offline_da_fn(image_size, image_size, augment=False)

    # Walk through splits
    masks_files = list(glob.glob(os.path.join(BASE_PATH, "masks", "masks256_manual", "*.png")))
    for filename in tqdm.tqdm(masks_files, total=len(masks_files)):
        head, tail = os.path.split(filename)

        # Open image
        ori_image = np.array(Image.open(filename))

        # Perform augmentation
        sample = da_fn(image=ori_image)
        image = sample['image']

        # Save image
        img_savepath = os.path.join(SAVE_PATH, "masks", f"masks{image_size}", tail)
        image = Image.fromarray(image)
        image.save(img_savepath)

    print("Done!")


if __name__ == "__main__":
    process_splits(repeats=16, image_size=512)
    # process_masks(image_size=512)
