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

BASE_PATH = "/Users/salvacarrion/Documents/Programming/datasets/nn/vision/covid19/front"
SAVE_PATH = "/Users/salvacarrion/Documents/Programming/datasets/nn/vision/covid19/front_augmented"


def main(repeats, image_size):
    # Augmentation
    da_fn = offline_da_fn(image_size, image_size)

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
                new_name = f"{fname}__a{j+1}{ext}"
                new_row["filepath"] = new_name

                # Open image
                img_path = os.path.join(BASE_PATH, "images", f"images512", filename)
                ori_image = np.array(Image.open(img_path))
                # if j == 0 and i <= 5:  # Preview (debugging)
                #     Image.fromarray(ori_image).show()

                # Perform augmentation
                sample = da_fn(image=ori_image)
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


if __name__ == "__main__":
    main(repeats=15, image_size=256)
