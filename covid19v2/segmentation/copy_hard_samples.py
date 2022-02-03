import os
import shutil
from pathlib import Path

import tqdm

BASE_PATH = "/home/scarrion/datasets/nn/vision/lungs_masks"
print(BASE_PATH)


def main():
    # Vars
    images_dir = os.path.join(BASE_PATH, "images", "train")
    hard_samples_pred_dir = os.path.join(BASE_PATH, "images", "hard_samples_pred")
    hard_samples_dir = os.path.join(BASE_PATH, "images", "hard_samples")

    # Get data by matching the filename (no basepath)
    hard_samples_files = set([f for f in os.listdir(hard_samples_pred_dir)])
    print(f"Total hard samples: {len(hard_samples_files)}")

    # Copy images
    for file in tqdm.tqdm(hard_samples_files, total=len(hard_samples_files)):
        try:
            src_path = os.path.join(images_dir, file)
            dst_path = os.path.join(hard_samples_dir, file)
            shutil.copy(src_path, dst_path)
        except Exception as e:
            print(f"File error: {file}")
            print(e)


if __name__ == "__main__":
    main()
    print("Done!")
