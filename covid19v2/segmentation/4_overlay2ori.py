import os
import shutil
from pathlib import Path

import tqdm

PARTITION = "train"  # Use for the human-in-the-loop training
BASE_PATH = "/home/scarrion/datasets/nn/vision/lungs_masks"
print(BASE_PATH)


def main():
    # Vars
    base = "masks"
    name_dir = "perfs"
    src_dir = os.path.join(BASE_PATH, PARTITION, base, name_dir)
    dst_dir = os.path.join(BASE_PATH, PARTITION, base, name_dir + "_ori")
    images_dir = os.path.join(BASE_PATH, PARTITION, base, "pred")

    # Create folders
    for dir_i in [dst_dir]:
        Path(dir_i).mkdir(parents=True, exist_ok=True)

    # Get data by matching the filename (no basepath)
    files = set([f for f in os.listdir(src_dir)])
    print(f"Total samples: {len(files)}")

    # Copy images
    for file in tqdm.tqdm(files, total=len(files)):
        try:
            src_path = os.path.join(images_dir, file)
            dst_path = os.path.join(dst_dir, file)
            shutil.copy(src_path, dst_path)
        except Exception as e:
            print(f"File error: {file}")
            print(e)


if __name__ == "__main__":
    main()
    print("Done!")
