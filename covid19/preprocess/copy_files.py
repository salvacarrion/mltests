import os
import shutil

import pandas as pd
import tqdm

BASE_PATH = "/home/scarrion/datasets/covid19/lateral"

# Read file
df = pd.read_csv(os.path.join(BASE_PATH, "data.csv"))

# Get file names
filenames = list(df["filepath"])

# Copy files
for fname in tqdm.tqdm(filenames, total=len(filenames)):
    shutil.copyfile(os.path.join(BASE_PATH, "images", fname), os.path.join(BASE_PATH, "images_raw", fname))
