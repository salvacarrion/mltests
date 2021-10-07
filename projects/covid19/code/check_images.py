import os
import numpy as np
import pandas as pd
import collections

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


BASE_PATH = "/home/scarrion/datasets/covid19"
df = pd.read_csv(os.path.join(BASE_PATH, "ecvl_bimcv_covid19.tsv"), delimiter='\t')
df["covid"] = None

# Remove duplicates
df.drop_duplicates(subset="filepath", keep=False, inplace=True)

# Remove lines without image
for i, row in df.iterrows():
    fname = row["filepath"].replace("ecvl_bimcv_covid19_preproc_data/", "")

    # Drop row if the file does not exists
    if os.path.exists(os.path.join(BASE_PATH, "images", fname)):
        df.loc[i, "filepath"] = fname
        df.loc[i, "covid"] = False if "normal" in row["labels"] else True
        df.loc[i, "split2"] = "test" if df.loc[i, "split"] == "test" else "training"
    else:
        df.drop(i, inplace=True)

# Shuffle
df = df.sample(frac=1)

# Save CSV
df.to_csv(os.path.join(BASE_PATH, "data.csv"), index=False)

# View images
df_covid = df[df["covid"] == True][:5]
df_nocovid = df[df["covid"] == False][:5]
df2 = pd.concat([df_covid, df_nocovid], axis=0)
df2 = df2.sample(frac=1)

for i, (index, row) in enumerate(df2.iterrows()):
    img_path = os.path.join(BASE_PATH, "images", row["filepath"])
    img = mpimg.imread(img_path)
    img = (img * 255).astype(np.uint8)
    imgplot = plt.imshow(img, cmap='gray')
    plt.title("covid" if row["covid"] == True else "No covid")
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_PATH, "tmp", f"{i}.jpg"))
    plt.show()
asd = 3