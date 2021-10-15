import os
import numpy as np
import pandas as pd
import collections

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

BASE_PATH = "/home/scarrion/datasets/covid19/all-covid"
df = pd.read_csv(os.path.join(BASE_PATH, "ecvl_bimcv_covid19.tsv"), delimiter='\t')
print(BASE_PATH)

# Remove duplicates
total_rows1 = len(df)
duplicate_names = {item for item, count in collections.Counter(df["filepath"].tolist()).items() if count > 1}
df.drop_duplicates(subset="filepath", keep=False, inplace=True)
total_rows2 = len(df)
print(f"Duplicates: {total_rows1-total_rows2}")

# Remove lines without image
total_rows1 = len(df)
for i, row in df.iterrows():
    fname = row["filepath"].replace("ecvl_bimcv_covid19_preproc_data/", "")

    # Drop row if the file does not exists
    if os.path.exists(os.path.join(BASE_PATH, "images", fname)):
        df.loc[i, "filepath"] = fname
        df.loc[i, "class"] = "no_covid" if "normal" in row["labels"] else "covid"
    else:
        df.drop(i, inplace=True)
total_rows2 = len(df)
print(f"Images not found: {total_rows1-total_rows2}")

# Shuffle
df = df.sample(frac=1)

# Split files
df_train = df[df.split == "training"]
df_val = df[df.split == "validation"]
df_test = df[df.split == "test"]

print(f"Partitions: {len(df_train)+len(df_val)+len(df_test)}")
print(f"- Training: {len(df_train)}")
print(f"\t- Covid: {len(df_train[df_train['class'] == 'covid'])}")
print(f"\t- No covid: {len(df_train[df_train['class'] == 'no_covid'])}")
print(f"")
print(f"- Validation: {len(df_val)}")
print(f"\t- Covid: {len(df_val[df_val['class'] == 'covid'])}")
print(f"\t- No covid: {len(df_val[df_val['class'] == 'no_covid'])}")
print(f"")
print(f"- Test: {len(df_test)}")
print(f"\t- Covid: {len(df_test[df_test['class'] == 'covid'])}")
print(f"\t- No covid: {len(df_test[df_test['class'] == 'no_covid'])}")
print(f"")

# Save CSV
df_train.to_csv(os.path.join(BASE_PATH, "train_data.csv"), index=False)
df_val.to_csv(os.path.join(BASE_PATH, "val_data.csv"), index=False)
df_test.to_csv(os.path.join(BASE_PATH, "test_data.csv"), index=False)
print("Files saved!")

# View images
# df_covid = df[df["class"] == 'covid'][:5]
# df_nocovid = df[df["class"] == 'no_covid'][:5]
# df2 = pd.concat([df_covid, df_nocovid], axis=0)
# df2 = df2.sample(frac=1)
#
# for i, (index, row) in enumerate(df2.iterrows()):
#     img_path = os.path.join(BASE_PATH, "images", row["filepath"])
#     img = mpimg.imread(img_path)
#     img = (img * 255).astype(np.uint8)
#     imgplot = plt.imshow(img, cmap='gray')
#     plt.title("covid" if row["covid"] == True else "No covid")
#     plt.tight_layout()
#     #plt.savefig(os.path.join(BASE_PATH, "tmp", f"{i}.jpg"))
#     plt.show()
# asd = 3
