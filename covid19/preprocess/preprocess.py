import os
import numpy as np
import pandas as pd
import collections

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

BASE_PATH = "/home/scarrion/datasets/covid19/all-covid"
df = pd.read_csv(os.path.join(BASE_PATH, "ecvl_bimcv_covid19.tsv"), delimiter='\t')
df["symptoms"] = 0
df["infiltrates"] = 0
df["pneumonia"] = 0
df["covid19"] = 0
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
        # df.loc[i, "class"] = "no_covid" if "normal" in row["labels"] else "covid"
        df.loc[i, "infiltrates"] = 1 if "infiltrates" in df.loc[i, "labels"].lower() else 0
        df.loc[i, "pneumonia"] = 1 if "pneumonia" in df.loc[i, "labels"].lower() else 0
        df.loc[i, "covid19"] = 1 if "covid" in df.loc[i, "labels"].lower() else 0
        df.loc[i, "symptoms"] = 1 if (df.loc[i, "infiltrates"] or df.loc[i, "pneumonia"] or df.loc[i, "covid19"]) else 0
    else:
        df.drop(i, inplace=True)
total_rows2 = len(df)
print(f"Images not found: {total_rows1-total_rows2}")

# Shuffle
df = df.sample(frac=1, random_state=1234)

# Split files
df_train = df[df.split == "training"]
df_val = df[df.split == "validation"]
df_test = df[df.split == "test"]

print(f"Partitions: {len(df_train)+len(df_val)+len(df_test)}")
print(f"- Training: {len(df_train)}")
print(f"\t- Infiltrates: {len(df_train[df_train['infiltrates'] == 1])}")
print(f"\t- No Infiltrates: {len(df_train[df_train['infiltrates'] == 0])}")
print(f"\t- Pneumonia: {len(df_train[df_train['pneumonia'] == 1])}")
print(f"\t- No Pneumonia: {len(df_train[df_train['pneumonia'] == 0])}")
print(f"\t- Covid19: {len(df_train[df_train['covid19'] == 1])}")
print(f"\t- No Covid19: {len(df_train[df_train['covid19'] == 0])}")
print(f"")
print(f"- Validation: {len(df_val)}")
print(f"\t- Infiltrates: {len(df_val[df_val['infiltrates'] == 1])}")
print(f"\t- No Infiltrates: {len(df_val[df_val['infiltrates'] == 0])}")
print(f"\t- Pneumonia: {len(df_val[df_val['pneumonia'] == 1])}")
print(f"\t- No Pneumonia: {len(df_val[df_val['pneumonia'] == 0])}")
print(f"\t- Covid19: {len(df_val[df_val['covid19'] == 1])}")
print(f"\t- No Covid19: {len(df_val[df_val['covid19'] == 0])}")
print(f"")
print(f"- Test: {len(df_test)}")
print(f"\t- Infiltrates: {len(df_test[df_test['infiltrates'] == 1])}")
print(f"\t- No Infiltrates: {len(df_test[df_test['infiltrates'] == 0])}")
print(f"\t- Pneumonia: {len(df_test[df_test['pneumonia'] == 1])}")
print(f"\t- No Pneumonia: {len(df_test[df_test['pneumonia'] == 0])}")
print(f"\t- Covid19: {len(df_test[df_test['covid19'] == 1])}")
print(f"\t- No Covid19: {len(df_test[df_test['covid19'] == 0])}")
print(f"")

# Save CSV
df_train.to_csv(os.path.join(BASE_PATH, "train_data.csv"), index=False)
df_val.to_csv(os.path.join(BASE_PATH, "val_data.csv"), index=False)
df_test.to_csv(os.path.join(BASE_PATH, "test_data.csv"), index=False)
df.to_csv(os.path.join(BASE_PATH, "all_data.csv"), index=False)
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
