import os
import random
import pandas as pd

BASE_PATH = "/home/scarrion/datasets/covid19/lateral"
SPLIT = (0.8, 0.1, 0.1)
random.seed(1234)

# Read CSV
df = pd.read_csv(os.path.join(BASE_PATH, "data.csv"))
subjects = list(set(df["subject"]))

# Ideal sizes
isize_tr, isize_val, isize_ts = int(SPLIT[0] * len(df)), int(SPLIT[1] * len(df)), int(SPLIT[2] * len(df))

# Generate partitions
counter = 1
while True:
    print(f"Shuffling: #{counter}")
    counter += 1

    # Create a copy and shuffle
    rnd_ids = list(subjects)
    random.shuffle(rnd_ids)

    # Split subjects
    ssize_tr, ssize_val, ssize_ts = int(SPLIT[0] * len(subjects)), int(SPLIT[1] * len(subjects)), int(SPLIT[2] * len(subjects))
    tr_ids, val_ids, ts_ids = rnd_ids[:ssize_tr], rnd_ids[ssize_tr:-ssize_ts], rnd_ids[-ssize_ts:]

    # Create partitions
    train_dataset = df[df["subject"].isin(tr_ids)]
    val_dataset = df[df["subject"].isin(val_ids)]
    test_dataset = df[df["subject"].isin(ts_ids)]

    # Check if they're correct
    size_tr, size_val, size_ts = len(train_dataset), len(val_dataset), len(test_dataset)

    # Check sizes of the splits and save
    if isize_val == size_val and isize_ts == size_ts:  # size_tr == isize_tr and
        train_dataset.to_csv(os.path.join(BASE_PATH, "train.csv"), index=False)
        val_dataset.to_csv(os.path.join(BASE_PATH, "val.csv"), index=False)
        test_dataset.to_csv(os.path.join(BASE_PATH, "test.csv"), index=False)
        print("CSVs saved!")
        break
    print("Done!")
