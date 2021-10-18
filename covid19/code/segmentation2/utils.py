import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras_preprocessing.image import ImageDataGenerator

from covid19.code.segmentation.constants import *


def get_aug(test=False):
    if test:
        return ImageDataGenerator(rescale=1./255)
    else:
        return ImageDataGenerator(
            rescale=1./255,
            zoom_range=0.05,
            rotation_range=3,
            shear_range=0.01,
            width_shift_range=[-0.05, +0.05],
            height_shift_range=[-0.05, +0.05],
            brightness_range=[0.95, 1.05],
            fill_mode="constant",
            cval=0,
            horizontal_flip=False)


def get_loader(df, loader_path, test):
    return get_aug(test=test).flow_from_dataframe(
        df,
        loader_path,
        x_col="filepath",
        class_mode=None,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        seed=SEED,
    )


def custom_generator(image_generator, mask_generator):
    train_generator = zip(image_generator, mask_generator)
    for (img, mask) in train_generator:
        yield img, mask


def get_generators(df, test):
    imgs_ds = get_loader(df, loader_path=os.path.join(BASE_PATH, "images256"), test=test)
    masks_ds = get_loader(df, loader_path=os.path.join(BASE_PATH, "masks256"), test=test)
    ds = custom_generator(imgs_ds, masks_ds)
    return ds


def plot_segmentation(img, mask):
    # import copy
    # my_cmap = copy.copy(plt.cm.get_cmap('gray'))  # get a copy of the gray color map
    # my_cmap.set_bad(alpha=0)  # set how the colormap handles 'bad' values
    # mask[mask <= 0] = np.nan

    fig, axes = plt.subplots(1, 1, dpi=150, figsize=(20, 20))
    axes.imshow((img * 255).astype("uint8"), interpolation='none')
    axes.imshow((mask * 255).astype("uint8"), interpolation='none', cmap=None, alpha=0.25)
    plt.tight_layout()
    plt.show()
    asd = 3


def preview_images(generator, n=4):
    for img, masks in generator:
        for i in range(n):
            plot_segmentation(img[i], masks[i])
        break


def get_data(filename="data.csv", filter_masks=True):
    # Read CSV
    df = pd.read_csv(os.path.join(BASE_PATH, filename))

    # Get only images with masks
    if filter_masks:
        # Read available masks
        file_masks = [os.path.split(file)[1] for file in glob.glob(os.path.join(BASE_PATH, "masks256", "*.png"))]

        # Filter CSV with masks
        df = df[df["filepath"].isin(file_masks)]

    # Get sizes
    size_tr, size_val, size_ts = int(SPLITS[0] * len(df)), int(SPLITS[1] * len(df)), int(SPLITS[2] * len(df))

    # Create partitions
    df_train = df[:size_tr]
    df_val = df[size_tr:-size_ts]
    df_test = df[-size_ts:]

    return df_train, df_val, df_test

