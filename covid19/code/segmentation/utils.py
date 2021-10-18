import glob
import os

import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras_preprocessing.image import ImageDataGenerator

import albumentations as A

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


# helper function for data visualization
def visualize(image, mask, original_image=None, original_mask=None):
    fontsize = 18
    plt.gray()

    if original_image is None and original_mask is None:
        f, ax = plt.subplots(2, 1, figsize=(8, 8))
        ax[0].imshow(image)
        ax[1].imshow(mask)
    else:
        f, ax = plt.subplots(2, 2, figsize=(8, 8))

        ax[0, 0].imshow(original_image)
        ax[0, 0].set_title('Original image', fontsize=fontsize)

        ax[0, 1].imshow(original_mask)
        ax[0, 1].set_title('Original mask', fontsize=fontsize)

        ax[1, 0].imshow(image)
        ax[1, 0].set_title('Transformed image', fontsize=fontsize)

        ax[1, 1].imshow(mask)
        ax[1, 1].set_title('Transformed mask', fontsize=fontsize)
    plt.show()


def negative(image, **kwargs):
    return 255-image

# define heavy augmentations
def get_training_augmentation(target_size=(256, 256)):
    height, width = target_size
    train_transform = [
        A.HorizontalFlip(p=0.5),
        A.Perspective(scale=(0.025, 0.04), p=0.2),
        A.ShiftScaleRotate(scale_limit=0.10, rotate_limit=7, shift_limit=0.10, border_mode=cv2.BORDER_CONSTANT, p=0.8),
        A.RandomResizedCrop(height=height, width=width, scale=(0.9, 1.0), p=0.3),

        A.OneOf(
            [
                A.CLAHE(p=1),
                A.RandomBrightness(p=1),
                A.RandomGamma(p=1),
                A.RandomContrast(limit=0.2, p=1.0),
            ],
            p=0.8,
        ),

        A.OneOf(
            [
                A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1.0),
                A.Blur(blur_limit=[2, 3], p=1.0),
                A.GaussNoise(var_limit=(5, 25), p=1.0),
                # A.MotionBlur(blur_limit=3, p=1.0),
            ],
            p=1.0,
        ),

        A.Lambda(image=negative, p=0.2),

        A.LongestMaxSize(max_size=max(height, width), always_apply=True),
        A.PadIfNeeded(min_height=height, min_width=width, border_mode=cv2.BORDER_CONSTANT, always_apply=True),
    ]
    return A.Compose(train_transform)


def get_validation_augmentation(target_size=(256, 256)):
    height, width = target_size

    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        A.LongestMaxSize(max_size=max(height, width), always_apply=True),
        A.PadIfNeeded(min_height=height, min_width=width, border_mode=cv2.BORDER_CONSTANT, always_apply=True),
    ]
    return A.Compose(test_transform)


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        A.Lambda(image=preprocessing_fn),
    ]
    return A.Compose(_transform)


def plot_hist(history, title="", savepath=None, suffix=""):
    # Plot training & validation iou_score values
    plt.figure(figsize=(30, 5))

    # Set title
    plt.title(title)

    plt.subplot(121)
    plt.plot(history.history['iou_score'])
    plt.plot(history.history['val_iou_score'])
    plt.title('Model IoU')
    plt.ylabel('IoU score')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')

    # Save figures
    if savepath:
        plt.savefig(os.path.join(savepath, f"iou{suffix}.pdf"))
        plt.savefig(os.path.join(savepath, f"iou{suffix}.png"))

    # Plot training & validation loss values
    plt.subplot(122)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')

    # Save figures
    if savepath:
        plt.savefig(os.path.join(savepath, f"loss{suffix}.pdf"))
        plt.savefig(os.path.join(savepath, f"loss{suffix}.png"))

    # Show
    plt.show()
