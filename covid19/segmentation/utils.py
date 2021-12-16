import glob
import os

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import pandas as pd


def get_splits(filename, filter_masks=True, masks_path=None, splits=(0.8, 0.1, 0.1)):
    assert sum(splits) == 1.0

    # Read CSV
    df = pd.read_csv(filename)

    # Get only images with masks
    if filter_masks:
        # Read available masks
        file_masks = [os.path.split(file)[1] for file in glob.glob(masks_path)]

        # Filter CSV with masks
        df = df[df["filepath"].isin(file_masks)]

    # Create partitions
    size_tr, size_val, size_ts = int(splits[0] * len(df)), int(splits[1] * len(df)), int(splits[2] * len(df))
    df_train, df_val, df_test = df[:size_tr], df[size_tr:-size_ts], df[-size_ts:]
    return df_train, df_val, df_test


# helper function for data visualization
def plot_4x4(image, mask, original_image=None, original_mask=None):
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


def da_negative(image, **kwargs):
    return 255 - image


# define heavy augmentations
def tr_da_fn(height, width):
    train_transform = [
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(scale_limit=0.10, rotate_limit=7, shift_limit=0.10, border_mode=cv2.BORDER_CONSTANT, p=1.0),
        A.Perspective(scale=(0.025, 0.04), p=0.3),
        A.RandomResizedCrop(height=height, width=width, scale=(0.9, 1.0), p=0.3),

        A.OneOf(
            [
                A.CLAHE(p=1),
                A.RandomBrightness(p=1),
                A.RandomGamma(p=1),
                A.RandomContrast(limit=0.2, p=1.0),
            ],
            p=0.5,
        ),

        A.OneOf(
            [
                A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1.0),
                A.Blur(blur_limit=[2, 3], p=1.0),
                A.GaussNoise(var_limit=(5, 25), p=1.0),
                # A.MotionBlur(blur_limit=3, p=1.0),
            ],
            p=0.5,
        ),

        A.Lambda(image=da_negative, p=0.2),

        A.LongestMaxSize(max_size=max(height, width), always_apply=True),
        A.PadIfNeeded(min_height=height, min_width=width, border_mode=cv2.BORDER_CONSTANT, always_apply=True),
    ]
    return A.Compose(train_transform)


def ts_da_fn(height, width):
    _transform = [
        A.LongestMaxSize(max_size=max(height, width), always_apply=True),
        A.PadIfNeeded(min_height=height, min_width=width, border_mode=cv2.BORDER_CONSTANT, always_apply=True),
    ]
    return A.Compose(_transform)


def preprocessing_fn(custom_fn):
    _transform = [
        A.Lambda(image=custom_fn),
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
