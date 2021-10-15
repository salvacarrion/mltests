import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalMaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import SGD, Adam
from keras_preprocessing.image import ImageDataGenerator

from segmentation_models import Unet
from segmentation_models import get_preprocessing
from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score

import wandb
from wandb.keras import WandbCallback
wandb.init(project='covid19', entity='salvacarrion')

BASE_PATH = "/home/scarrion/datasets/covid19/80-10-10"
IMAGES_PATHS = "/home/scarrion/datasets/covid19/lung_segmentation/JPEGImages/images256"
MASKS_PATHS = "/home/scarrion/datasets/covid19/lung_segmentation/SegmentationClass/images256"
BACKBONE = 'resnet34'
EPOCHS1 = 10
BATCH_SIZE = 32
IMAGE_SIZE = (256, 256)
SEED = 1234


def get_aug():
    return ImageDataGenerator(
        rescale=1. / 255,
        zoom_range=0.05,
        rotation_range=3,
        shear_range=0.01,
        width_shift_range=[-0.05, +0.05],
        height_shift_range=[-0.05, +0.05],
        brightness_range=[0.95, 1.05],
        fill_mode="constant",
        cval=0,
        horizontal_flip=False,
        validation_split=0.1)


def get_loader(df, loader_path):
    return get_aug().flow_from_dataframe(
        df,
        loader_path,
        x_col="filepath",
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        seed=SEED,
    )


def custom_generator(image_generator, mask_generator):
    train_generator = zip(image_generator, mask_generator)
    for (img, mask) in train_generator:
        yield img, mask

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

# Read CSV
df_train = pd.read_csv(os.path.join(BASE_PATH, "train_data.csv"))
df_val = pd.read_csv(os.path.join(BASE_PATH, "val_data.csv"))
df_test = pd.read_csv(os.path.join(BASE_PATH, "test_data.csv"))
df = pd.concat([df_train, df_val, df_test], ignore_index=True)
df_train = df_val = df_test = None

# Read available masks
file_masks = [os.path.split(file)[1] for file in glob.glob(os.path.join(MASKS_PATHS, "*.png"))]

# Filter CSV with masks
df = df[df["filepath"].isin(file_masks)]

# load your data
train_imgs_ds = get_loader(df, loader_path=os.path.join(IMAGES_PATHS, ""))
train_masks_ds = get_loader(df, loader_path=os.path.join(MASKS_PATHS, ""))
train_ds = custom_generator(train_imgs_ds, train_masks_ds)

# Preview images
preview = 4
for img, masks in train_ds:
    for i in range(preview):
        plot_segmentation(img[0][i], masks[0][i])
    break

# preprocess input
preprocess_input = get_preprocessing(BACKBONE)

# define model
model = Unet(BACKBONE, encoder_weights='imagenet')


my_callbacks = [
    WandbCallback(),
    tf.keras.callbacks.EarlyStopping(patience=10),
    tf.keras.callbacks.ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.h5'),
    tf.keras.callbacks.TensorBoard(log_dir='./logs'),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-5)
]

# Compile the model
model.compile(optimizer=Adam(learning_rate=1e-3), loss=bce_jaccard_loss, metrics=[iou_score])

# train the model on the new data for a few epochs
history1 = model.fit(train_ds, validation_data=test_ds, epochs=EPOCHS1, callbacks=my_callbacks, shuffle=True)

# # we need to recompile the model for these modifications to take effect
# # we use SGD with a low learning rate
# model.compile(optimizer=SGD(learning_rate=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=["accuracy"])
#
# # we train our model again (this time fine-tuning the top 2 inception blocks
# # alongside the top Dense layers
# history2 = model.fit(train_ds, validation_data=test_ds, epochs=EPOCHS2, callbacks=my_callbacks, shuffle=True)
#
# # Evaluate model
# scores = model.evaluate(test_ds)
# print(scores)