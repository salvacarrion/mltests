import os
import math

import tensorflow as tf
from tensorflow import keras

from PIL import Image
import numpy as np


# classes for data loading and preprocessing
class Dataset:
    """Covid19 Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)

    """

    def __init__(
            self,
            df,
            imgs_dir,
            masks_dir,
            augmentation=None,
            preprocessing=None,
            target_size=None,
            show_originals=False,
            memory_map=True,
    ):
        self.ids = list(df["filepath"])
        self.images_fps = [os.path.join(imgs_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]

        # Load images in memory (minor speed-up)
        self.memory_map = memory_map
        if self.memory_map:
            self.mem_images = [np.array(Image.open(self.images_fps[i])) for i in range(len(self.ids))]
            self.mem_masks = [np.array(Image.open(self.masks_fps[i]))[..., 0] for i in range(len(self.ids))]

        # convert str names to class values on masks
        self.CLASSES = ["lungs"]

        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.target_size = target_size
        self.show_originals = show_originals

    def __getitem__(self, i):
        # Read/Load data
        if self.memory_map:
            image = self.mem_images[i]
            mask = self.mem_masks[i]
        else:
            image = np.array(Image.open(self.images_fps[i]))
            mask = np.array(Image.open(self.masks_fps[i]))[..., 0]

        # # Keep originals (just for visualization)
        if self.show_originals:
            original_image = np.array(image)
            original_mask = np.array(mask)

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # Convert images
        image = np.stack((image,)*3, axis=-1)  # Grayscale to RGB
        mask = mask.astype(np.bool).astype(np.float32)

        # Check shapes
        assert image.shape[:2] == self.target_size
        assert mask.shape[:2] == self.target_size

        if self.show_originals:
            return image, mask, original_image, original_mask
        else:
            return image, mask

    def __len__(self):
        return len(self.ids)


class Dataloder(keras.utils.Sequence):
    """Load data from dataset and form batches

    Args:
        dataset: instance of Dataset class for image loading and preprocessing.
        batch_size: Integer number of images in batch.
        shuffle: Boolean, if `True` shuffle image indexes each epoch.
    """

    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))

        self.on_epoch_end()

    def __getitem__(self, i):

        # collect batch data
        start = i * self.batch_size
        stop = min((i + 1) * self.batch_size, len(self.dataset))
        data = []
        for j in range(start, stop):
            data.append(self.dataset[j])

        # Transpose list of lists
        X, y = tuple([np.stack(samples, axis=0) for samples in zip(*data)])
        return X, y

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return math.ceil(len(self.indexes) / self.batch_size)

    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)