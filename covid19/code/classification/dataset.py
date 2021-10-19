import os
import math

import tensorflow as tf
from tensorflow import keras

from PIL import Image
import numpy as np


class Dataset:

    def __init__(
            self,
            df,
            images_dir,
            da_fn=None,
            preprocess_fn=None,
            target_size=None,
            memory_map=True,
    ):
        # Get X
        self.ids = list(df["filepath"])
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]

        # Get y
        self.target_infiltrates = list(df["infiltrates"])
        self.target_pneumonia = list(df["pneumonia"])
        self.target_covid19 = list(df["covid19"])

        # Load images in memory (minor speed-up)
        self.memory_map = memory_map
        if self.memory_map:
            self.mem_images = [np.array(Image.open(self.images_fps[i])) for i in range(len(self.ids))]

        self.da_fn = da_fn
        self.preprocess_fn = preprocess_fn
        self.target_size = target_size

    def __getitem__(self, i, show_originals=False):
        # Read/Load data
        if self.memory_map:
            image = self.mem_images[i]
        else:
            image = np.array(Image.open(self.images_fps[i]))

        # # Keep originals (just for visualization)
        original_image = np.array(image) if show_originals else None

        # apply augmentations
        if self.da_fn:
            sample = self.da_fn(image=image)
            image = sample['image']

        # Convert images to RGB
        image = np.stack((image,) * 3, axis=-1)  # Grayscale to RGB

        # apply preprocessing
        if self.preprocess_fn:
            image = self.preprocess_fn(image)

        # Check shapes
        assert image.shape[:2] == self.target_size

        # Get outputs
        if show_originals:
            x = (image, original_image)
            y = None
        else:
            x = (image,)
            y = (self.target_infiltrates[i], self.target_pneumonia[i], self.target_covid19[i])
        return x, y

    def __len__(self):
        return len(self.ids)


class Dataloader(keras.utils.Sequence):

    def __init__(self, dataset, batch_size=1, shuffle=False, predict=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))
        self.predict = predict
        self.on_epoch_end()

    def __getitem__(self, i):

        # collect batch data
        start = i * self.batch_size
        stop = min((i + 1) * self.batch_size, len(self.dataset))

        dataID, dataX, dataY = [], [], []
        for j in range(start, stop):
            _id = self.dataset.ids[j]
            x, y = self.dataset[j]
            dataID.append(_id)
            dataX.append(x)
            dataY.append(y)

        # Transpose list of lists
        # X = np.stack(data, axis=0)
        X = tuple([np.stack(samples, axis=0) for samples in zip(*dataX)])
        Y = tuple([np.stack(samples, axis=0) for samples in zip(*dataY)])
        return X, Y

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return math.ceil(len(self.indexes) / self.batch_size)

    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)
