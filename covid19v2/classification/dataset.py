import math
import os

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow import keras


class Dataset:

    def __init__(
            self,
            df,
            images_dir,
            da_fn=None,
            preprocess_fn=None,
            target_size=None,
            memory_map=True,
            tab_input=False,
    ):
        # Get X1
        self.ids = list(df["ImageFile"])
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]

        # Load images in memory (minor speed-up)
        self.memory_map = memory_map
        if self.memory_map:
            self.mem_images = [np.array(Image.open(self.images_fps[i])) for i in range(len(self.ids))]

        self.classes = np.array(df["Prognosis"] == "MILD").astype(np.long)
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

        x = (image,)
        y = (self.classes[i])
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
        dataID2 = []
        for j in range(start, stop):
            _id = self.dataset.ids[j]
            x, y = self.dataset[j]
            dataID2.append(j)
            dataID.append(_id)
            dataX.append(x)
            dataY.append(y)

        # Format data
        X = tf.squeeze(np.stack(dataX, axis=0), axis=1)
        Y = np.stack(dataY, axis=0)

        return X, Y, dataID2

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return math.ceil(len(self.indexes) / self.batch_size)

    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)
