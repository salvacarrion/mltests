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
        self.ids = list(df["filepath"])
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]

        # Get X2
        self.tab = None
        if tab_input:
            trans_table = {'M': 0, 'F': 1}
            self.tab = [(trans_table[x1], x2) for x1, x2 in zip(df["gender"], df["age"])]

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
        self.tab_input = tab_input

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
            x = (image,) if not self.tab_input else (image, np.array(self.tab[i], dtype=np.float32))
            y = (self.target_infiltrates[i], self.target_pneumonia[i], self.target_covid19[i])
        return x, y

    def __len__(self):
        return len(self.ids)


class Dataloader(keras.utils.Sequence):

    def __init__(self, dataset, batch_size=1, shuffle=False, predict=False, single_output_idx=None,
                 multiple_inputs=False, add_normal_cls=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))
        self.predict = predict
        self.single_output_idx = single_output_idx
        self.multiple_inputs = multiple_inputs
        self.add_normal_cls = add_normal_cls
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
        if self.multiple_inputs:
            X = [x for x in zip(*dataX)]
            X = [np.stack(x, axis=0) for x in X]
        else:
            X = tf.squeeze(np.stack(dataX, axis=0), axis=1)

        # Stack list
        Y = np.stack(dataY, axis=0)

        # Add normal
        if self.add_normal_cls:
            Y_normal = np.expand_dims((np.sum(Y, axis=1) == 0).astype(np.int64), axis=1)
            Y = np.concatenate([Y, Y_normal], axis=1)

        # Number of outputs
        Y = Y if self.single_output_idx is None else Y[:, self.single_output_idx]
        return X, Y  # infiltrates, pneumonia, covid19, (normal)

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return math.ceil(len(self.indexes) / self.batch_size)

    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)
