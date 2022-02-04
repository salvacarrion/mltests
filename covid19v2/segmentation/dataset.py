import math
import os

import numpy as np
from PIL import Image
from tensorflow import keras


def force_2d(img):
    if len(img.shape) == 2:
        return img
    elif len(img.shape) == 3:
        return img[:, :, 0]
    else:
        raise ValueError("Images must have either 2D or 3D")


# classes for data loading and preprocessing
class DatasetMasks:

    def __init__(
            self,
            base_path,
            folder,
            files,
            da_fn=None,
            preprocess_fn=None,
            memory_map=True,

    ):
        # Get full paths
        self.file_ids = list(files)
        self.image_files = [os.path.join(base_path, "images", folder, file) for file in self.file_ids]
        self.masks_files = [os.path.join(base_path, "masks", folder, file) for file in self.file_ids]

        # Load images in memory (minor speed-up)
        self.memory_map = memory_map
        if self.memory_map:
            self.mem_images = [force_2d(np.array(Image.open(self.image_files[i]))).astype(np.uint8) for i in range(len(self.file_ids))]
            self.mem_masks = [force_2d(np.array(Image.open(self.masks_files[i]))).astype(np.uint8) for i in range(len(self.file_ids))]  # masks could be RGB

        # Other
        self.da_fn = da_fn
        self.preprocess_fn = preprocess_fn

    def __getitem__(self, i):
        # Read/Load data
        if self.memory_map:
            image = self.mem_images[i]
            mask = self.mem_masks[i]
        else:
            image = np.array(Image.open(self.image_files[i]))
            mask = np.array(Image.open(self.masks_files[i]))

        # Force 2D
        image, mask = force_2d(image), force_2d(mask)

        # # Keep originals (just for visualization)
        original_image = np.array(image)
        original_mask = np.array(mask)

        # apply augmentations
        if self.da_fn:
            sample = self.da_fn(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            assert image.shape[:2] == (256, 256)
            assert mask.shape[:2] == (256, 256)

        # apply preprocessing
        if self.preprocess_fn:
            sample = self.preprocess_fn(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            assert image.shape[:2] == (256, 256)
            assert mask.shape[:2] == (256, 256)

        # Convert images
        image = np.stack((image,) * 3, axis=-1).astype(np.uint8)  # Grayscale to RGB
        mask = mask.astype(np.bool).astype(np.uint8)

        return image, mask, original_image, original_mask

    def __len__(self):
        return len(self.file_ids)


class DataloaderMasks(keras.utils.Sequence):

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

        # Transpose list of lists  (get first two elements: images, masks, *_)
        X, y, _, _ = tuple([samples for samples in zip(*data)])
        X = np.stack(X, axis=0).astype(np.float32)
        y = np.stack(y, axis=0).astype(np.float32)
        return X, y

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return math.ceil(len(self.indexes) / self.batch_size)

    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)


class DatasetImages:

    def __init__(
            self,
            base_path,
            folder,
            files,
            da_fn=None,
            preprocess_fn=None,
            memory_map=True,

    ):
        # Get full paths
        self.file_ids = list(files)
        self.image_files = [os.path.join(base_path, "images", folder, file) for file in self.file_ids]

        # Load images in memory (minor speed-up)
        self.memory_map = memory_map
        if self.memory_map:
            self.mem_images = [np.array(Image.open(self.image_files[i])).astype(np.uint8) for i in
                               range(len(self.file_ids))]

        # Other
        self.da_fn = da_fn
        self.preprocess_fn = preprocess_fn

    def __getitem__(self, i):
        # Read/Load data
        if self.memory_map:
            image = self.mem_images[i]
        else:
            image = np.array(Image.open(self.image_files[i]))

        # # Keep originals (just for visualization)
        original_image = np.array(image)

        # apply augmentations
        if self.da_fn:
            sample = self.da_fn(image=image)
            image = sample['image']
            assert image.shape[:2] == (256, 256)

        # apply preprocessing
        if self.preprocess_fn:
            sample = self.preprocess_fn(image=image)
            image = sample['image']
            assert image.shape[:2] == (256, 256)

        # Convert images
        image = np.stack((image,) * 3, axis=-1).astype(np.uint8)  # Grayscale to RGB

        return image, original_image

    def __len__(self):
        return len(self.file_ids)


class DataloaderImages(keras.utils.Sequence):
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
        data, data_ids = [], []
        for j in range(start, stop):
            data_ids.append(self.dataset.file_ids[j])
            data.append(self.dataset[j])

        # Transpose list of lists  (get first two elements: images, masks, *_)
        X, _ = tuple([samples for samples in zip(*data)])
        X = np.stack(X, axis=0).astype(np.float32)
        return data_ids, X

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return math.ceil(len(self.indexes) / self.batch_size)

    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)
