import os

import tensorflow as tf

import matplotlib.pyplot as plt
from matplotlib import ticker

from covid19.code.classification.helpers import *


def get_losses():
    losses = [tf.keras.losses.BinaryCrossentropy()]
    return losses


def get_metrics(single_output_idx):
    metrics = []
    if single_output_idx is None:  # Multi-label
        print("###### Multi-label classification ######")
        metrics += [
            BinaryAccuracy_Infiltrates,
            BinaryAccuracy_Pneumonia,
            BinaryAccuracy_Covid19
        ]
    else:
        print(f"###### Multi-class classification (cls: '{single_output_idx}') ######")
        metrics = [
            tf.keras.metrics.BinaryAccuracy(),
            tf.keras.metrics.AUC(),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall()
        ]
    return metrics


def get_model(backbone, classes, target_size=None, freeze_base_model=True):
    istrainable = not freeze_base_model

    # Select backbone
    if backbone == "resnet50":
        from tensorflow.keras.applications.resnet import ResNet50 as TFModel
        from tensorflow.keras.applications.resnet import preprocess_input
    elif backbone == "resnet50v2":
        from tensorflow.keras.applications.resnet_v2 import ResNet50V2 as TFModel
        from tensorflow.keras.applications.resnet_v2 import preprocess_input
    elif backbone == "resnet101v2":
        from tensorflow.keras.applications.resnet_v2 import ResNet101V2 as TFModel
        from tensorflow.keras.applications.resnet_v2 import preprocess_input
    elif backbone == "vgg16":
        from tensorflow.keras.applications.vgg16 import VGG16 as TFModel
        from tensorflow.keras.applications.vgg16 import preprocess_input
    elif backbone == "efficientnetb0":
        from tensorflow.keras.applications.efficientnet import EfficientNetB0 as TFModel
        from tensorflow.keras.applications.efficientnet import preprocess_input
    elif backbone == "efficientnetb7":
        from tensorflow.keras.applications.efficientnet import EfficientNetB7 as TFModel
        from tensorflow.keras.applications.efficientnet import preprocess_input
    else:
        raise ValueError(f"Unknown backbone: {backbone}")

    # Instantiate base model with pre-trained weights
    base_model = TFModel(input_shape=(*target_size, 3), include_top=False, weights="imagenet")

    # Freeze base model
    # base_model.trainable = istrainable
    for layers in base_model.layers:
        layers.trainable = istrainable

    # Create a new model on top
    inputs = base_model.input
    x = base_model(inputs)

    # Option A
    x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)

    # Option B
    # x = tf.keras.layers.Flatten(name='flatten')(x)
    # x = tf.keras.layers.Dense(512, activation='relu', name='fc1')(x)
    # x = tf.keras.layers.Dense(512, activation='relu', name='fc2')(x)

    # Outputs
    outputs = tf.keras.layers.Dense(classes, activation="sigmoid", name='predictions')(x)
    model = tf.keras.Model(inputs, outputs)
    return model, preprocess_input


def unfreeze_base_model(model, n=None):
    base_model = model.layers[1].layers

    # Select number of layers to unfreeze
    idx = 0
    if n is not None:
        if isinstance(n, int):
            idx = n
            print(f"Unfreezing {len(base_model)-idx} layers")
        elif isinstance(n, float) and 0.0 < n <= 1.0:
            idx = int(len(base_model)*n)
            print(f"Unfreezing {idx} layers")
        else:
            raise ValueError("Invalid number of layers")

    # We unfreeze all layers but BatchNorm (to not destroy the non-trainable weights)
    for layer in base_model[-idx:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True


# helper function for data visualization
def plot_da_image_and_masks(image, mask, original_image=None, original_mask=None):
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


def plot_da_image(image, original_image=None):
    fontsize = 18
    plt.gray()

    if original_image is None:
        f, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(image)
    else:
        f, ax = plt.subplots(1, 2, figsize=(8, 8))

        ax[0].imshow(original_image)
        ax[0].set_title('Original image', fontsize=fontsize)

        ax[1].imshow(image)
        ax[1].set_title('Transformed image', fontsize=fontsize)
    plt.show()


def plot_hist(history, title="", savepath=None, suffix="", show_plot=True):
    metrics = [m for m in history.history.keys() if not m.startswith("val")]

    for m in metrics:
        fig, ax = plt.subplots(1, 1, figsize=(13, 8))

        # Plot
        x = range(1, len(history.history[m])+1)
        ax.plot(x, history.history[m])

        val_metric = f"val_{m}"
        if val_metric in history.history:
            ax.plot(x, history.history[val_metric])
            ax.legend(['Train', 'Val'], loc='upper left')
        else:
            ax.legend(['Train'], loc='upper left')

        # Common
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.set_xlabel('Epoch')
        ax.set_ylabel(m.replace("_", " ").title())
        plt.title(f'{m.replace("_", " ").title()}\n({title})')

        # Save figures
        if savepath:
            plt.savefig(os.path.join(savepath, f"{m}{suffix}.pdf"))
            plt.savefig(os.path.join(savepath, f"{m}{suffix}.png"))

        # Show
        if show_plot:
            plt.show()


def visualize_da(dataset, i, n=5):
    for _ in range(n):
        (image, original_image), _ = dataset.__getitem__(i, show_originals=True)
        plot_da_image(
            image=image,
            original_image=original_image,
        )