import os

import matplotlib.pyplot as plt
from matplotlib import ticker

import numpy as np
from PIL import Image

from covid19v2.segmentation import utils


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


def plot_da_image(image, original_image=None, title=None):
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

    if title:
        plt.suptitle(title)

    plt.show()


def plot_hist(history, title="", savepath=None, suffix="", show_plot=True):
    metrics = [m for m in history.history.keys() if not m.startswith("val")]

    for m in metrics:
        fig, ax = plt.subplots(1, 1, figsize=(13, 8))

        # Plot
        x = range(1, len(history.history[m]) + 1)
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

# helper function for data visualization
def plot_4x4(image, mask, original_image=None, original_mask=None, title=None):
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

    if title:
        plt.suptitle(title)

    plt.show()


def plot_dataset(dataset, img_i, num_aug=5, mode="img+img_da"):
    for _ in range(num_aug):
        image, mask, original_image, original_mask = dataset.__getitem__(img_i)
        filename = dataset.file_ids[img_i]

        if mode == "img+img_da":
            plot_da_image(
                image=image,
                original_image=original_image,
                title=filename,
            )
        elif mode == "img+mask":
            plot_da_image(
                image=image,
                original_image=mask,
                title=filename,
            )
        else:  # all
            plot_4x4(
                image=image,
                mask=mask,
                original_image=original_image,
                original_mask=original_mask,
                title=filename,
            )


def masks2overlay(image, mask, output_path, file_id):
    assert image.shape[:2] == mask.shape[:2]

    # From 0-1 to 0-255
    image = image.astype(np.uint8)
    mask = ((mask > 0) * 255).astype(np.uint8)

    # Convert to PIL
    pil_img = Image.fromarray(image)
    pil_mask = Image.fromarray(mask)

    # Convert to RGBA
    pil_img = pil_img.convert('RGBA')
    pil_mask = pil_mask.convert('RGBA')

    # Make the background transparent
    pil_mask_alpha = utils.make_transparent(pil_mask, color=(0, 0, 0))
    pil_mask_alpha.putalpha(75)

    # Overlay mask and save image
    overlaid_img = Image.alpha_composite(pil_img, pil_mask_alpha)
    overlaid_img.save(os.path.join(output_path, file_id))
