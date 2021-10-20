import os

import matplotlib.pyplot as plt
from matplotlib import ticker


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
        # ax.xaxis.set_major_locator(ticker.AutoLocator())
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
