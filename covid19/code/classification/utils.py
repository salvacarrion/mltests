import os

import matplotlib.pyplot as plt


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
