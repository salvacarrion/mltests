import glob
import os
import pandas as pd
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
sns.set()

BASE_PATH = "/home/scarrion/datasets/covid19/all-covid"

# Check if all masks match an images
img_files = [os.path.splitext(os.path.split(file)[1])[0] for file in glob.glob(os.path.join(BASE_PATH, "images256", "*.jpg"))]
mask_files = [os.path.splitext(os.path.split(file)[1])[0] for file in glob.glob(os.path.join(BASE_PATH, "masks256", "*.png"))]

print("Summary:")
print(f"\t- Total images: {len(img_files)}")
print(f"\t- Total masks: {len(mask_files)}")
print(f"\t- Total images with no mask: {len(set(img_files).difference(mask_files))}")
print(f"\t- Total masks with no image: {len(set(mask_files).difference(img_files))}")


def plot_segmentation(img, mask, file_name=""):
    # import copy
    # my_cmap = copy.copy(plt.cm.get_cmap('gray'))  # get a copy of the gray color map
    # my_cmap.set_bad(alpha=0)  # set how the colormap handles 'bad' values
    # mask[mask <= 0] = np.nan

    # Blend image
    img_blended = Image.blend(img, mask, 0.5)
    # img_blended = Image.alpha_composite(img, mask)

    # To numpy
    img_blended = np.asarray(img_blended)

    fig, axes = plt.subplots(1, 1, dpi=150, figsize=(20, 20))
    axes.imshow(img_blended, interpolation='none')
    # axes.imshow((img * 255).astype("uint8"), interpolation='none')
    # axes.imshow((mask * 255).astype("uint8"), interpolation='none', cmap=None, alpha=0.25)
    plt.title(file_name)
    plt.tight_layout()
    plt.show()
    asd = 3

# Preview images
preview = 4
for file in mask_files[:preview]:
    # img = mpimg.imread(os.path.join(BASE_PATH, "images256", file + ".jpg"))
    # masks = mpimg.imread(os.path.join(BASE_PATH, "masks256", file + ".png"))
    img = Image.open(os.path.join(BASE_PATH, "images256", file + ".jpg")).convert("RGBA")
    masks = Image.open(os.path.join(BASE_PATH, "masks256", file + ".png")).convert("RGBA")
    plot_segmentation(img, masks, file_name=file)

print("Done!")
