import os.path

import tensorflow as tf
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.models import load_model

import segmentation_models as sm
from segmentation_models import Unet
from segmentation_models.utils import set_trainable
from segmentation_models import get_preprocessing
from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score

from covid19.code.segmentation.utils import *
from covid19.code.segmentation.dataset import Dataset, Dataloder

# Fix sm
sm.set_framework('tf.keras')
sm.framework()

# Get data
_, _, df_test = get_data(filename="data.csv", filter_masks=True)

# Dirs
imgs_dir = os.path.join(BASE_PATH, "images256")
masks_dir = os.path.join(BASE_PATH, "masks256")

# preprocess input
preprocess_input = sm.get_preprocessing(BACKBONE)

# Build dataset
test_dataset = Dataset(df_test, imgs_dir=imgs_dir, masks_dir=masks_dir, augmentation=get_validation_augmentation(), preprocessing=get_preprocessing(preprocess_input), target_size=IMAGE_SIZE)

# # Visualize image
# for i in range(10):
#     image, mask, original_image, original_mask = train_dataset[5]  # get some sample
#     visualize(
#         image=image,
#         mask=mask,
#         original_image=original_image,
#         original_mask=original_mask,
#     )
# asdsa = 3

test_dataloader = Dataloder(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Save model
print("Loading model...")
model = load_model(filepath=os.path.join(OUTPUT_PATH, "models", "my_last_model.h5"), compile=False)
model.summary()

# Compile the model
model.compile(optimizer=Adam(learning_rate=1e-3), loss=bce_jaccard_loss, metrics=[iou_score])

# Evaluate model
print("Evaluating model...")
scores = model.evaluate(test_dataloader)
print("Evaluation results")
print(scores)

print("Done!")

