import tensorflow as tf
from tensorflow.keras.optimizers import SGD, Adam

import segmentation_models as sm
from segmentation_models import Unet
from segmentation_models.utils import set_trainable
from segmentation_models import get_preprocessing
from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score

from covid19.code.segmentation.utils import *

# Fix sm
sm.set_framework('tf.keras')
sm.framework()

# Get data
_, _, df_test = get_data(filename="data.csv", filter_masks=True)

# Load generators
test_ds = get_generators(df_test, test=True)

# Preview images
#preview_images(test_ds, n=4)

# Define model
model = Unet(BACKBONE, encoder_weights='imagenet', classes=1, activation='sigmoid', encoder_freeze=True)
model.summary()

# Load weights
model = tf.keras.load_model(os.path.join(MODELS_PATH, MODELS_NAME))
model.summary()

# Compile the model
model.compile(optimizer=Adam(learning_rate=1e-3), loss=bce_jaccard_loss, metrics=[iou_score])

# Evaluate model
print("Evaluating model")
scores = model.evaluate(test_ds)
print("Evaluation results")
print(scores)

print("Done!")

