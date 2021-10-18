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
df_train, df_val, df_test = get_data(filename="data.csv", filter_masks=True)

# Load generators
train_ds = get_generators(df_train, test=False)
val_ds = get_generators(df_val, test=True)
test_ds = get_generators(df_test, test=True)

# Preview images
# preview_images(train_ds, n=4)

# preprocess input
preprocess_input = get_preprocessing(BACKBONE)
my_callbacks = [
    # WandbCallback(),
    tf.keras.callbacks.EarlyStopping(patience=10),
    tf.keras.callbacks.ModelCheckpoint(filepath=f'./{OUTPUTS_PATH}/models/' + 'model.{epoch:02d}-{val_loss:.2f}.h5'),
    tf.keras.callbacks.TensorBoard(log_dir=f'./{OUTPUTS_PATH}/logs'),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-5)
]

# define model
model = Unet(BACKBONE, encoder_weights='imagenet', classes=1, activation='sigmoid', encoder_freeze=True)
model.summary()

# Compile the model
model.compile(optimizer=Adam(learning_rate=1e-3), loss=bce_jaccard_loss, metrics=[iou_score])

# train the model on the new data for a few epochs
print("Training decoder first...")
history1 = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS1, callbacks=my_callbacks, shuffle=True)
print("Initial training results:")
print(history1)


# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
print("Fine-tuning model...")
set_trainable(model, recompile=False)
model.compile(optimizer=SGD(learning_rate=1e-4, momentum=0.9), loss=bce_jaccard_loss, metrics=[iou_score])

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
history2 = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS2, callbacks=my_callbacks, shuffle=True)
print("Fine-tuning results:")
print(history2)

# Save model
print("Saving model...")
savepath = os.path.join(MODELS_PATH, MODELS_NAME)
model.save(savepath)
print(f"Model saved at: {savepath}")

# Evaluate model
print("Evaluating model...")
scores = model.evaluate(test_ds)
print("Evaluation results")
print(scores)

print("Done!")

