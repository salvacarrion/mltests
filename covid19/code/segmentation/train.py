import os.path

import tensorflow as tf
from tensorflow.keras.optimizers import SGD, Adam

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
df_train, df_val, df_test = get_data(filename="data.csv", filter_masks=True)

# Dirs
imgs_dir = os.path.join(BASE_PATH, "images256")
masks_dir = os.path.join(BASE_PATH, "masks256")

# preprocess input
preprocess_input = sm.get_preprocessing(BACKBONE)

# Build dataset
train_dataset = Dataset(df_train, imgs_dir=imgs_dir, masks_dir=masks_dir, augmentation=get_training_augmentation(), preprocessing=get_preprocessing(preprocess_input), target_size=IMAGE_SIZE)
val_dataset = Dataset(df_val, imgs_dir=imgs_dir, masks_dir=masks_dir, augmentation=get_validation_augmentation(), preprocessing=get_preprocessing(preprocess_input), target_size=IMAGE_SIZE)
test_dataset = Dataset(df_test, imgs_dir=imgs_dir, masks_dir=masks_dir, augmentation=get_validation_augmentation(), preprocessing=get_preprocessing(preprocess_input), target_size=IMAGE_SIZE)

# Visualize image
# for i in range(10):
#     image, mask, original_image, original_mask = train_dataset[5]  # get some sample
#     visualize(
#         image=image,
#         mask=mask,
#         original_image=original_image,
#         original_mask=original_mask,
#     )
# asdsa = 3

train_dataloader = Dataloder(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = Dataloder(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_dataloader = Dataloder(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Callbacks
my_callbacks = [
    tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(OUTPUT_PATH, "models"), save_best_only=True, mode='min'),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-5),
    tf.keras.callbacks.EarlyStopping(patience=10),
    # WandbCallback(),
    tf.keras.callbacks.TensorBoard(log_dir=os.path.join(OUTPUT_PATH, "logs")),
]

# define model
model = Unet(BACKBONE, encoder_weights='imagenet', classes=1, activation='sigmoid', encoder_freeze=True)
model.summary()

# Compile the model
model.compile(optimizer=Adam(learning_rate=1e-3), loss=bce_jaccard_loss, metrics=[iou_score])

# train the model on the new data for a few epochs
print("Training decoder first...")
history1 = model.fit(train_dataloader, validation_data=val_dataloader, epochs=EPOCHS1, callbacks=my_callbacks, use_multiprocessing=USE_MULTIPROCESSING, workers=NUM_WORKERS)
print("Initial training results:")
print(history1)
plot_hist(history1, title="Training decoder", savepath=os.path.join(OUTPUT_PATH, "plots"), suffix="_initial")

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
print("Fine-tuning model...")
set_trainable(model, recompile=False)
model.compile(optimizer=SGD(learning_rate=1e-4, momentum=0.9), loss=bce_jaccard_loss, metrics=[iou_score])

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
history2 = model.fit(train_dataloader, validation_data=val_dataloader, epochs=EPOCHS2, callbacks=my_callbacks, use_multiprocessing=USE_MULTIPROCESSING, workers=NUM_WORKERS)
print("Fine-tuning results:")
print(history2)
plot_hist(history2, title="Finetuning whole model", savepath=os.path.join(OUTPUT_PATH, "plots"), suffix="_finetuning")

# Save model
print("Saving model...")
savepath = os.path.join(OUTPUT_PATH, "models", "my_last_model.h5")
model.save(savepath)
print(f"Model saved at: {savepath}")

# Evaluate model
print("Evaluating model...")
scores = model.evaluate(test_dataloader)
print("Evaluation results")
print(scores)

print("Done!")

