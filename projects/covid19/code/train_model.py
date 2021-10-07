import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalMaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import SGD, Adam
from keras_preprocessing.image import ImageDataGenerator

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

EPOCHS1 = 10
EPOCHS2 = 15
BATCH_SIZE = 32
IMAGE_SIZE = (256, 256)
BASE_PATH = "/home/scarrion/datasets/covid19"

# Load csv
df_train = pd.read_csv(os.path.join(BASE_PATH, "new_train.csv"))
df_test = pd.read_csv(os.path.join(BASE_PATH, "new_test.csv"))

train_da = ImageDataGenerator(
    rescale=1./255,
    zoom_range=0.05,
    rotation_range=3,
    shear_range=0.01,
    width_shift_range=[-0.05, +0.05],
    height_shift_range=[-0.05, +0.05],
    brightness_range=[0.95, 1.05],
    fill_mode="constant",
    cval=0,
    horizontal_flip=False,
    validation_split=0.1)
test_da = ImageDataGenerator(rescale=1./255)

train_ds = train_da.flow_from_dataframe(
    df_train,
    os.path.join(BASE_PATH, "images"),
    x_col="filepath",
    y_col="covid",
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    # color_mode="grayscale",
    classes=["covid", "no_covid"],
)
test_ds = train_da.flow_from_dataframe(
    df_test,
    os.path.join(BASE_PATH, "images"),
    x_col="filepath",
    y_col="covid",
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    color_mode="grayscale",
    classes=["covid", "no_covid"],
)
id2lbl = {v: k for k, v in train_ds.class_indices.items()}

# # Preview images
# plt.figure(figsize=(10, 10))
# for images, labels in [train_ds.next()]:
#     for i in range(9):
#         ax = plt.subplot(3, 3, i + 1)
#         plt.imshow((images[i]*255).astype("uint8"), vmin=0, vmax=255, cmap="gray")
#         plt.title(id2lbl.get(int(labels[i]), "unknown"))
#         plt.axis("off")
#     plt.show()

# Create network
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(*IMAGE_SIZE, 3), classes=2)
x = base_model.output
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(1, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=outputs)

# Freeze top layers
for layer in base_model.layers:
    layer.trainable = False

# Make sure you have frozen the correct layers
for i, layer in enumerate(model.layers):
    print(i, layer.name, layer.trainable)

my_callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=10),
    tf.keras.callbacks.ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.h5'),
    tf.keras.callbacks.TensorBoard(log_dir='./logs'),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-5)
]

# Compile the model
model.compile(optimizer=Adam(learning_rate=1e-3), loss='categorical_crossentropy', metrics=["accuracy"])

# train the model on the new data for a few epochs
history1 = model.fit(train_ds, validation_data=test_ds, epochs=EPOCHS1, callbacks=my_callbacks, shuffle=True)

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
for layer in model.layers[:249]:
   layer.trainable = False
for layer in model.layers[249:]:
   layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=["accuracy"])

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
history2 = model.fit(train_ds, validation_data=test_ds, epochs=EPOCHS2, callbacks=my_callbacks, shuffle=True)

# Evaluate model
scores = model.evaluate(test_ds)
print(scores)