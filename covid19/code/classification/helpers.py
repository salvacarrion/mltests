import tensorflow as tf


@tf.function
def BinaryAccuracy_Infiltrates(y_true, y_pred, i=0):
    return tf.keras.metrics.binary_accuracy(y_true[:, i], y_pred[:, i])

@tf.function
def BinaryAccuracy_Pneumonia(y_true, y_pred, i=1):
    return tf.keras.metrics.binary_accuracy(y_true[:, i], y_pred[:, i])

@tf.function
def BinaryAccuracy_Covid19(y_true, y_pred, i=2):
    return tf.keras.metrics.binary_accuracy(y_true[:, i], y_pred[:, i])


class CustomModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.wait_epoch_warmup = kwargs.get("wait_epoch_warmup")

    def on_epoch_end(self, epoch, logs=None):
        if self.wait_epoch_warmup:
            if (epoch+1) >= self.wait_epoch_warmup:
                super().on_epoch_end(epoch, logs)
            else:
                self.epochs_since_last_save += 1
                print(f"Skipping save model (wait_epoch_warmup={self.wait_epoch_warmup-(epoch+1)})")
        else:
            super().on_epoch_end(epoch, logs)


class CustomEarlyStopping(tf.keras.callbacks.EarlyStopping):
    def __init__(self, *args, **kwargs):
        self.minimum_epochs = kwargs.get("minimum_epochs", 0)
        kwargs.pop('minimum_epochs', None)  # Problems with EarlyStopping kwargs
        super().__init__(*args, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        if epoch >= self.minimum_epochs:
            super().on_epoch_end(epoch, logs)


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


def get_model(backbone, classes=None, target_size=None, freeze_base_model=True, ignore_model=None):
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

    if ignore_model:
        model = None
    else:
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


def unfreeze_base_model(model, n=None, unfreeze=True):
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
