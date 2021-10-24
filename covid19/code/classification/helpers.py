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

