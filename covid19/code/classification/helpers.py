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
