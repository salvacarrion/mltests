import os
import tensorflow as tf
import tf2onnx
import onnx


def convert2onnx(model_path, savepath="model.onnx", input_size=(1, 256, 256, 3)):
    # Load model
    print("Loading model...")
    model = tf.keras.models.load_model(filepath=model_path, compile=False)
    model.summary()

    # Convert to ONNX
    print("Converting model to onnx...")
    input_spec = (tf.TensorSpec(input_size, dtype=tf.float32),)
    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=input_spec)

    # Save ONNX to file
    onnx.save(onnx_model, savepath)
    print("Model saved!")


if __name__ == "__main__":
    MODEL_PATH = os.path.join("/Users/salvacarrion/Desktop/resnet34_segmentation", "unet_resnet34.h5")
    SAVEPATH = "./model.onnx"
    INPUT_SIZE = (1, 256, 256, 3)
    convert2onnx(model_path=MODEL_PATH, savepath=SAVEPATH, input_size=INPUT_SIZE)
