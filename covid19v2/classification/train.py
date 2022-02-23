from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow.keras.optimizers import SGD, Adam

from da import *
from dataset import Dataset, Dataloader
from helpers import *
from utils import *

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from IPython.display import Image

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import xception

from PIL import Image as pil_image
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Display
from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# tf.config.experimental_run_functions_eagerly(True)

# Vars
SHOW_PLOTS = True
SHOW_DA_SAMPLES = False
PATIENCE = 10
WAIT_EPOCH_WARMUP = 10


def train(model, train_dataset, val_dataset, batch_size, epochs1, epochs2,
          checkpoints_path=None, logs_path=None,
          plots_path=None, use_multiprocessing=False, workers=1, single_output_idx=None):
    # Build dataloaders
    train_dataloader = Dataloader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = Dataloader(val_dataset, batch_size=batch_size, shuffle=False)

    # Callbacks
    model_callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7),
        CustomEarlyStopping(patience=PATIENCE, minimum_epochs=WAIT_EPOCH_WARMUP),
        CustomModelCheckpoint(filepath=checkpoints_path, save_best_only=True, wait_epoch_warmup=WAIT_EPOCH_WARMUP),
        # It can make the end of an epoch extremely slow
        tf.keras.callbacks.TensorBoard(log_dir=logs_path),
        # WandbCallback(),
    ]

    # # Unfreezing layers
    # print("------------------------------------------")
    # unfreeze_base_model(model, n=UNFREEZE_N)
    # print("------------------------------------------")

    # Compile the model
    metrics = [
        tf.keras.metrics.BinaryAccuracy(),
        tf.keras.metrics.AUC(),
        tf.keras.metrics.Precision(),
        tf.keras.metrics.Recall()
    ]
    model.compile(optimizer=Adam(learning_rate=LR_EPOCH1), loss=tf.keras.losses.BinaryCrossentropy(),  metrics=metrics)

    # Print model
    model.summary()

    # train the model on the new data for a few epochs
    if epochs1 <= 0:
        print("Skipping training output layers")
    else:
        print("Training output layers...")
        history1 = model.fit(train_dataloader, validation_data=val_dataloader, epochs=epochs1,
                             callbacks=model_callbacks,
                             use_multiprocessing=use_multiprocessing, workers=workers)
        print("Initial training results:")
        print(history1.history)
        if plots_path:
            plot_hist(history1, title="Training output layers", savepath=plots_path, suffix="_initial",
                      show_plot=SHOW_PLOTS)

    # Fine-tune?
    if epochs2 <= 0:
        print("Skipping fine-tuning")
    else:
        # Unfreezing layers
        print("------------------------------------------")
        unfreeze_base_model(model, n=UNFREEZE_N)
        print("------------------------------------------")

        # we need to recompile the model for these modifications to take effect
        # we use SGD with a low learning rate
        print("Fine-tuning model...")
        model.compile(optimizer=SGD(learning_rate=LR_EPOCH2, momentum=0.9), loss=tf.keras.losses.BinaryCrossentropy(), metrics=metrics)

        # Print model
        model.summary()

        # we train our model again (this time fine-tuning the top 2 inception blocks
        # alongside the top Dense layers
        history2 = model.fit(train_dataloader, validation_data=val_dataloader, epochs=epochs2,
                             callbacks=model_callbacks,
                             use_multiprocessing=use_multiprocessing, workers=workers)
        print("Fine-tuning results:")
        print(history2.history)
        if plots_path:
            plot_hist(history2, title="Fine-tuning full model", savepath=plots_path, suffix="_finetuning",
                      show_plot=SHOW_PLOTS)

def get_datasets(df, images_dir, target_size, prep_fn=None):
    # Get data
    tr_size = int(len(df)*0.8)
    df_train = df[:tr_size]
    df_val = df[tr_size:]

    # Build dataset
    train_dataset = Dataset(df_train, images_dir=images_dir, da_fn=tr_da_fn(*target_size),
                            preprocess_fn=prep_fn, target_size=target_size)
    val_dataset = Dataset(df_val, images_dir=images_dir, da_fn=ts_da_fn(*target_size),
                          preprocess_fn=prep_fn, target_size=target_size)

    print("****** Stats: **********************")
    print_stats(train_dataset, title="Train stats:")
    print_stats(val_dataset, title="Val stats:")

    return train_dataset, val_dataset


def print_stats(dataset, title="Stats:"):
    total = len(dataset)
    mild = sum(dataset.classes)
    severe = total - mild

    # Print stats
    print(title)
    print("\t- MILD: {} of {} ({:.2f}%)".format(mild, total, mild / total * 100))
    print("\t- SEVERE: {} of {} ({:.2f}%)".format(severe, total, severe / total * 100))



def main(backbone, input_size, target_size, batch_size, epochs1, epochs2=0, base_path=".", output_path=".",
         use_multiprocessing=False, workers=1, train_model=True, test_model=True,
         show_da_samples=False, run_name="run"):
    # Vars
    images_dir = os.path.join(base_path, "images", f"{input_size}_aligned")
    df = pd.read_excel(os.path.join(base_path, "data", "data.xls"))
    df["Prognosis"] = df["Prognosis"].apply(lambda x: x.replace("LIEVE", "MILD"))
    # classes = len(set(df["Prognosis"].values))

    # Outputs
    checkpoints_path = os.path.join(output_path, run_name, "models")
    logs_path = os.path.join(output_path, run_name, "logs")
    plots_path = os.path.join(output_path, run_name, "plots")

    # Create folders
    for dir_i in [checkpoints_path, logs_path, plots_path]:
        Path(dir_i).mkdir(parents=True, exist_ok=True)

    # Get model
    model, prep_fn = get_model(backbone=backbone, classes=1, target_size=target_size, freeze_base_model=True)

    # Get data
    train_dataset, val_dataset = get_datasets(df, images_dir=images_dir, target_size=target_size, prep_fn=prep_fn)

    # Train
    # train(model, train_dataset, val_dataset, batch_size=batch_size, epochs1=epochs1, epochs2=epochs2,
    #       checkpoints_path=checkpoints_path, logs_path=logs_path,
    #       plots_path=plots_path, use_multiprocessing=use_multiprocessing, workers=workers)

    # Test
    test(val_dataset, checkpoints_path, batch_size)

class GradVisualizer:
    """Plot gradients of the outputs w.r.t an input image."""

    def __init__(self, positive_channel=None, negative_channel=None):
        if positive_channel is None:
            self.positive_channel = [0, 255, 0]
        else:
            self.positive_channel = positive_channel

        if negative_channel is None:
            self.negative_channel = [255, 0, 0]
        else:
            self.negative_channel = negative_channel

    def apply_polarity(self, attributions, polarity):
        if polarity == "positive":
            return np.clip(attributions, 0, 1)
        else:
            return np.clip(attributions, -1, 0)

    def apply_linear_transformation(
        self,
        attributions,
        clip_above_percentile=99.9,
        clip_below_percentile=70.0,
        lower_end=0.2,
    ):
        # 1. Get the thresholds
        m = self.get_thresholded_attributions(
            attributions, percentage=100 - clip_above_percentile
        )
        e = self.get_thresholded_attributions(
            attributions, percentage=100 - clip_below_percentile
        )

        # 2. Transform the attributions by a linear function f(x) = a*x + b such that
        # f(m) = 1.0 and f(e) = lower_end
        transformed_attributions = (1 - lower_end) * (np.abs(attributions) - e) / (
            m - e
        ) + lower_end

        # 3. Make sure that the sign of transformed attributions is the same as original attributions
        transformed_attributions *= np.sign(attributions)

        # 4. Only keep values that are bigger than the lower_end
        transformed_attributions *= transformed_attributions >= lower_end

        # 5. Clip values and return
        transformed_attributions = np.clip(transformed_attributions, 0.0, 1.0)
        return transformed_attributions

    def get_thresholded_attributions(self, attributions, percentage):
        if percentage == 100.0:
            return np.min(attributions)

        # 1. Flatten the attributions
        flatten_attr = attributions.flatten()

        # 2. Get the sum of the attributions
        total = np.sum(flatten_attr)

        # 3. Sort the attributions from largest to smallest.
        sorted_attributions = np.sort(np.abs(flatten_attr))[::-1]

        # 4. Calculate the percentage of the total sum that each attribution
        # and the values about it contribute.
        cum_sum = 100.0 * np.cumsum(sorted_attributions) / total

        # 5. Threshold the attributions by the percentage
        indices_to_consider = np.where(cum_sum >= percentage)[0][0]

        # 6. Select the desired attributions and return
        attributions = sorted_attributions[indices_to_consider]
        return attributions

    def binarize(self, attributions, threshold=0.001):
        return attributions > threshold

    def morphological_cleanup_fn(self, attributions, structure=np.ones((4, 4))):
        closed = ndimage.grey_closing(attributions, structure=structure)
        opened = ndimage.grey_opening(closed, structure=structure)
        return opened

    def draw_outlines(
        self, attributions, percentage=90, connected_component_structure=np.ones((3, 3))
    ):
        # 1. Binarize the attributions.
        attributions = self.binarize(attributions)

        # 2. Fill the gaps
        attributions = ndimage.binary_fill_holes(attributions)

        # 3. Compute connected components
        connected_components, num_comp = ndimage.measurements.label(
            attributions, structure=connected_component_structure
        )

        # 4. Sum up the attributions for each component
        total = np.sum(attributions[connected_components > 0])
        component_sums = []
        for comp in range(1, num_comp + 1):
            mask = connected_components == comp
            component_sum = np.sum(attributions[mask])
            component_sums.append((component_sum, mask))

        # 5. Compute the percentage of top components to keep
        sorted_sums_and_masks = sorted(component_sums, key=lambda x: x[0], reverse=True)
        sorted_sums = list(zip(*sorted_sums_and_masks))[0]
        cumulative_sorted_sums = np.cumsum(sorted_sums)
        cutoff_threshold = percentage * total / 100
        cutoff_idx = np.where(cumulative_sorted_sums >= cutoff_threshold)[0][0]
        if cutoff_idx > 2:
            cutoff_idx = 2

        # 6. Set the values for the kept components
        border_mask = np.zeros_like(attributions)
        for i in range(cutoff_idx + 1):
            border_mask[sorted_sums_and_masks[i][1]] = 1

        # 7. Make the mask hollow and show only the border
        eroded_mask = ndimage.binary_erosion(border_mask, iterations=1)
        border_mask[eroded_mask] = 0

        # 8. Return the outlined mask
        return border_mask

    def process_grads(
        self,
        image,
        attributions,
        polarity="positive",
        clip_above_percentile=99.9,
        clip_below_percentile=0,
        morphological_cleanup=False,
        structure=np.ones((3, 3)),
        outlines=False,
        outlines_component_percentage=90,
        overlay=True,
    ):
        if polarity not in ["positive", "negative"]:
            raise ValueError(
                f""" Allowed polarity values: 'positive' or 'negative'
                                    but provided {polarity}"""
            )
        if clip_above_percentile < 0 or clip_above_percentile > 100:
            raise ValueError("clip_above_percentile must be in [0, 100]")

        if clip_below_percentile < 0 or clip_below_percentile > 100:
            raise ValueError("clip_below_percentile must be in [0, 100]")

        # 1. Apply polarity
        if polarity == "positive":
            attributions = self.apply_polarity(attributions, polarity=polarity)
            channel = self.positive_channel
        else:
            attributions = self.apply_polarity(attributions, polarity=polarity)
            attributions = np.abs(attributions)
            channel = self.negative_channel

        # 2. Take average over the channels
        attributions = np.average(attributions, axis=2)

        # 3. Apply linear transformation to the attributions
        attributions = self.apply_linear_transformation(
            attributions,
            clip_above_percentile=clip_above_percentile,
            clip_below_percentile=clip_below_percentile,
            lower_end=0.0,
        )

        # 4. Cleanup
        if morphological_cleanup:
            attributions = self.morphological_cleanup_fn(
                attributions, structure=structure
            )
        # 5. Draw the outlines
        if outlines:
            attributions = self.draw_outlines(
                attributions, percentage=outlines_component_percentage
            )

        # 6. Expand the channel axis and convert to RGB
        attributions = np.expand_dims(attributions, 2) * channel

        # 7.Superimpose on the original image
        if overlay:
            attributions = np.clip((attributions * 0.8 + image), 0, 255)
        return attributions

    def visualize(
        self,
        image,
        gradients,
        integrated_gradients,
        polarity="positive",
        clip_above_percentile=99.9,
        clip_below_percentile=0,
        morphological_cleanup=False,
        structure=np.ones((3, 3)),
        outlines=False,
        outlines_component_percentage=90,
        overlay=True,
        figsize=(15, 8),
    ):
        # 1. Make two copies of the original image
        img1 = np.copy(image)
        img2 = np.copy(image)

        # 2. Process the normal gradients
        grads_attr = self.process_grads(
            image=img1,
            attributions=gradients,
            polarity=polarity,
            clip_above_percentile=clip_above_percentile,
            clip_below_percentile=clip_below_percentile,
            morphological_cleanup=morphological_cleanup,
            structure=structure,
            outlines=outlines,
            outlines_component_percentage=outlines_component_percentage,
            overlay=overlay,
        )

        # 3. Process the integrated gradients
        igrads_attr = self.process_grads(
            image=img2,
            attributions=integrated_gradients,
            polarity=polarity,
            clip_above_percentile=clip_above_percentile,
            clip_below_percentile=clip_below_percentile,
            morphological_cleanup=morphological_cleanup,
            structure=structure,
            outlines=outlines,
            outlines_component_percentage=outlines_component_percentage,
            overlay=overlay,
        )

        _, ax = plt.subplots(1, 3, figsize=figsize)
        ax[0].imshow(image)
        ax[1].imshow(grads_attr.astype(np.uint8))
        ax[2].imshow(igrads_attr.astype(np.uint8))

        ax[0].set_title("Input")
        ax[1].set_title("Normal gradients")
        ax[2].set_title("Integrated gradients")
        plt.show()

def get_img_array(img_path, size=(299, 299)):
    # `img` is a PIL image of size 299x299
    img = keras.preprocessing.image.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array


def get_gradients(model, img_input, top_pred_idx):
    """Computes the gradients of outputs w.r.t input image.

    Args:
        img_input: 4D image tensor
        top_pred_idx: Predicted label for the input image

    Returns:
        Gradients of the predictions w.r.t img_input
    """
    images = tf.cast(img_input, tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(images)
        preds = model(images)
        # top_class = preds[:, top_pred_idx]

    grads = tape.gradient(preds, images)
    return grads


def get_integrated_gradients(model, img_input, top_pred_idx, baseline=None, num_steps=50):
    """Computes Integrated Gradients for a predicted label.

    Args:
        img_input (ndarray): Original image
        top_pred_idx: Predicted label for the input image
        baseline (ndarray): The baseline image to start with for interpolation
        num_steps: Number of interpolation steps between the baseline
            and the input used in the computation of integrated gradients. These
            steps along determine the integral approximation error. By default,
            num_steps is set to 50.

    Returns:
        Integrated gradients w.r.t input image
    """
    # If baseline is not provided, start with a black image
    # having same size as the input image.
    if baseline is None:
        baseline = np.zeros(img_input.shape).astype(np.float32)
    else:
        baseline = baseline.astype(np.float32)

    # 1. Do interpolation.
    img_input = img_input.astype(np.float32)
    interpolated_image = [
        baseline + (step / num_steps) * (img_input - baseline)
        for step in range(num_steps + 1)
    ]
    interpolated_image = np.array(interpolated_image).astype(np.float32)

    # 2. Preprocess the interpolated images
    interpolated_image = xception.preprocess_input(interpolated_image)

    # 3. Get the gradients
    grads = []
    for i, img in enumerate(interpolated_image):
        img = tf.expand_dims(img, axis=0)
        grad = get_gradients(model, img, top_pred_idx=top_pred_idx)
        grads.append(grad[0])
    grads = tf.convert_to_tensor(grads, dtype=tf.float32)

    # 4. Approximate the integral using the trapezoidal rule
    grads = (grads[:-1] + grads[1:]) / 2.0
    avg_grads = tf.reduce_mean(grads, axis=0)

    # 5. Calculate integrated gradients and return
    integrated_grads = (img_input - baseline) * avg_grads
    return integrated_grads


def random_baseline_integrated_gradients(model,
    img_input, top_pred_idx, num_steps=50, num_runs=2
):
    """Generates a number of random baseline images.

    Args:
        img_input (ndarray): 3D image
        top_pred_idx: Predicted label for the input image
        num_steps: Number of interpolation steps between the baseline
            and the input used in the computation of integrated gradients. These
            steps along determine the integral approximation error. By default,
            num_steps is set to 50.
        num_runs: number of baseline images to generate

    Returns:
        Averaged integrated gradients for `num_runs` baseline images
    """
    # 1. List to keep track of Integrated Gradients (IG) for all the images
    integrated_grads = []

    # 2. Get the integrated gradients for all the baselines
    for run in range(num_runs):
        baseline = np.random.random(img_input.shape) * 255
        igrads = get_integrated_gradients(
            model=model,
            img_input=img_input,
            top_pred_idx=top_pred_idx,
            baseline=baseline,
            num_steps=num_steps,
        )
        integrated_grads.append(igrads)

    # 3. Return the average integrated gradients for the image
    integrated_grads = tf.convert_to_tensor(integrated_grads)
    return tf.reduce_mean(integrated_grads, axis=0)

def test(test_dataset, checkpoints_path, batch_size):
    # Build dataloader
    test_dataloader = Dataloader(test_dataset, batch_size=1, shuffle=False)

    # Load model
    print("Loading best model...")
    model = tf.keras.models.load_model(filepath=checkpoints_path, compile=False)  # Loads best model automatically
    model.summary()

    # Compile the model
    metrics = [
        tf.keras.metrics.BinaryAccuracy(),
        tf.keras.metrics.AUC(),
        tf.keras.metrics.Precision(),
        tf.keras.metrics.Recall()
    ]
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(),  metrics=metrics)

    # # Evaluate model
    # print("Evaluating model...")
    # scores = model.evaluate(test_dataloader)
    # print("Evaluation results")
    # print(scores)

    # Grad cam
    for x, y, id in test_dataloader:
        orig_img2 = test_dataset.mem_images[id[0]]
        orig_img = ((np.array(x).squeeze(0) + 1) / 2 * 255).astype(np.uint8)

        # 5. Get the gradients of the last layer for the predicted label
        grads = get_gradients(model, x, top_pred_idx=y)

        # 6. Get the integrated gradients
        igrads = random_baseline_integrated_gradients(model,
            np.copy(orig_img), top_pred_idx=y, num_steps=50, num_runs=2
        )

        # 7. Process the gradients and plot
        vis = GradVisualizer()
        vis.visualize(
            image=orig_img,
            gradients=grads[0].numpy(),
            integrated_gradients=igrads.numpy(),
            clip_above_percentile=99,
            clip_below_percentile=0,
        )

        vis.visualize(
            image=orig_img,
            gradients=grads[0].numpy(),
            integrated_gradients=igrads.numpy(),
            clip_above_percentile=95,
            clip_below_percentile=28,
            morphological_cleanup=True,
            outlines=True,
        )
        asd = 33


if __name__ == "__main__":
    BASE_PATH = "/home/scarrion/datasets/nn/vision/covid19v2"
    OUTPUT_PATH = "/home/scarrion/projects/mltests/covid19v2/classification/.outputs"

    BACKBONE = "resnet101v2"  # "efficientnetb0"  #"resnet101v2"
    BATCH_SIZE = 32
    INPUT_SIZE = 512
    TARGET_SIZE = (INPUT_SIZE, INPUT_SIZE)
    EPOCHS1 = 50
    EPOCHS2 = 30  # Careful when unfreezing. More gradients, more memory.
    LR_EPOCH1 = 10e-3
    LR_EPOCH2 = 10e-5
    UNFREEZE_N = 5
    NAME_AUX = ""

    # Set name
    RUN_NAME = f"model2_{BACKBONE}"

    print(f"##################################################")
    print(f"##################################################")
    print(f"MODEL NAME: {RUN_NAME}")
    print(f"##################################################")
    print(f"##################################################")

    # Run
    main(backbone=BACKBONE, input_size=INPUT_SIZE, target_size=TARGET_SIZE, batch_size=BATCH_SIZE,
         epochs1=EPOCHS1, epochs2=EPOCHS2, base_path=BASE_PATH, output_path=OUTPUT_PATH,
         use_multiprocessing=False, workers=1, train_model=True, test_model=True,
         show_da_samples=SHOW_DA_SAMPLES, run_name=RUN_NAME)
    print("Done!")

    # # Create a MirroredStrategy.
    # strategy = tf.distribute.MirroredStrategy()
    # print("Number of devices: {}".format(strategy.num_replicas_in_sync))
    #
    # # Open a strategy scope.
    # with strategy.scope():
