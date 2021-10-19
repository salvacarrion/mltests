import os
import time
import datetime
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.optimizers import SGD, Adam


from covid19.code.classification.da import *
from covid19.code.classification.utils import *
from covid19.code.classification.dataset import *

RUN_NAME = f"resnet50_i256_covid19"

LOSS = tf.keras.losses.BinaryCrossentropy()
METRICS = [tf.keras.metrics.BinaryAccuracy(),
           tf.keras.metrics.AUC(),
           tf.keras.metrics.Precision(),
           tf.keras.metrics.Recall()]
TARGET_SIZE = (224, 224)

def get_model(backbone, target_size=None, freeze_base_model=True):
    istrainable = not freeze_base_model

    if backbone == "resnet50":
        from tensorflow.keras.applications.resnet50 import ResNet50
        from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

        # Instantiate model
        base_model = ResNet50(include_top=True, weights="imagenet")
        base_model.trainable = istrainable  # Freeze base model

        # Create a model on top of the base model
        inputs = tf.keras.layers.Input(shape=(*target_size, 3))  # Same as base_model
        x = base_model(inputs, training=istrainable)
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        model = tf.keras.Model(inputs, outputs)
        return model, preprocess_input, decode_predictions

    else:
        raise ValueError("Unknown backbone")


def unfreeze_model(model):
    # We unfreeze all layers but BatchNorm (to not destroy the non-trainable weights)
    for layer in model.layers:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True


def train(model, train_dataset, val_dataset, batch_size, epochs1, epochs2,
          checkpoints_path=None, last_checkpoint_path=None, logs_path=None,
          plots_path=None, use_multiprocessing=False, workers=1):
    # Build dataloaders
    train_dataloader = Dataloader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = Dataloader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Callbacks
    model_callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-5),
        tf.keras.callbacks.EarlyStopping(patience=10),
        # tf.keras.callbacks.ModelCheckpoint(filepath=checkpoints_path, save_best_only=True, mode='min'),  # It can make the end of an epoch extremely slow
        tf.keras.callbacks.TensorBoard(log_dir=logs_path),
        # WandbCallback(),
    ]

    # Print model
    model.summary()
    
    # Compile the model
    model.compile(optimizer=Adam(learning_rate=1e-3), loss=LOSS, metrics=METRICS)

    # train the model on the new data for a few epochs
    print("Training output layers...")
    history1 = model.fit(train_dataloader, validation_data=val_dataloader, epochs=epochs1, callbacks=model_callbacks,
                         use_multiprocessing=use_multiprocessing, workers=workers)
    print("Initial training results:")
    print(history1)
    if plots_path:
        plot_hist(history1, title="Training output layers", savepath=plots_path, suffix="_initial")

    # Unfreezing layers
    print("Unfreezing layers")
    unfreeze_model(model)

    # we need to recompile the model for these modifications to take effect
    # we use SGD with a low learning rate
    print("Fine-tuning model...")
    model.compile(optimizer=SGD(learning_rate=1e-5, momentum=0.9), loss=LOSS, metrics=METRICS)
    
    # we train our model again (this time fine-tuning the top 2 inception blocks
    # alongside the top Dense layers
    history2 = model.fit(train_dataloader, validation_data=val_dataloader, epochs=epochs2, callbacks=model_callbacks,
                         use_multiprocessing=use_multiprocessing, workers=workers)
    print("Fine-tuning results:")
    print(history2)
    if plots_path:
        plot_hist(history2, title="Fine-tuning full model", savepath=plots_path, suffix="_finetuning")
    
    # Save model
    if last_checkpoint_path:
        print("Saving last model...")
        model.save(last_checkpoint_path)
        print(f"Model saved at: {last_checkpoint_path}")


def test(test_dataset, model_path, batch_size):
    # Build dataloader
    test_dataloader = Dataloader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Load model
    print("Loading model...")
    model = tf.keras.models.load_model(filepath=model_path, compile=False)
    model.summary()

    # Compile the model
    model.compile(loss=LOSS, metrics=METRICS)

    # Evaluate model
    print("Evaluating model...")
    scores = model.evaluate(test_dataloader)
    print("Evaluation results")
    print(scores)
    return scores
    

def get_datasets(csv_dir, images_dir, target_size, prep_fn=None):
    # Get data
    df_train = pd.read_csv(os.path.join(csv_dir, "train.csv"))
    df_val = pd.read_csv(os.path.join(csv_dir, "val.csv"))
    df_test = pd.read_csv(os.path.join(csv_dir, "test.csv"))

    # Build dataset
    train_dataset = Dataset(df_train, images_dir=images_dir, da_fn=tr_da_fn(*target_size),
                            preprocess_fn=prep_fn, target_size=target_size)
    val_dataset = Dataset(df_val, images_dir=images_dir, da_fn=ts_da_fn(*target_size),
                          preprocess_fn=prep_fn, target_size=target_size)
    test_dataset = Dataset(df_test, images_dir=images_dir, da_fn=ts_da_fn(*target_size),
                           preprocess_fn=prep_fn, target_size=target_size)
    
    return train_dataset, val_dataset, test_dataset


def visualize(dataset, i, n=5): 
    for _ in range(n):
        image, original_image = dataset.__getitem__(i, show_originals=True)
        plot_da_image(
            image=image,
            original_image=original_image,
        )


def main(batch_size=32, backbone="resnet50", epochs1=5, epochs2=15, base_path=".", output_path=".",
         target_size=(256, 256), use_multiprocessing=False, workers=1, train_model=True, test_model=True,
         show_samples=False):

    # Vars
    images_dir = os.path.join(base_path, "images256")
    #masks_dir = os.path.join(base_path, "masks256")

    # Outputs
    checkpoints_path = os.path.join(output_path, RUN_NAME, "models")
    last_checkpoint_path = os.path.join(checkpoints_path, "last_model.h5")
    logs_path = os.path.join(output_path, RUN_NAME, "logs")
    plots_path = os.path.join(output_path, RUN_NAME, "plots")

    # Create folders
    for dir_i in [checkpoints_path, logs_path, plots_path]:
        Path(dir_i).mkdir(parents=True, exist_ok=True)

    # Get model + auxiliar functions
    model, prep_fn, _ = get_model(backbone=backbone, target_size=target_size, freeze_base_model=True)

    # Get data
    train_dataset, val_dataset, test_dataset = get_datasets(csv_dir=base_path, images_dir=images_dir,
                                                            target_size=target_size, prep_fn=prep_fn)

    # Visualize
    if show_samples:
        visualize(train_dataset, i=5, n=10)

    # Train
    if train_model:
        train(model, train_dataset, val_dataset, batch_size=batch_size, epochs1=epochs1, epochs2=epochs2,
              checkpoints_path=checkpoints_path, last_checkpoint_path=last_checkpoint_path, logs_path=logs_path,
              plots_path=plots_path, use_multiprocessing=use_multiprocessing, workers=workers)

    # Evaluate
    if test_model:
        test(test_dataset, model_path=last_checkpoint_path, batch_size=batch_size)
    
    
if __name__ == "__main__":
    BASE_PATH = "/home/scarrion/datasets/covid19/front"
    OUTPUT_PATH = "/home/scarrion/projects/mltests/covid19/code/classification/.outputs"

    # Run
    main(train_model=True, test_model=True, show_samples=False,
         target_size=TARGET_SIZE, base_path=BASE_PATH, output_path=OUTPUT_PATH)
    print("Done!")

