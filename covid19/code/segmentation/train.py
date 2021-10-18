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


def train(train_dataset, val_dataset, batch_size, backbone, epochs1, epochs2,
          checkpoints_path, last_checkpoint_path=None, logs_path=None, plots_path=None,
          use_multiprocessing=False, workers=1):
    # Build dataloaders
    train_dataloader = Dataloder(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = Dataloder(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Callbacks
    model_callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-5),
        tf.keras.callbacks.EarlyStopping(patience=10),
        tf.keras.callbacks.ModelCheckpoint(filepath=checkpoints_path, save_best_only=True, mode='min'),
        tf.keras.callbacks.TensorBoard(log_dir=logs_path),
        # WandbCallback(),
    ]

    # define model
    model = Unet(backbone, encoder_weights='imagenet', classes=1, activation='sigmoid', encoder_freeze=True)
    model.summary()
    
    # Compile the model
    model.compile(optimizer=Adam(learning_rate=1e-3), loss=bce_jaccard_loss, metrics=[iou_score])
    
    # train the model on the new data for a few epochs
    print("Training decoder first...")
    history1 = model.fit(train_dataloader, validation_data=val_dataloader, epochs=epochs1, callbacks=model_callbacks,
                         use_multiprocessing=use_multiprocessing, workers=workers)
    print("Initial training results:")
    print(history1)
    if plots_path:
        plot_hist(history1, title="Training decoder", savepath=plots_path, suffix="_initial")
    
    # we need to recompile the model for these modifications to take effect
    # we use SGD with a low learning rate
    print("Fine-tuning model...")
    set_trainable(model, recompile=False)
    model.compile(optimizer=SGD(learning_rate=1e-4, momentum=0.9), loss=bce_jaccard_loss, metrics=[iou_score])
    
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
    test_dataloader = Dataloder(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Load model
    print("Loading model...")
    model = tf.keras.models.load_model(filepath=model_path, compile=False)
    model.summary()

    # Compile the model
    model.compile(loss=bce_jaccard_loss, metrics=[iou_score])

    # Evaluate model
    print("Evaluating model...")
    scores = model.evaluate(test_dataloader)
    print("Evaluation results")
    print(scores)
    return scores
    

def get_datasets(filename, images_dir, masks_dir, backbone, target_size):
    # Get data
    df_train, df_val, df_test = get_splits(filename=filename, filter_masks=True,
                                           masks_path=os.path.join(masks_dir, "*.png"))

    # Preprocessing
    prep_fn = preprocessing_fn(custom_fn=sm.get_preprocessing(backbone))

    # Build dataset
    train_dataset = Dataset(df_train, imgs_dir=images_dir, masks_dir=masks_dir, da_fn=tr_da_fn(*target_size),
                            preprocess_fn=prep_fn, target_size=target_size)
    val_dataset = Dataset(df_val, imgs_dir=images_dir, masks_dir=masks_dir, da_fn=ts_da_fn(*target_size),
                          preprocess_fn=prep_fn, target_size=target_size)
    test_dataset = Dataset(df_test, imgs_dir=images_dir, masks_dir=masks_dir, da_fn=ts_da_fn(*target_size),
                           preprocess_fn=prep_fn, target_size=target_size)
    
    return train_dataset, val_dataset, test_dataset


def visualize(dataset, i, n=5): 
    for _ in range(n):
        image, mask, original_image, original_mask = dataset.__getitem__(i, show_originals=True)
        plot_4x4(
            image=image,
            mask=mask,
            original_image=original_image,
            original_mask=original_mask,
        )


def main(batch_size=32, backbone="resnet34", epochs1=1, epochs2=1, base_path=".", output_path=".",
         target_size=(256, 256), use_multiprocessing=True, workers=8, train_model=True, test_model=True,
         show_samples=False):
    # Vars
    filename_csv = os.path.join(base_path, "data.csv")
    images_dir = os.path.join(base_path, "images256")
    masks_dir = os.path.join(base_path, "masks256")
    checkpoints_path = os.path.join(output_path, "models")
    last_checkpoint_path = os.path.join(checkpoints_path, "my_last_model2.h5")
    logs_path = os.path.join(output_path, "logs")
    plots_path = os.path.join(output_path, "plots")

    # Get data
    train_dataset, val_dataset, test_dataset = get_datasets(filename=filename_csv,
                                                            images_dir=images_dir, masks_dir=masks_dir,
                                                            backbone=backbone, target_size=target_size)

    # Visualize
    if show_samples:
        visualize(train_dataset, i=5, n=10)

    # Train
    if train_model:
        train(train_dataset, val_dataset, batch_size=batch_size, backbone=backbone, epochs1=epochs1, epochs2=epochs2,
              checkpoints_path=checkpoints_path, last_checkpoint_path=last_checkpoint_path, logs_path=logs_path,
              plots_path=plots_path, use_multiprocessing=use_multiprocessing, workers=workers)

    # Evaluate
    if test_model:
        test(test_dataset, model_path=last_checkpoint_path, batch_size=batch_size)
    
    
if __name__ == "__main__":
    BASE_PATH = "/home/scarrion/datasets/covid19/front"
    OUTPUT_PATH = "/home/scarrion/projects/mltests/covid19/code/.outputs"

    # Run
    main(train_model=True, test_model=True, show_samples=True, base_path=BASE_PATH, output_path=OUTPUT_PATH)
    print("Done!")

