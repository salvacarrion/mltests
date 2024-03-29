from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow.keras.optimizers import SGD, Adam

from da import *
from dataset import Dataset, Dataloader
from helpers import *
from utils import *

# tf.config.experimental_run_functions_eagerly(True)

# Vars
SHOW_PLOTS = False
SHOW_DA_SAMPLES = False
FINETUNE_ONLY = False
TAB_INPUT = False
ADD_NORMAL_CLS = True
USE_CROPS = True

if FINETUNE_ONLY:
    PATIENCE = 15
    WAIT_EPOCH_WARMUP = 20
else:
    PATIENCE = 15
    WAIT_EPOCH_WARMUP = 30


def train(model, train_dataset, val_dataset, batch_size, epochs1, epochs2,
          checkpoints_path=None, logs_path=None,
          plots_path=None, use_multiprocessing=False, workers=1, single_output_idx=None):
    # Build dataloaders
    train_dataloader = Dataloader(train_dataset, batch_size=batch_size, shuffle=True,
                                  single_output_idx=single_output_idx, multiple_inputs=TAB_INPUT,
                                  add_normal_cls=ADD_NORMAL_CLS)
    val_dataloader = Dataloader(val_dataset, batch_size=batch_size, shuffle=False, single_output_idx=single_output_idx,
                                multiple_inputs=TAB_INPUT, add_normal_cls=ADD_NORMAL_CLS)

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
    model.compile(optimizer=Adam(learning_rate=LR_EPOCH1), loss=get_losses(),
                  metrics=get_metrics(single_output_idx, add_normal=ADD_NORMAL_CLS))

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
        model.compile(optimizer=SGD(learning_rate=LR_EPOCH2, momentum=0.9), loss=get_losses(),
                      metrics=get_metrics(single_output_idx, add_normal=ADD_NORMAL_CLS))

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


def test(test_dataset, checkpoints_path, batch_size, single_output_idx=None):
    # Build dataloader
    test_dataloader = Dataloader(test_dataset, batch_size=batch_size, shuffle=False,
                                 single_output_idx=single_output_idx, multiple_inputs=TAB_INPUT,
                                 add_normal_cls=ADD_NORMAL_CLS)

    # Load model
    print("Loading best model...")
    model = tf.keras.models.load_model(filepath=checkpoints_path, compile=False)  # Loads best model automatically
    model.summary()

    # Compile the model
    model.compile(loss=get_losses(), metrics=get_metrics(single_output_idx, add_normal=ADD_NORMAL_CLS))

    # Evaluate model
    print("Evaluating model...")
    scores = model.evaluate(test_dataloader)
    print("Evaluation results")
    print(scores)
    return scores


def trunc_df(df, single_output_idx):
    if single_output_idx == 0:
        return df[((df.infiltrates == 0) & (df.pneumonia == 0) & (df.covid19 == 0)) | (
                (df.infiltrates == 1) & (df.pneumonia == 0) & (df.covid19 == 0))]
    elif single_output_idx == 1:
        return df[((df.infiltrates == 0) & (df.pneumonia == 0) & (df.covid19 == 0)) | (
                (df.infiltrates == 0) & (df.pneumonia == 1) & (df.covid19 == 0))]
    elif single_output_idx == 2:
        return df[((df.infiltrates == 0) & (df.pneumonia == 0) & (df.covid19 == 0)) | (
                (df.infiltrates == 0) & (df.pneumonia == 0) & (df.covid19 == 1))]
    else:
        raise ValueError("Invalid 'single_output_idx'")


def get_datasets(csv_dir, images_dir, target_size, prep_fn=None, trunc_data=False, single_output_idx=None,
                 tab_input=False):
    # Get data
    df_train = pd.read_csv(os.path.join(csv_dir, "train.csv"))
    df_val = pd.read_csv(os.path.join(csv_dir, "val.csv"))
    df_test = pd.read_csv(os.path.join(csv_dir, "test.csv"))

    # Truncate data
    if trunc_data:
        df_train = trunc_df(df_train, single_output_idx)
        df_val = trunc_df(df_val, single_output_idx)
        df_test = trunc_df(df_test, single_output_idx)

    # Build dataset
    train_dataset = Dataset(df_train, images_dir=images_dir, da_fn=tr_da_fn(*target_size),
                            preprocess_fn=prep_fn, target_size=target_size, tab_input=tab_input)
    val_dataset = Dataset(df_val, images_dir=images_dir, da_fn=ts_da_fn(*target_size),
                          preprocess_fn=prep_fn, target_size=target_size, tab_input=tab_input)
    test_dataset = Dataset(df_test, images_dir=images_dir, da_fn=ts_da_fn(*target_size),
                           preprocess_fn=prep_fn, target_size=target_size, tab_input=tab_input)

    print("****** Stats: **********************")
    print_stats(train_dataset, title="Train stats:")
    print_stats(val_dataset, title="Val stats:")
    print_stats(test_dataset, title="Test stats:")

    return train_dataset, val_dataset, test_dataset


def print_stats(dataset, title="Stats:"):
    total = len(dataset)
    infiltrates = sum(dataset.target_infiltrates)
    pneumonia = sum(dataset.target_pneumonia)
    covid19 = sum(dataset.target_covid19)
    normal = int((np.sum(np.stack(
        [np.array(dataset.target_infiltrates), np.array(dataset.target_pneumonia), np.array(dataset.target_covid19)],
        axis=1), axis=1) == 0).astype(np.int64).sum())

    # Print stats
    print(title)
    print("\t- Infiltrates: {} of {} ({:.2f}%)".format(infiltrates, total, infiltrates / total * 100))
    print("\t- Pneumonia: {} of {} ({:.2f}%)".format(pneumonia, total, pneumonia / total * 100))
    print("\t- Covid19: {} of {} ({:.2f}%)".format(covid19, total, covid19 / total * 100))

    # Add normal class
    if ADD_NORMAL_CLS:
        print("\t- Normal: {} of {} ({:.2f}%)".format(normal, total, normal / total * 100))


def main(backbone, input_size, target_size, batch_size, epochs1, epochs2=0, base_path=".", output_path=".",
         use_multiprocessing=False, workers=1, train_model=True, test_model=True,
         show_da_samples=False, run_name="run", trunc_data=False, single_output_idx=None, model_path=None):
    # Vars
    img_dir_name = f"images{input_size}_crop" if USE_CROPS else f"images{input_size}"
    images_dir = os.path.join(base_path, img_dir_name)
    # masks_dir = os.path.join(base_path, f"masks{input_size}")

    # Outputs
    checkpoints_path = os.path.join(output_path, run_name, "models")
    logs_path = os.path.join(output_path, run_name, "logs")
    plots_path = os.path.join(output_path, run_name, "plots")

    # Create folders
    for dir_i in [checkpoints_path, logs_path, plots_path]:
        Path(dir_i).mkdir(parents=True, exist_ok=True)

    # Get model + auxiliar functions
    classes = 3 + int(ADD_NORMAL_CLS) if single_output_idx is None else 1
    if not model_path:
        model, prep_fn = get_model(backbone=backbone, classes=classes, target_size=target_size, freeze_base_model=True)
    else:
        print("Loading best model...")
        _, prep_fn = get_model(backbone=backbone, ignore_model=True)
        model = tf.keras.models.load_model(filepath=model_path, compile=False)  # Loads best model automatically

        # Freeze base model
        if TAB_INPUT:
            for layer in model.layers:
                layer.trainable = False
        else:
            for layers in model.layers[1].layers:
                layers.trainable = False

    # Add tabular input
    if TAB_INPUT:
        model = add_tabular_input(model, classes)

    # Get data
    train_dataset, val_dataset, test_dataset = get_datasets(csv_dir=base_path, images_dir=images_dir,
                                                            target_size=target_size, prep_fn=prep_fn,
                                                            trunc_data=trunc_data, single_output_idx=single_output_idx,
                                                            tab_input=TAB_INPUT)

    # Visualize
    if show_da_samples:
        visualize_da(train_dataset, i=5, n=10)

    # Train
    if train_model:
        train(model, train_dataset, val_dataset, batch_size=batch_size, epochs1=epochs1, epochs2=epochs2,
              checkpoints_path=checkpoints_path, logs_path=logs_path,
              plots_path=plots_path, use_multiprocessing=use_multiprocessing, workers=workers,
              single_output_idx=single_output_idx)

    # Evaluate
    if test_model:
        test(test_dataset, checkpoints_path=checkpoints_path, batch_size=batch_size,
             single_output_idx=single_output_idx)


if __name__ == "__main__":
    BASE_PATH = "/home/scarrion/datasets/covid19/front"
    OUTPUT_PATH = "/home/scarrion/projects/mltests/covid19/code/classification/.outputs"

    BACKBONE = "resnet101v2"  # "efficientnetb0"  #"resnet101v2"
    BATCH_SIZE = 32
    INPUT_SIZE = 512
    TARGET_SIZE = (INPUT_SIZE, INPUT_SIZE)
    EPOCHS1 = 250
    EPOCHS2 = 0  # Careful when unfreezing. More gradients, more memory.
    LR_EPOCH1 = 10e-3
    LR_EPOCH2 = 10e-5
    UNFREEZE_N = 0
    NAME_AUX = ""

    for SINGLE_OUTPUT_IDX in [2]:
        for TRUNCATE_DATA in [False, True]:
            model_path = None

            # Skip this run
            if SINGLE_OUTPUT_IDX is None and TRUNCATE_DATA:
                print("Skipping run: SINGLE_OUTPUT_IDX==None AND TRUNCATE_DATA==True")
                continue

            if FINETUNE_ONLY:
                old_run_name = f"{BACKBONE}_" \
                               f"batch{BATCH_SIZE}_" \
                               f"inputsize{INPUT_SIZE}_" \
                               f"targetsize{TARGET_SIZE[0]}x{TARGET_SIZE[1]}_" \
                               f"output-{'all4' if SINGLE_OUTPUT_IDX is None else SINGLE_OUTPUT_IDX}_" \
                               f"1ep{250}_2ep{0}_" \
                               f"unfreeze{0}_" \
                               f"truncdata{str(TRUNCATE_DATA)}" \
                               f""
                model_path = os.path.join(OUTPUT_PATH, old_run_name, "models")
                print(f"Model path: {model_path}")

            # Set name
            RUN_NAME = f"{BACKBONE}_" \
                       f"batch{BATCH_SIZE}_" \
                       f"inputsize{INPUT_SIZE}_" \
                       f"targetsize{TARGET_SIZE[0]}x{TARGET_SIZE[1]}_" \
                       f"output-{f'all{3 + int(ADD_NORMAL_CLS)}' if SINGLE_OUTPUT_IDX is None else SINGLE_OUTPUT_IDX}_" \
                       f"1ep{EPOCHS1}_2ep{EPOCHS2}_" \
                       f"unfreeze{str(UNFREEZE_N)}_" \
                       f"truncdata{str(TRUNCATE_DATA)}" \
                       f"{'_cropped' if USE_CROPS else ''}" \
                       f"{NAME_AUX}"

            print(f"##################################################")
            print(f"##################################################")
            print(f"MODEL NAME: {RUN_NAME}")
            print(f"##################################################")
            print(f"##################################################")

            # Run
            main(backbone=BACKBONE, input_size=INPUT_SIZE, target_size=TARGET_SIZE, batch_size=BATCH_SIZE,
                 epochs1=EPOCHS1, epochs2=EPOCHS2, base_path=BASE_PATH, output_path=OUTPUT_PATH,
                 use_multiprocessing=False, workers=1, train_model=True, test_model=True,
                 show_da_samples=SHOW_DA_SAMPLES, run_name=RUN_NAME,
                 trunc_data=TRUNCATE_DATA, single_output_idx=SINGLE_OUTPUT_IDX, model_path=model_path)
            print("Done!")

    # # Create a MirroredStrategy.
    # strategy = tf.distribute.MirroredStrategy()
    # print("Number of devices: {}".format(strategy.num_replicas_in_sync))
    #
    # # Open a strategy scope.
    # with strategy.scope():
