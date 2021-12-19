import os
import random
import shutil
import subprocess
from pathlib import Path

random.seed(123)

from translation.autonmt import commands
from translation.custom import preprocessor, trainer

CONDA_ENVNAME = "mltests"
TOOLKIT_NAME = "custom"


def custom_preprocess(ds_path, src_lang, trg_lang, subword_model, vocab_size, force_overwrite, interactive):
    # Create path (if needed)
    toolkit_path = os.path.join(ds_path, "models", TOOLKIT_NAME)
    path = Path(toolkit_path)
    path.mkdir(parents=True, exist_ok=True)

    # Check if data-bin exists
    data_bin_path = os.path.join(toolkit_path, "data-bin", subword_model, str(vocab_size))
    if not force_overwrite and os.path.exists(data_bin_path):
        if not interactive:
            print("\t=> Skipping preprocessing as it already exists")
            return
        else:
            print(f"\t=> The following directory is going to be delete: {data_bin_path}")
            res = input(f"\t=> Do you want to continue? [y/N]")
            if res.lower().strip() == 'y':
                print(f"\t=> Deleting directory... ({data_bin_path})")
                shutil.rmtree(data_bin_path)
            else:
                print("\t=> Operation cancelled")
                print("\t=> Skipping preprocessing as it already exists")
                return

    # Get paths
    data_path = os.path.join(ds_path, "data", "encoded", subword_model, str(vocab_size))
    vocab_path = os.path.join(ds_path, "vocabs", "spm", subword_model, str(vocab_size))
    src_vocab = os.path.join(vocab_path, f"spm_{src_lang}-{trg_lang}.vocab")
    src_spm_model = os.path.join(vocab_path, f"spm_{src_lang}-{trg_lang}.model")

    # Preprocess files
    preprocessor.preprocess(destdir=data_bin_path, src_lang=src_lang, trg_lang=trg_lang,
                            trainpref=os.path.join(data_path, "train"), validpref=os.path.join(data_path, "val"),
                            testpref=os.path.join(data_path, "test"),
                            src_vocab=src_vocab, trg_vocab=src_vocab,
                            src_spm_model=src_spm_model, trg_spm_model=src_spm_model)


def custom_train(data_path, run_name, src_lang, trg_lang, subword_model, vocab_size, model_path, num_gpus,
                 force_overwrite, interactive):
    toolkit_path = os.path.join(data_path, "models", TOOLKIT_NAME)
    data_bin_path = os.path.join(toolkit_path, "data-bin", subword_model, str(vocab_size))
    checkpoints_path = os.path.join(toolkit_path, "runs", run_name, "checkpoints")
    last_checkpoint_path = os.path.join(checkpoints_path, "checkpoint_last.pt")
    logs_path = os.path.join(toolkit_path, "runs", run_name, "logs")

    # Check if data-bin exists
    if os.path.exists(last_checkpoint_path):
        if force_overwrite:
            print("\t=> Removing last checkpoint to train from scratch...")
            os.remove(last_checkpoint_path)
        else:
            if not interactive:
                print("\t=> Skipping training as it already exists")
                return
            else:
                print(f"\t=> There are checkpoints in this experiment: {last_checkpoint_path}")
                res = input("\t=> Do you want to continue [y/N]? ").strip().lower()

                # Check if we can continue with the training
                if res.lower().strip() == 'y':
                    print("\t=> Removing last checkpoint to train from scratch...")
                    os.remove(last_checkpoint_path)
                else:
                    print("\t=> Operation cancelled")
                    print("\t=> Skipping training as it already exists")
                    return

    # Train model
    trainer.train(data_path=data_bin_path, src_lang=src_lang, trg_lang=trg_lang, checkpoints_path=checkpoints_path,
                  logs_path=logs_path)


def custom_translate(data_path, checkpoint_path, output_path, src_lang, trg_lang, subword_model, spm_model_path,
                      force_overwrite, interactive, beam_width=5, max_length=200, score=True, metrics=None):
    # Check checkpoint path
    if not os.path.exists(checkpoint_path):
        raise IOError("\t=> The checkpoint does not exists")

    # Check if there are translations
    if not force_overwrite and os.path.exists(output_path):
        if not interactive:
            print("\t=> Skipping evaluation as it already exists")
            return
        else:
            print(f"\t=> There are translations for this experiment [beam={str(beam_width)}]: {output_path}")
            res = input("\t=> Do you want to continue [y/N]? ").strip().lower()

            # Check if we can continue with the translation
            if res.lower().strip() != 'y':
                print("\t=> Evaluation cancelled.")
                return

    # Translate model
    trainer.translate(data_path, checkpoint_path, output_path, src_lang, trg_lang, beam_width=beam_width,
                      max_length=max_length)

    # Detokenize
    src_tok_path = os.path.join(output_path, "src.tok")
    ref_tok_path = os.path.join(output_path, "ref.tok")
    hyp_tok_path = os.path.join(output_path, "hyp.tok")
    src_txt_path = os.path.join(output_path, "src.txt")
    ref_txt_path = os.path.join(output_path, "ref.txt")
    hyp_txt_path = os.path.join(output_path, "hyp.txt")
    commands.spm_decode(spm_model_path, input_file=src_tok_path, output_file=src_txt_path)
    commands.spm_decode(spm_model_path, input_file=ref_tok_path, output_file=ref_txt_path)
    commands.spm_decode(spm_model_path, input_file=hyp_tok_path, output_file=hyp_txt_path)

    # Score
    if score:
        commands.score_test_files(data_path=output_path, src_lang=src_lang, trg_lang=trg_lang,
                                  metrics=metrics, force_overwrite=force_overwrite)
