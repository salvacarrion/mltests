import datetime
import os
import time
from pathlib import Path
from itertools import islice
from shutil import copyfile
import subprocess
import re
import random
random.seed(123)

import unicodedata
from tqdm import tqdm

from translation.paper1.build_datasets.utils import *

CONDA_ENVNAME = "fairseq"


def fairseq_model(data_path, run_name, eval_name, src_lang, trg_lang, use_pretokenized, force_overwrite):
    # Create path (if needed)
    preprocess_path = os.path.join(data_path, "models", "fairseq")
    path = Path(preprocess_path)
    path.mkdir(parents=True, exist_ok=True)

    # Preprocess files
    # fairseq_preprocess(data_path, src_lang, trg_lang, use_pretokenized, force_overwrite)

    # Train model
    # fairseq_train(data_path, run_name, force_overwrite)

    # Evaluate
    fairseq_translate(data_path, run_name, eval_name, src_lang, trg_lang, force_overwrite, beams=[5])


def fairseq_preprocess(data_path, src_lang, trg_lang, use_pretokenized, force_overwrite):
    # Create path (if needed)
    toolkit_path = os.path.join(data_path, "models", "fairseq")
    path = Path(toolkit_path)
    path.mkdir(parents=True, exist_ok=True)

    # Check if data-bin exists
    data_bin_path = os.path.join(toolkit_path, "data-bin")
    if not force_overwrite and os.path.exists(data_bin_path):
        print("\t=> Skipping preprocessing as it already exists")
    else:
        # Preprocess
        raw_folder = "splits" if not use_pretokenized else "pretokenized"
        train_path = os.path.join(data_path, raw_folder, "train")
        val_path = os.path.join(data_path, raw_folder, "val")
        test_path = os.path.join(data_path, raw_folder, "test")

        # Run command
        preprocess_cmd = f"fairseq-preprocess --source-lang {src_lang} --target-lang {trg_lang} --trainpref {train_path} --validpref {val_path} --testpref {test_path} --destdir {data_bin_path}"
        subprocess.call(['/bin/bash', '-i', '-c', f"conda activate {CONDA_ENVNAME} && {preprocess_cmd}"])  # https://stackoverflow.com/questions/12060863/python-subprocess-call-a-bash-alias/25099813


def fairseq_train(data_path, run_name, force_overwrite):
    toolkit_path = os.path.join(data_path, "models", "fairseq")
    data_bin_path = os.path.join(toolkit_path, "data-bin")
    checkpoints_path = os.path.join(toolkit_path, "runs", run_name, "checkpoints")
    logs_path = os.path.join(toolkit_path, "runs", run_name, "logs")

    # Check if data-bin exists
    res = "y"
    if False and not force_overwrite and os.path.exists(checkpoints_path):
        print("There are checkpoints in this experiment.")
        res = input("Do you want to continue [y/N]? ").strip().lower()

    # Check if we can continue with the training
    if res != "y":
        print("Training cancelled.")
    else:
        wait_seconds = 1
        print(f"[IMPORTANT]: Training overwrite is enabled. (Waiting {wait_seconds} seconds)")
        time.sleep(wait_seconds)

        # Write command
        train_command = [f"fairseq-train {data_bin_path}"]

        # Add model
        train_command += [
            "--arch transformer",
            "--encoder-embed-dim 256",
            "--decoder-embed-dim 256",
            "--encoder-layers 3",
            "--decoder-layers 3",
            "--encoder-attention-heads 8",
            "--decoder-attention-heads 8",
            "--encoder-ffn-embed-dim 512",
            "--decoder-ffn-embed-dim 512",
            "--dropout 0.1",
        ]

        # Add training stuff
        train_command += [
            "--lr 0.001",
            "--optimizer adam",
            "--criterion cross_entropy",
            "--batch-size 128",
            "--max-epoch 10",
            "--clip-norm 0.0",
            "--update-freq 1",
            "--warmup-updates 4000",
            "--patience 10",
            "--seed 1234",
            #"--max-tokens 4096",
            #"--lr-scheduler reduce_lr_on_plateau",
           ]

        # Add checkpoint stuff
        train_command += [
            f"--save-dir {checkpoints_path}",
            "--no-epoch-checkpoints",
            "--maximize-best-checkpoint-metric",
            #"--best-checkpoint-metric bleu",
            ]

        # Add evaluation stuff
        # train_command += [
        #     "--eval-bleu",
        #     "--eval-bleu-args '{\"beam\": 5}'",
        #     "--eval-bleu-detok moses",
        #     "--eval-bleu-print-samples",
        # ]

        # Logs and stuff
        train_command += [
            "--log-format simple",
            f"--tensorboard-logdir {logs_path}",
            "--task translation",
            "--num-workers $(nproc)",
        ]

        # Run command
        train_command = " ".join(train_command)
        subprocess.call(['/bin/bash', '-i', '-c', f"conda activate {CONDA_ENVNAME} && {train_command}"])  # https://stackoverflow.com/questions/12060863/python-subprocess-call-a-bash-alias/25099813


def fairseq_translate(data_path, checkpoint_path, output_path, src_lang, trg_lang, subword_model, spm_model_path,
                      force_overwrite, beam_width=5, max_length=200):
    # Check if data-bin exists
    res = "y"
    if False and not force_overwrite and os.path.exists(output_path):
        print("There are evaluations in this experiment.")
        res = input("Do you want to continue [y/N]? ").strip().lower()

    # Check if we can continue with the training
    if res != "y":
        print("Evaluation cancelled.")
    else:
        wait_seconds = 1
        if wait_seconds > 0:
            print(f"[IMPORTANT]: Evaluation overwrite is enabled. (Waiting {wait_seconds} seconds)")
            time.sleep(wait_seconds)

        # Write command
        gen_command = [f"fairseq-generate {data_path}"]

        # Add stuff
        gen_command += [
            f"--source-lang {src_lang}",
            f"--target-lang {trg_lang}",
            f"--path {checkpoint_path}",
            f"--results-path {output_path}",
            f"--beam {beam_width}",
            f"--max-len-a {0}",  # max_len = ax+b
            f"--max-len-b {max_length}",
            f"--nbest 1",
            "--skip-invalid-size-inputs-valid-test",
            "--scoring sacrebleu",
            "--num-workers $(nproc)",
        ]

        # Run command
        gen_command = " ".join(gen_command)
        subprocess.call(['/bin/bash', '-i', '-c', f"conda activate {CONDA_ENVNAME} && {gen_command}"])  # https://stackoverflow.com/questions/12060863/python-subprocess-call-a-bash-alias/25099813

        # Parse output file
        gen_test_path = os.path.join(output_path, "generate-test.txt")
        src_tok_path = os.path.join(output_path, "src.tok")
        ref_tok_path = os.path.join(output_path, "ref.tok")
        hyp_tok_path = os.path.join(output_path, "hyp.tok")
        subprocess.call(['/bin/bash', '-i', '-c', f"grep ^S {gen_test_path} | cut -f2- > {src_tok_path}"])
        subprocess.call(['/bin/bash', '-i', '-c', f"grep ^T {gen_test_path} | cut -f2- > {ref_tok_path}"])
        subprocess.call(['/bin/bash', '-i', '-c', f"grep ^H {gen_test_path} | cut -f3- > {hyp_tok_path}"])

        # Detokenize
        src_txt_path = os.path.join(output_path, "src.txt")
        ref_txt_path = os.path.join(output_path, "ref.txt")
        hyp_txt_path = os.path.join(output_path, "hyp.txt")
        if subword_model == "word":
            src_cmd = f"sacremoses -l {src_lang} -j$(nproc) detokenize < {src_tok_path} > {src_txt_path}"
            ref_cmd = f"sacremoses -l {trg_lang} -j$(nproc) detokenize < {ref_tok_path} > {ref_txt_path}"
            hyp_cmd = f"sacremoses -l {trg_lang} -j$(nproc) detokenize < {hyp_tok_path} > {hyp_txt_path}"
        else:
            src_cmd = f"spm_decode --model={spm_model_path} --input_format=piece < {src_tok_path} > {src_txt_path}"
            ref_cmd = f"spm_decode --model={spm_model_path} --input_format=piece < {ref_tok_path} > {ref_txt_path}"
            hyp_cmd = f"spm_decode --model={spm_model_path} --input_format=piece < {hyp_tok_path} > {hyp_txt_path}"
        subprocess.call(['/bin/bash', '-i', '-c', f"conda activate {CONDA_ENVNAME} && {src_cmd}"])
        subprocess.call(['/bin/bash', '-i', '-c', f"conda activate {CONDA_ENVNAME} && {ref_cmd}"])
        subprocess.call(['/bin/bash', '-i', '-c', f"conda activate {CONDA_ENVNAME} && {hyp_cmd}"])
