import os
import time
from pathlib import Path
import subprocess
import random
import shutil
random.seed(123)

from translation.autonmt import commands

CONDA_ENVNAME = "fairseq"


def fairseq_preprocess(ds_path, src_lang, trg_lang, subword_model, vocab_size, force_overwrite):
    # Create path (if needed)
    toolkit_path = os.path.join(ds_path, "models", "fairseq")
    path = Path(toolkit_path)
    path.mkdir(parents=True, exist_ok=True)

    # Check if data-bin exists
    data_bin_path = os.path.join(toolkit_path, "data-bin", subword_model, str(vocab_size))
    if not force_overwrite and os.path.exists(data_bin_path):
        print("\t=> Skipping preprocessing as it already exists")
    else:
        # Delete folder to prevent errors that are hard to debug
        if force_overwrite and os.path.exists(data_bin_path):
            print(f"The following directory is going to be delete: {data_bin_path}")
            res = input(f"Do you want to continue? [y/N]")
            if res.lower().strip() == 'y':
                print(f"Deleting directory... ({data_bin_path})")
                shutil.rmtree(data_bin_path)
            else:
                print("Deletion cancelled")

        # Create path (needed only because of the fix separator)
        path = Path(data_bin_path)
        path.mkdir(parents=True, exist_ok=True)

        # Copy vocabs and fix separator
        ori_vocab_path = os.path.join(ds_path, "vocabs", "spm", subword_model, str(vocab_size), f"spm_{src_lang}-{trg_lang}.vocabf")
        src_vocab_fairseq_path = os.path.join(data_bin_path, f"dict.{src_lang}.txt")
        trg_vocab_fairseq_path = os.path.join(data_bin_path, f"dict.{trg_lang}.txt")
        shutil.copyfile(ori_vocab_path, src_vocab_fairseq_path)
        shutil.copyfile(ori_vocab_path, trg_vocab_fairseq_path)
        commands.replace_in_file(search_string='\t', replace_string=' ', filename=src_vocab_fairseq_path)
        commands.replace_in_file(search_string='\t', replace_string=' ', filename=trg_vocab_fairseq_path)

        # Preprocess
        data_path = os.path.join(ds_path, "data", "encoded", subword_model, str(vocab_size))
        fairseq_preprocess_with_vocab(data_path, data_bin_path=data_bin_path, src_lang=src_lang, trg_lang=trg_lang, src_vocab_path=src_vocab_fairseq_path, trg_vocab_path=trg_vocab_fairseq_path)


def fairseq_preprocess_with_vocab(data_path, data_bin_path, src_lang, trg_lang, src_vocab_path=None, trg_vocab_path=None, train_fname="train", val_fname="val", test_fname="test"):
    val_fname = f"--validpref {data_path}/{val_fname}" if val_fname else ""
    cmd = f"fairseq-preprocess --source-lang {src_lang} --target-lang {trg_lang} --trainpref {data_path}/{train_fname} {val_fname} --testpref {data_path}/{test_fname} --destdir {data_bin_path} --workers $(nproc)"
    cmd += f" --srcdict {src_vocab_path}" if src_vocab_path else ""
    cmd += f" --tgtdict {trg_vocab_path}" if trg_vocab_path else ""
    subprocess.call(['/bin/bash', '-i', '-c', f"conda activate {CONDA_ENVNAME} && {cmd}"])  # https://stackoverflow.com/questions/12060863/python-subprocess-call-a-bash-alias/25099813


def fairseq_train(data_path, run_name, subword_model, vocab_size, force_overwrite, model_path=None):
    toolkit_path = os.path.join(data_path, "models", "fairseq")
    data_bin_path = os.path.join(toolkit_path, "data-bin", subword_model, str(vocab_size))
    checkpoints_path = os.path.join(toolkit_path, "runs", run_name, "checkpoints")
    logs_path = os.path.join(toolkit_path, "runs", run_name, "logs")

    # Check if data-bin exists
    res = "y"
    if not force_overwrite and os.path.exists(checkpoints_path):
        print("There are checkpoints in this experiment.")
        res = input("Do you want to continue [y/N]? ").strip().lower()

    # Check if we can continue with the training
    if res != "y":
        print("Training cancelled.")
    else:
        wait_seconds = 1
        print(f"[IMPORTANT]: Training overwrite is enabled. (Waiting {wait_seconds} seconds)")
        time.sleep(wait_seconds)

        # Remove last checkpoint
        last_checkpoint_path = os.path.join(checkpoints_path, "checkpoint_last.pt")
        if model_path is None and os.path.exists(last_checkpoint_path):
            print("Removing last checkpoint to train from scratch...")
            os.remove(last_checkpoint_path)

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
            "--lr 0.25",
            "--optimizer nag --clip-norm 0.1",
            "--criterion cross_entropy",
            "--max-tokens 4096",
            #"--batch-size 128",
            "--max-epoch 10",
            # "--clip-norm 0.0",
            # "--update-freq 1",
            # "--warmup-updates 4000",
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
        train_command += [
            "--eval-bleu",
            "--eval-bleu-args '{\"beam\": 5}'",
            "--eval-bleu-detok moses",
            "--eval-bleu-print-samples",
        ]

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

        # Extract sentences from generate-test.txt
        gen_test_path = os.path.join(output_path, "generate-test.txt")
        src_tok_path = os.path.join(output_path, "src.tok")
        ref_tok_path = os.path.join(output_path, "ref.tok")
        hyp_tok_path = os.path.join(output_path, "hyp.tok")
        subprocess.call(['/bin/bash', '-i', '-c', f"grep ^S {gen_test_path} | cut -f2- > {src_tok_path}"])
        subprocess.call(['/bin/bash', '-i', '-c', f"grep ^T {gen_test_path} | cut -f2- > {ref_tok_path}"])
        subprocess.call(['/bin/bash', '-i', '-c', f"grep ^H {gen_test_path} | cut -f3- > {hyp_tok_path}"])

        # Replace "<<unk>>" with "<unk>" in ref.tok
        commands.replace_in_file(search_string="<<unk>>", replace_string="<unk>", filename=ref_tok_path)

        # Detokenize
        src_txt_path = os.path.join(output_path, "src.txt")
        ref_txt_path = os.path.join(output_path, "ref.txt")
        hyp_txt_path = os.path.join(output_path, "hyp.txt")
        commands.spm_decode(spm_model_path, input_file=src_tok_path, output_file=src_txt_path)
        commands.spm_decode(spm_model_path, input_file=ref_tok_path, output_file=ref_txt_path)
        commands.spm_decode(spm_model_path, input_file=hyp_tok_path, output_file=hyp_txt_path)

