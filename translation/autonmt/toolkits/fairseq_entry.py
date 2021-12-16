import os
import time
from pathlib import Path
import subprocess
import random
import shutil
random.seed(123)

from translation.autonmt import commands

CONDA_ENVNAME = "fairseq"


def fairseq_preprocess(ds_path, src_lang, trg_lang, subword_model, vocab_size, force_overwrite, interactive):
    # Create path (if needed)
    toolkit_path = os.path.join(ds_path, "models", "fairseq")
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


def fairseq_train(data_path, run_name, subword_model, vocab_size, model_path, num_gpus, force_overwrite, interactive):
    toolkit_path = os.path.join(data_path, "models", "fairseq")
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
        "--max-tokens 4096",
        # "--batch-size 128",
        "--max-epoch 10",
        "--clip-norm 1.0",
        "--update-freq 1",
        "--patience 10",
        "--seed 1234",
        # "--warmup-updates 4000",
        #"--lr-scheduler reduce_lr_on_plateau",
       ]

    # Add checkpoint stuff
    train_command += [
        f"--save-dir {checkpoints_path}",
        "--no-epoch-checkpoints",
        "--maximize-best-checkpoint-metric",
        "--best-checkpoint-metric bleu",
        ]

    # Add evaluation stuff
    train_command += [
        "--eval-bleu",
        "--eval-bleu-args '{\"beam\": 5}'",
        "--eval-bleu-print-samples",
        "--scoring sacrebleu",
    ]

    # Logs and stuff
    train_command += [
        "--log-format simple",
        f"--tensorboard-logdir {logs_path}",
        "--task translation",
        # "--num-workers $(nproc)",
    ]

    # Run command
    train_command = " ".join(train_command)
    num_gpus = f"CUDA_VISIBLE_DEVICES={','.join([str(i) for i in range(num_gpus)])}" if num_gpus else ""
    subprocess.call(['/bin/bash', '-i', '-c', f"conda activate {CONDA_ENVNAME} && {num_gpus} {train_command}"])  # https://stackoverflow.com/questions/12060863/python-subprocess-call-a-bash-alias/25099813


def fairseq_translate(data_path, checkpoint_path, output_path, src_lang, trg_lang, subword_model, spm_model_path,
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
        "--scoring sacrebleu",
        "--skip-invalid-size-inputs-valid-test",
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


    # Score
    if score:
        commands.score_test_files(data_path=output_path, src_lang=src_lang, trg_lang=trg_lang,
                                  metrics=metrics, force_overwrite=force_overwrite)