import os
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


def create_splits(base_path, datasets, val_size=5000, test_size=5000, shuffle=True):
    print("Creating splits...")

    # LEVEL 0: Dataset names
    for ds in datasets:  # Dataset
        ds_name = ds["name"]

        # LEVEL 1: Sizes
        for ds_size_name, ds_max_lines in [("original", None)]:  # Lengths

            # LEVEL 2: Languages
            for lang_pair in ds["languages"]:  # Languages
                src_lang, trg_lang = lang_pair.split("-")

                # Read raw files
                raw_path = os.path.join(base_path, ds_name, ds_size_name, lang_pair, "raw")
                print(f"\t=> Processing raw files: {raw_path}")
                with open(os.path.join(raw_path, f"data.{src_lang}"), 'r') as f:
                    src_lines = f.readlines()
                with open(os.path.join(raw_path, f"data.{trg_lang}"), 'r') as f:
                    trg_lines = f.readlines()

                # Clean lines
                lines = preprocess_pairs(src_lines, trg_lines, shuffle=shuffle)

                # Check size type
                val_size = int(val_size*len(lines)) if isinstance(val_size, float) else val_size
                test_size = int(test_size*len(lines)) if isinstance(test_size, float) else test_size

                # Create partitions
                train_lines = lines[:-(val_size+test_size)]
                val_lines = lines[-(val_size+test_size):-test_size]
                test_lines = lines[-test_size:]
                splits = [(train_lines, "train"), (val_lines, "val"), (test_lines, "test")]

                # Create splits folder
                splits_path = os.path.join(base_path, ds_name, ds_size_name, lang_pair, "splits")
                path = Path(splits_path)
                path.mkdir(parents=True, exist_ok=True)

                # Save partitions
                for split_lines, split_name in splits:
                    for i, split_lang in enumerate([src_lang, trg_lang]):  # Languages
                        savepath = os.path.join(splits_path, f"{split_name}.{split_lang}")
                        with open(savepath, 'w') as fs:
                            lines = [line[i] + '\n' for line in split_lines]  # split_lines is a tuple (src, trg)
                            fs.writelines(lines)


def create_reduced_versions(base_path, datasets, autofix=False):
    print("Creating reduced versions...")

    # LEVEL 0: Dataset names
    for ds in datasets:  # Dataset
        ds_name = ds["name"]

        # LEVEL 1: Sizes
        for ds_size_name, ds_max_lines in ds["sizes"]:  # Lengths

            # LEVEL 2: Languages
            for lang_pair in ds["languages"]:  # Languages
                src_lang, trg_lang = lang_pair.split("-")
                trans_files = get_translation_files(src_lang, trg_lang)

                if ds_size_name == "original":  # Check for splits
                    for trans_fname in trans_files:
                        # Check level 3.1: Translation files
                        ds_level = os.path.join(ds_name, ds_size_name, lang_pair, "splits", trans_fname)
                        fname = os.path.join(base_path, ds_level)
                        if not os.path.exists(fname):
                            print(f"- Checking file '{ds_level}' => Failed")
                            continue

                else:  # Create folder
                    # Create folders (recursively)
                    ds_level = os.path.join(ds_name, ds_size_name, lang_pair, "splits")
                    ds_level_path = os.path.join(base_path, ds_level)

                    if not os.path.exists(ds_level_path):
                        if not autofix:
                            print(f"- Checking dataset '{ds_level}' => Failed")
                            continue
                        else:
                            # Create folder recursively
                            if not os.path.exists(ds_level_path):
                                print(f"- Autofix for '{ds_level}': Creating directories...")
                                path = Path(ds_level_path)
                                path.mkdir(parents=True, exist_ok=True)

                            # Add trunc file names
                            for trans_fname in trans_files:
                                # Check level 3.1: Translation files
                                ori_filename = os.path.join(base_path, ds_name, "original", lang_pair, "splits", trans_fname)
                                new_filename = os.path.join(base_path, ds_name, ds_size_name, lang_pair, "splits", trans_fname)

                                # Copy n lines efficiently
                                if not os.path.exists(new_filename):
                                    with open(ori_filename, 'r') as fin, open(new_filename, 'w') as fout:
                                        lines = list(islice(fin, ds_max_lines))
                                        fout.writelines(lines)
                                        print(f"- Autofix for '{ds_level}': Creating {trans_fname}...")

    return True


def pretokenize(base_path, datasets, force_overwrite=False):
    print("Pretokenizing splits...")

    # 1. Normalization => Whitespace, Strip accents, Unicode normalization,...
    # 2. Pre-Tokenization => Split text into "words" (hello world !)
    # 3. (subword) Tokenizer => BPE, Unigram, WordPiece,...
    # 4. Post-Processing => <unk>, <sos>, <eos>, max lengths,...
    # Here: Normalization (Python) + Pre-Tokenization (sacremoses) + BPE (fastBPE vs. subword-nmt)
    for ds in datasets:  # Dataset
        ds_name = ds["name"]
        for ds_size_name, ds_max_lines in ds["sizes"]:  # Lengths
            for lang_pair in ds["languages"]:  # Languages
                src_lang, trg_lang = lang_pair.split("-")
                trans_files = get_translation_files(src_lang, trg_lang)

                # Create folder
                pretokenize_path = os.path.join(base_path, ds_name, ds_size_name, lang_pair, "pretokenized")
                path = Path(pretokenize_path)
                path.mkdir(parents=True, exist_ok=True)

                # Process each file
                print(f"\t=> Pretokenizing: {pretokenize_path}")
                for trans_fname in trans_files:
                    file_lang = trans_fname.split(".")[-1]
                    ori_filename = os.path.join(base_path, ds_name, ds_size_name, lang_pair, "splits", trans_fname)
                    new_filename = os.path.join(pretokenize_path, trans_fname)

                    # Check if the pretokenized files already exists
                    if not force_overwrite and os.path.exists(new_filename):
                        print("\t\t=> Skipping pretokenization as this file already exists")
                    else:
                        # Tokenize (sacremoses): https://github.com/alvations/sacremoses
                        command = f"sacremoses -j$(nproc) -l {file_lang} tokenize < {ori_filename} > {new_filename}"
                        subprocess.call(['/bin/bash', '-i', '-c', command])  # https://stackoverflow.com/questions/12060863/python-subprocess-call-a-bash-alias/25099813


def build_vocab(base_path, datasets, use_pretokenized=False, merge_trains=True,
                subword_model="unigram", vocab_size=8000, character_coverage=1.0, force_overwrite=False):
    print(f"- Building vocabs: (subword_model={subword_model}; vocab_size={vocab_size})")

    for ds in datasets:  # Dataset
        ds_name = ds["name"]
        for ds_size_name, ds_max_lines in ds["sizes"]:  # Lengths
            for lang_pair in ds["languages"]:  # Languages
                src_lang, trg_lang = lang_pair.split("-")
                trans_files = get_translation_files(src_lang, trg_lang)

                # Create dirs
                vocab_path = os.path.join(base_path, ds_name, ds_size_name, lang_pair, "vocabs", "spm")
                vocab_data_path = os.path.join(vocab_path, "data")
                model_vocab_path = os.path.join(vocab_path, subword_model, str(vocab_size))
                print(f"\t=> Build vocab for: {vocab_path}")
                for p in [vocab_path, vocab_data_path, model_vocab_path]:
                    path = Path(p)
                    path.mkdir(parents=True, exist_ok=True)

                # Create joined trained
                tr_fname = "train_pretok.txt" if use_pretokenized else "train_raw.txt"
                new_filename = os.path.join(vocab_data_path, tr_fname)

                # Concatenate train files
                if not merge_trains:
                    raise NotImplementedError("Only merge train files is allowed")
                else:
                    # Check if concatenated train file exists
                    if not force_overwrite and os.path.isfile(new_filename):
                        print("\t\t=> Skipping concatenated train file as it already exists")
                        # raise IOError("Concatenated train file exists")
                    else:
                        # Concat train files
                        with open(new_filename, 'w') as outfile:
                            for trans_fname in [f"train.{src_lang}", f"train.{trg_lang}"]:
                                raw_folder = "splits" if not use_pretokenized else "pretokenized"
                                ori_filename = os.path.join(base_path, ds_name, ds_size_name, lang_pair, raw_folder, trans_fname)
                                with open(ori_filename) as infile:
                                    outfile.write(infile.read())

                # Learn model
                model_prefix = os.path.join(model_vocab_path, f"spm_{src_lang}-{trg_lang}")
                if not force_overwrite and os.path.exists(model_prefix+".vocab"):
                    print("\t\t=> Skipping spm training as it already exists")
                else:
                    command = f"spm_train --input={new_filename} --model_prefix={model_prefix} --vocab_size={vocab_size} --character_coverage={character_coverage} --model_type={subword_model}"
                    subprocess.call(['/bin/bash', '-i', '-c', command])  # https://stackoverflow.com/questions/12060863/python-subprocess-call-a-bash-alias/25099813


def main(base_path, datasets, use_pretokenized=False, force_overwrite=False):
    # Split raw data
    # create_splits(base_path, datasets, val_size=0.15, test_size=0.15)

    # Create reduced versions
    create_reduced_versions(base_path, datasets, autofix=True)

    # Pretokenize (sacremoses)
    if use_pretokenized:
        pretokenize(base_path, datasets, force_overwrite=force_overwrite)

    # Create vocabs
    for subword_model in ["word"]:  # unigram, bpe, char, or word
        for vocab_size in [16000]:
            flag_pretok = (subword_model == "word" or use_pretokenized)
            build_vocab(base_path, datasets, subword_model=subword_model, vocab_size=vocab_size,
                        use_pretokenized=flag_pretok, force_overwrite=force_overwrite)


if __name__ == "__main__":
    # Download datasets: https://opus.nlpl.eu/

    # BASE_PATH = "/Users/salvacarrion/Documents/Programming/Datasets/nn/translation"
    BASE_PATH = "/home/scarrion/datasets/nn/translation"
    DATASETS = [
        # {"name": "europarl", "sizes": [("original", None), ("50k", 50000)], "languages": ["es-en"]},
        {"name": "multi30k", "sizes": [("original", None)], "languages": ["de-en"]},

        # {"name": "ccaligned", "sizes": [("original", None)], "languages": ["ti-en"]},

        # {"name": "commoncrawl", "sizes": [("original", None), ("100k", 100000), ("50k", 50000)], "languages": ["es-en"]},
        # {"name": "europarl", "sizes": [("original", None), ("100k", 100000), ("50k", 50000)], "languages": ["cs-en", "de-en", "es-en", "fr-en"]},
        # {"name": "iwlst16", "sizes": [("original", None)], "languages": ["de-en"]},
        # {"name": "newscommentary", "sizes": [("original", None), ("100k", 100000), ("50k", 50000)], "languages": ["es-en"]},
        # {"name": "scielo/health", "sizes": [("original", None), ("100k", 100000), ("50k", 50000)], "languages": ["es-en", "pt-en"]},
        # {"name": "scielo/biological", "sizes": [("original", None), ("100k", 100000), ("50k", 50000)], "languages": ["es-en", "pt-en"]},
    ]

    # Create datasets
    main(base_path=BASE_PATH, datasets=DATASETS)
    print("Done!")
