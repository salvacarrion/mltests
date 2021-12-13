import datetime
import glob
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
from translation.paper1.training import fairseq_entry, opennmt_entry, common_entry

CONDA_OPENNMT_ENVNAME = "mltests"


def preprocess(toolkit, base_path, datasets, subword_model, vocab_size, force_overwrite):
    print(f"- Process: (subword_model={subword_model}; vocab_size={vocab_size})")

    for ds in datasets:  # Training dataset
        ds_name = ds["name"]
        for ds_size_name, ds_max_lines in ds["sizes"]:  # Training lengths
            for lang_pair in ds["languages"]:  # Training languages
                src_lang, trg_lang = lang_pair.split("-")

                # Get dataset path
                ds_path = os.path.join(base_path, ds_name, ds_size_name, lang_pair)

                # Select toolkit
                if toolkit == "fairseq":
                    fairseq_entry.fairseq_preprocess(ds_path, src_lang, trg_lang, subword_model, vocab_size, force_overwrite)
                else:
                    raise NotImplementedError(f"Unknown toolkit: {toolkit}")


def train(toolkit, base_path, datasets, run_name, subword_model, vocab_size, force_overwrite):
    # Run name
    print(f"- Train & Score: (run_name={run_name}, subword_model={subword_model}; vocab_size={vocab_size})")

    for ds in datasets:  # Training dataset
        ds_name = ds["name"]
        for ds_size_name, ds_max_lines in ds["sizes"]:  # Training lengths
            for lang_pair in ds["languages"]:  # Training languages
                src_lang, trg_lang = lang_pair.split("-")

                # Get dataset path
                ds_path = os.path.join(base_path, ds_name, ds_size_name, lang_pair)

                # Select toolkit
                if toolkit == "fairseq":
                    fairseq_entry.fairseq_train(ds_path, run_name, subword_model, vocab_size, force_overwrite)
                else:
                    raise NotImplementedError(f"Unknown toolkit: {toolkit}")


def evaluate(toolkit, base_path, train_datasets, eval_datasets, run_name, subword_model, vocab_size, beams, force_overwrite):
    # Run name
    print(f"- Evaluate models: (run_name={run_name}, subword_model={subword_model}; vocab_size={vocab_size})")

    for ds in train_datasets:  # Dataset name (of the trained model)
        ds_name = ds["name"]
        for ds_size_name, ds_max_lines in ds["sizes"]:  # Dataset length (of the trained model)
            for lang_pair in ds["languages"]:  # Dataset language (of the trained model)
                src_lang, trg_lang = lang_pair.split("-")

                # Get base path of the trained model
                train_ds_path = os.path.join(base_path, ds_name, ds_size_name, lang_pair)

                # Evaluate model
                for chkpt_fname in ["checkpoint_best.pt"]:
                    checkpoint_path = os.path.join(train_ds_path, "models", toolkit, "runs", run_name, "checkpoints", chkpt_fname)
                    spm_model_path = os.path.join(train_ds_path, "vocabs", "spm", subword_model, str(vocab_size),  f"spm_{src_lang}-{trg_lang}.model")
                    evaluate_model(toolkit=toolkit, base_path=base_path, train_ds_path=train_ds_path,
                                   eval_datasets=eval_datasets, run_name=run_name,
                                   checkpoint_path=checkpoint_path, spm_model_path=spm_model_path, beams=beams,
                                   subword_model=subword_model, vocab_size=vocab_size, force_overwrite=force_overwrite)


def evaluate_model(toolkit, base_path, train_ds_path, eval_datasets, run_name, checkpoint_path, spm_model_path, beams,
                   subword_model, vocab_size, force_overwrite):
    print(f"- Evaluate model: (run_name= {run_name}, checkpoint_path={checkpoint_path}, beams={str(beams)}])")

    # Set paths
    output_path = os.path.join(train_ds_path, "models", toolkit, "runs", run_name, "eval")

    for ds in eval_datasets:  # Dataset name (to evaluate)
        ds_name = ds["name"]
        for ds_size_name, ds_max_lines in ds["sizes"]:  # Dataset lengths (to evaluate)
            for lang_pair in ds["languages"]:  # Dataset languages (to evaluate)
                src_lang, trg_lang = lang_pair.split("-")

                # Get eval name
                eval_name = "_".join([ds_name, ds_size_name, lang_pair])

                # Get dataset path (to evaluate)
                ds_path = os.path.join(base_path, ds_name, ds_size_name, lang_pair)

                # Create path for encoded files
                eval_data_path = os.path.join(output_path, eval_name, "data")
                path = Path(eval_data_path)
                path.mkdir(parents=True, exist_ok=True)

                # Encode dataset using the SPM of this model
                for fname in [f"test.{src_lang}", f"test.{trg_lang}"]:
                    raw_folder = "pretokenized" if (subword_model == "word") else "splits"
                    ori_filename = os.path.join(ds_path, "data", raw_folder, fname)
                    new_filename = os.path.join(eval_data_path, fname)

                    # Check if the file exists
                    if not force_overwrite and os.path.exists(new_filename):
                        print("\t\t- Skipping eval file encoding as it already exists")
                    else:
                        command = f"spm_encode --model={spm_model_path} --output_format=piece < {ori_filename} > {new_filename}"  # --vocabulary={spm_model_path}.vocab --vocabulary_threshold={min_vocab_frequency}
                        subprocess.call(['/bin/bash', '-i', '-c', command])

                # Select toolkit
                if toolkit == "fairseq":
                    # Get train vocabulary
                    src_vocab_path = os.path.join(train_ds_path, "models", "fairseq", "data-bin", subword_model, str(vocab_size), f"dict.{src_lang}.txt")
                    trg_vocab_path = os.path.join(train_ds_path, "models", "fairseq", "data-bin", subword_model, str(vocab_size), f"dict.{trg_lang}.txt")

                    # Preprocess data
                    eval_data_bin_path = os.path.join(output_path, eval_name, "data-bin")
                    preprocess_cmd = f"fairseq-preprocess --source-lang {src_lang} --target-lang {trg_lang} --trainpref {eval_data_path}/test --testpref {eval_data_path}/test --destdir {eval_data_bin_path} --srcdict {src_vocab_path} --tgtdict {trg_vocab_path} --workers	$(nproc)"
                    subprocess.call(['/bin/bash', '-i', '-c', f"conda activate fairseq && {preprocess_cmd}"])  # https://stackoverflow.com/questions/12060863/python-subprocess-call-a-bash-alias/25099813

                    for beam in beams:
                        # Create outpath (if needed)
                        beam_output_path = os.path.join(output_path, eval_name, "beams", f"beam_{beam}")
                        path = Path(beam_output_path)
                        path.mkdir(parents=True, exist_ok=True)

                        # Translate
                        fairseq_entry.fairseq_translate(data_path=eval_data_bin_path, checkpoint_path=checkpoint_path,
                                                        output_path=beam_output_path, src_lang=src_lang, trg_lang=trg_lang,
                                                        subword_model=subword_model, spm_model_path=spm_model_path,
                                                        force_overwrite=force_overwrite, beam_width=beam)

                        # Score
                        common_entry.score_test_files(data_path=beam_output_path, src_lang=src_lang, trg_lang=trg_lang,
                                                      force_overwrite=force_overwrite)
                else:
                    raise NotImplementedError(f"Unknown toolkit: {toolkit}")


if __name__ == "__main__":
    # Download datasets: https://opus.nlpl.eu/

    # BASE_PATH = "/home/scarrion/datasets/nn/translation"
    BASE_PATH = "/home/salva/Documents/datasets/nn/translation"

    ENCODING_MODE = "pretokenized"  # splits (raw), pretokenized (moses), encoded (spm)
    SUBWORD_MODELS = ["word"]  # unigram, bpe, char, or word
    VOCAB_SIZE = [16000]
    BEAMS = [5]
    FORCE_OVERWRITE = True
    TOOLKIT = "fairseq"
    RUN_NAME = "mymodel"

    DATASETS = [
        {"name": "multi30k", "sizes": [("original", None)], "languages": ["de-en"]},
        # {"name": "ccaligned", "sizes": [("original", None)], "languages": ["ti-en"]},

        # {"name": "commoncrawl", "sizes": [("original", None), ("100k", 100000), ("50k", 50000)], "languages": ["es-en"]},
        # {"name": "europarl", "sizes": [("original", None), ("100k", 100000), ("50k", 50000)], "languages": ["cs-en", "de-en", "es-en", "fr-en"]},
        # {"name": "iwlst16", "sizes": [("original", None)], "languages": ["de-en"]},
        # {"name": "multi30k", "sizes": [("original", None)], "languages": ["de-en"]},
        # {"name": "newscommentary", "sizes": [("original", None), ("100k", 100000), ("50k", 50000)], "languages": ["es-en"]},
        # {"name": "scielo/health", "sizes": [("original", None), ("100k", 100000), ("50k", 50000)], "languages": ["es-en", "pt-en"]},
        # {"name": "scielo/biological", "sizes": [("original", None), ("100k", 100000), ("50k", 50000)], "languages": ["es-en", "pt-en"]},
    ]

    EVAL_DATASETS = [
        {"name": "iwlst16", "sizes": [("original", None)], "languages": ["de-en"]},
    ]

    for sw_model in SUBWORD_MODELS:
        for voc_size in VOCAB_SIZE:
            run_name = f"{RUN_NAME}_{sw_model}_{voc_size}"

            # # Preprocess datasets
            # preprocess(toolkit=TOOLKIT, base_path=BASE_PATH, datasets=DATASETS, subword_model=sw_model,
            #            vocab_size=voc_size, force_overwrite=FORCE_OVERWRITE)
            #
            # # Train model
            # train(toolkit=TOOLKIT, base_path=BASE_PATH, datasets=DATASETS, run_name=run_name, subword_model=sw_model,
            #       vocab_size=voc_size, force_overwrite=FORCE_OVERWRITE)

            # Evaluate models
            evaluate(toolkit=TOOLKIT, base_path=BASE_PATH, train_datasets=DATASETS, eval_datasets=EVAL_DATASETS,
                     run_name=run_name, subword_model=sw_model, vocab_size=voc_size, beams=BEAMS,
                     force_overwrite=FORCE_OVERWRITE)
    print("Done!")
