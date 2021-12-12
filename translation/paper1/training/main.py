import datetime
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


def preprocess(toolkit, base_path, datasets, subword_model, vocab_size, use_pretokenized, force_overwrite):
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
                    fairseq_entry.fairseq_preprocess(ds_path, src_lang, trg_lang, use_pretokenized, force_overwrite)
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
                    fairseq_entry.fairseq_train(ds_path, run_name, force_overwrite)
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
                    output_path = os.path.join(train_ds_path, "models", toolkit, "runs", run_name, "eval")
                    spm_model_path = os.path.join(train_ds_path, "vocabs", "spm", subword_model, str(vocab_size), f"spm_{src_lang}-{trg_lang}.model")
                    evaluate_model(toolkit=toolkit, base_path=base_path, eval_datasets=eval_datasets, run_name=run_name,
                                   checkpoint_path=checkpoint_path, output_path=output_path, beams=beams,
                                   subword_model=subword_model, spm_model_path=spm_model_path,
                                   force_overwrite=force_overwrite)


def evaluate_model(toolkit, base_path, eval_datasets, run_name, checkpoint_path, output_path, beams, subword_model,
                   spm_model_path, force_overwrite):
    print(f"- Evaluate model: (run_name= {run_name}, checkpoint_path={checkpoint_path}, beams={str(beams)}])")

    for ds in eval_datasets:  # Dataset name (to evaluate)
        ds_name = ds["name"]
        for ds_size_name, ds_max_lines in ds["sizes"]:  # Dataset lengths (to evaluate)
            for lang_pair in ds["languages"]:  # Dataset languages (to evaluate)
                src_lang, trg_lang = lang_pair.split("-")

                # Get eval name
                eval_name = "_".join([ds_name, ds_size_name, lang_pair])

                # Get dataset path (to evaluate)
                ds_path = os.path.join(base_path, ds_name, ds_size_name, lang_pair)

                # Select toolkit
                if toolkit == "fairseq":
                    ds_eval_path = os.path.join(ds_path, "models", "fairseq", "data-bin")
                    for beam in beams:
                        # Create outpath (if needed)
                        beam_output_path = os.path.join(output_path, eval_name, "beams", f"beam_{beam}")
                        path = Path(beam_output_path)
                        path.mkdir(parents=True, exist_ok=True)

                        # Translate
                        fairseq_entry.fairseq_translate(data_path=ds_eval_path, checkpoint_path=checkpoint_path,
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

    RUN_NAME = "mymodel"
    # BASE_PATH = "/home/scarrion/datasets/nn/translation"
    BASE_PATH = "/home/salva/Documents/datasets/nn/translation"
    SUBWORD_MODELS = ["word"]  # unigram, bpe, char, or word
    VOCAB_SIZE = [16000]
    BEAMS = [5]
    FORCE_PRETOKENIZED = False
    FORCE_OVERWRITE = False
    TOOLKIT = "fairseq"
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

    for sw_model in SUBWORD_MODELS:
        for voc_size in VOCAB_SIZE:
            flag_pretok = (sw_model == "word" or FORCE_PRETOKENIZED)

            # # Preprocess datasets
            # preprocess(toolkit=TOOLKIT, base_path=BASE_PATH, datasets=DATASETS, subword_model=sw_model,
            #            vocab_size=voc_size, use_pretokenized=flag_pretok, force_overwrite=FORCE_OVERWRITE)
            #
            # # Train model
            # train(toolkit=TOOLKIT, base_path=BASE_PATH, datasets=DATASETS, run_name=RUN_NAME, subword_model=sw_model,
            #       vocab_size=voc_size, force_overwrite=FORCE_OVERWRITE)

            # Evaluate models
            evaluate(toolkit=TOOLKIT, base_path=BASE_PATH, train_datasets=DATASETS, eval_datasets=DATASETS,
                     run_name=RUN_NAME, subword_model=sw_model, vocab_size=voc_size, beams=BEAMS,
                     force_overwrite=FORCE_OVERWRITE)
    print("Done!")
