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
from translation.paper1.training import fairseq_entry, opennmt_entry

CONDA_OPENNMT_ENVNAME = "mltests"


def train_model(toolkit, base_path, datasets, run_name, subword_model, vocab_size, use_pretokenized, force_overwrite):
    print(f"- Training model: (run_name={run_name}, subword_model={subword_model}; vocab_size={vocab_size})")

    for ds in datasets:  # Dataset
        ds_name = ds["name"]
        for ds_size_name, ds_max_lines in ds["sizes"]:  # Lengths
            for lang_pair in ds["languages"]:  # Languages
                src_lang, trg_lang = lang_pair.split("-")

                # Get dataset path
                ds_path = os.path.join(base_path, ds_name, ds_size_name, lang_pair)

                # Run model
                if toolkit == "fairseq":
                    fairseq_entry.fairseq_model(ds_path, run_name, src_lang, trg_lang, use_pretokenized, force_overwrite)
                elif toolkit == "opennmt":
                    opennmt_entry.opennmt_model(ds_path, run_name, src_lang, trg_lang, use_pretokenized, force_overwrite)
                else:
                    raise NotImplementedError(f"Unknown toolkit: {toolkit}")


def main(base_path, datasets, use_pretokenized=False, prefix="exp1_transformer", toolkit="fairseq", force_overwrite=False):
    # Train range of models
    for subword_model in ["word"]:  # unigram, bpe, char, or word
        for vocab_size in [16000]:
            flag_pretok = (subword_model == "word" or use_pretokenized)

            # Run name
            dt = datetime.datetime.today()
            run_name = f"{prefix}"
            # run_name = f"{prefix}__{dt.year}_{dt.month}_{dt.day}_{dt.hour}_{dt.minute}_{dt.second}"

            # Train model
            train_model(toolkit, base_path, datasets, run_name,  subword_model=subword_model, vocab_size=vocab_size,
                        use_pretokenized=flag_pretok, force_overwrite=force_overwrite)


if __name__ == "__main__":
    # Download datasets: https://opus.nlpl.eu/

    # BASE_PATH = "/Users/salvacarrion/Documents/Programming/Datasets/nn/translation"
    BASE_PATH = "/home/scarrion/datasets/nn/translation"
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

    # Create datasets
    main(base_path=BASE_PATH, datasets=DATASETS)
    print("Done!")
