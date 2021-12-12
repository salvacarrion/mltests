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


def train_model(base_path, datasets, model_type="unigram", vocab_size=8000):
    print("Building vocabs...")

    for ds in datasets:  # Dataset
        ds_name = ds["name"]
        for ds_size_name, ds_max_lines in ds["sizes"]:  # Lengths
            for lang_pair in ds["languages"]:  # Languages
                src_lang, trg_lang = lang_pair.split("-")

                # Learn model
                command = f""
                subprocess.call(['/bin/bash', '-i', '-c', command])  # https://stackoverflow.com/questions/12060863/python-subprocess-call-a-bash-alias/25099813


def main(base_path, datasets):
    # Train range of models
    for model_type in ["char", "unigram"]:  # unigram, bpe, char, or word
        for vocab_size in [16000]:
            print(f"- Training model: (model_type={model_type}; vocab_size={vocab_size})")
            train_model(base_path, datasets, model_type=model_type, vocab_size=vocab_size)


if __name__ == "__main__":
    # Download datasets: https://opus.nlpl.eu/

    # BASE_PATH = "/Users/salvacarrion/Documents/Programming/Datasets/nn/translation"
    BASE_PATH = "/home/scarrion/datasets/nn/translation"
    DATASETS = [
        {"name": "ccaligned", "sizes": [("original", None)], "languages": ["ti-en"]},

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
