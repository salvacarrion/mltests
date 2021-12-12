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

CONDA_ENVNAME = "mltests"


def opennmt_model(data_path, run_name, eval_name, src_lang, trg_lang, use_pretokenized, force_overwrite):
    # Create path (if needed)
    preprocess_path = os.path.join(data_path, "models", "opennmt")
    path = Path(preprocess_path)
    path.mkdir(parents=True, exist_ok=True)

    # Preprocess files
    opennmt_preprocess(data_path, src_lang, trg_lang, use_pretokenized, force_overwrite)

    # Train model

    # Evaluate


def opennmt_preprocess(data_path, src_lang, trg_lang, use_pretokenized, force_overwrite):
    # Create path (if needed)
    preprocess_path = os.path.join(data_path, "models", "opennmt")
    path = Path(preprocess_path)
    path.mkdir(parents=True, exist_ok=True)

    # Check if data-bin exists
    data_path = os.path.join(preprocess_path, "data-bin")
    if os.path.exists(data_path):
        print("\t=> Skipping preprocessing as it already exists")
    else:
        # Preprocess
        raw_folder = "splits" if not use_pretokenized else "pretokenized"
        train_path = os.path.join(data_path, raw_folder, "train")
        val_path = os.path.join(data_path, raw_folder, "val")
        test_path = os.path.join(data_path, raw_folder, "test")

        # Run command
        preprocess_cmd = f""
        subprocess.call(['/bin/bash', '-i', '-c',
                         f"conda activate {CONDA_ENVNAME} && {preprocess_cmd}"])  # https://stackoverflow.com/questions/12060863/python-subprocess-call-a-bash-alias/25099813

