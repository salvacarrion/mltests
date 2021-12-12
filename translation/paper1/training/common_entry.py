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

CONDA_ENVNAME = "mltests"


def score_test_files(data_path, src_lang, trg_lang, force_overwrite=True, bleu=True, chrf=True, ter=False, bertscore=True, comet=True):
    # Create path (if needed)
    score_path = os.path.join(data_path, "scores")
    path = Path(score_path)
    path.mkdir(parents=True, exist_ok=True)

    # Test files (cleaned)
    src_file_path = os.path.join(data_path, "src.txt")
    ref_file_path = os.path.join(data_path, "ref.txt")
    hyp_file_path = os.path.join(data_path, "hyp.txt")

    # Sacrebleu
    metrics = ""
    metrics += "bleu " if bleu else ""
    metrics += "chrf " if chrf else ""
    metrics += "ter " if ter else ""
    if metrics:
        sacrebleu_scores_path = os.path.join(score_path, "sacrebleu_scores.json")
        cmd = f"sacrebleu {ref_file_path} -i {hyp_file_path} -m {metrics} -w 5 > {sacrebleu_scores_path}"  # bleu chrf ter
        subprocess.call(['/bin/bash', '-i', '-c', f"conda activate {CONDA_ENVNAME} && {cmd}"])

    # BertScore
    if bertscore:
        bertscore_scores_path = os.path.join(score_path, "bertscores.txt")
        cmd = f"bert-score -r {ref_file_path} -c {hyp_file_path} --lang {trg_lang} > {bertscore_scores_path}"
        subprocess.call(['/bin/bash', '-i', '-c', f"conda activate {CONDA_ENVNAME} && {cmd}"])

    # Comet
    if comet:
        comet_scores_path = os.path.join(score_path, "comet_scores.txt")
        cmd = f"comet-score -s {src_file_path} -t {hyp_file_path} -r {ref_file_path} > {comet_scores_path}"
        subprocess.call(['/bin/bash', '-i', '-c', f"conda activate {CONDA_ENVNAME} && {cmd}"])
