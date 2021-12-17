import json
import random
import re
from collections import defaultdict
import sys
import logging
import os
import time
from pathlib import Path

random.seed(123)

import unicodedata
from tqdm import tqdm


def get_translation_files(src_lang, trg_lang):
    files = []
    for split in ["train", "val", "test"]:
        for lang in [src_lang, trg_lang]:
            files.append(f"{split}.{lang}")
    return files


def preprocess_text(text):
    try:
        p_whitespace = re.compile(" +")

        # Remove repeated whitespaces "   " => " "
        text = p_whitespace.sub(' ', text)

        # Normalization Form Compatibility Composition
        text = unicodedata.normalize("NFKC", text)

        # Strip whitespace
        text = text.strip()
    except TypeError as e:
        # print(f"=> Error preprocessing: '{text}'")
        text = ""
    return text


def preprocess_pairs(src_lines, trg_lines, shuffle):
    assert len(src_lines) == len(trg_lines)

    lines = []
    for _src_line, _trg_line in tqdm(zip(src_lines, trg_lines), total=len(src_lines)):
        src_line = preprocess_text(_src_line)
        trg_line = preprocess_text(_trg_line)

        # Remove empty line
        remove_pair = False
        if len(src_line) == 0 or len(trg_line) == 0:
            remove_pair = True
        # elif math.fabs(len(src)-len(trg)) > 20:
        #     remove_pair = True

        # Add lines
        if not remove_pair:
            lines.append((src_line, trg_line))

    # Shuffle
    if shuffle:
        random.shuffle(lines)

    return lines


def get_frequencies(filename):
    vocab_frequencies = defaultdict(int)
    with open(filename, 'r') as f:
        for line in tqdm(f):
            tokens = line.strip().split(' ')
            for tok in tokens:
                vocab_frequencies[tok] += 1
    return vocab_frequencies


def get_tokens_by_sentence(filename):
    with open(filename, 'r') as f:
        token_sizes = [len(line.strip().split(' ')) for line in f.readlines()]
    return token_sizes


def human_format(num, decimals=2):
    if num < 10000:
        return str(num)
    else:
        magnitude = 0
        template = f'%.{decimals}f%s'

        while abs(num) >= 1000:
            magnitude += 1
            num /= 1000.0

        return template % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])


def load_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)


def save_json(d, savepath):
    with open(savepath, 'w') as f:
        json.dump(d, f)


def create_logger(logs_path):
    # Create logget path
    Path(logs_path).mkdir(parents=True, exist_ok=True)

    # Create logger
    mylogger = logging.getLogger()
    mylogger.setLevel(logging.DEBUG)

    # Define format
    logformat = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Define handlers
    # Log file
    log_handler = logging.FileHandler(os.path.join(logs_path, 'logger.log'), mode='w')
    log_handler.setFormatter(logformat)
    mylogger.addHandler(log_handler)

    # Standard output
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(logformat)
    mylogger.addHandler(stdout_handler)

    # Print something
    mylogger.info('Starting logger...')
    return mylogger


def logged_task(logger, row, fn_name, fn, **kwargs):
    start_fn = time.time()
    logger.info(f"***** {fn_name.title()} started *****")
    logger.info(f"----- [{fn_name.title()}] Start time (ss.ms): {start_fn} -----")

    # Call function (...and propagate errors)
    result = None
    # try:
    fn_status = "okay"
    result = fn(**kwargs)
    # except Exception as e:
    #     logger.error(str(e))
    #     fn_status = str(e)

    # Get elapsed time
    end_fn = time.time()
    elapsed_fn = end_fn - start_fn
    elapsed_fn_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_fn))

    # Log time
    logger.info(f"----- [{fn_name.title()}] End time (ss.ms): {end_fn} -----")
    # logger.info(f"----- [{fn_name.title()}] Time elapsed (ss.ms): {elapsed_fn} -----")
    logger.info(f"----- [{fn_name.title()}] Time elapsed (hh:mm:ss.ms): {elapsed_fn_str} -----")
    logger.info(f"***** {fn_name.title()} ended *****")

    # Store results
    row[f"start_{fn_name}"] = start_fn
    row[f"end_{fn_name}"] = end_fn
    row[f"elapsed_{fn_name}"] = elapsed_fn
    row[f"elapsed_{fn_name}_str"] = elapsed_fn_str
    row[f"{fn_name}_status"] = fn_status

    return result


def parse_sacrebleu(text):
    result = {}
    metrics = json.loads("".join(text))
    for m_dict in metrics:
        m_name = f"sacrebleu_{m_dict['name']}".lower().strip()
        result[m_name] = float(m_dict["score"])
    return result


def parse_bertscore(text):
    pattern = r"P: ([01]\.\d*) R: ([01]\.\d*) F1: ([01]\.\d*)"
    line = text[-1].strip()
    groups = re.search(pattern, line).groups()
    result = {"precision": float(groups[0]), "recall": float(groups[1]), "f1": float(groups[2])}
    return result


def parse_comet(text):
    pattern = r"score: (-?[01]\.\d*)"
    line = text[-1].strip()
    groups = re.search(pattern, line).groups()
    result = {"score": float(groups[0])}
    return result


def parse_beer(text):
    pattern = r"total BEER ([01]\.\d*)"
    line = text[-1].strip()
    groups = re.search(pattern, line).groups()
    result = {"score": float(groups[0])}
    return result


def count_datasets(datasets):
    counter = 0
    for ds in datasets:  # Training dataset
        for ds_size_name, ds_max_lines in ds["sizes"]:  # Training lengths
            for lang_pair in ds["languages"]:
                counter += 1
    return counter
