import re
import json
import random
from collections import defaultdict

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


def save_json(d, savepath):
    with open(savepath, 'w') as f:
        json.dump(d, f)
