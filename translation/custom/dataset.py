import os
import math
import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


def _read_file(filename, strip=False, remove_break_lines=True):
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f.readlines()] if strip else f.readlines()
        lines = [line.replace('\n', '') for line in lines] if remove_break_lines else lines
    return lines


class Vocabulary:
    def __init__(self, filename, lang,
                 unk_id=0, sos_id=1, eos_id=2, pad_id=3,
                 unk_piece="<unk>", sos_piece="<s>", eos_piece="</s>", pad_piece="<pad>"):
        self.lang = lang

        # Set special tokens
        self.unk_id = unk_id
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.unk_piece = unk_piece
        self.sos_piece = sos_piece
        self.eos_piece = eos_piece
        self.pad_piece = pad_piece

        # Parse file. Special tokens must appear first in the file
        tokens = [line.split('\t') for line in _read_file(filename)]
        self.voc2idx = {tok: idx for idx, (tok, log_prob) in enumerate(tokens)}
        self.idx2voc = {idx: tok for idx, (tok, log_prob) in enumerate(tokens)}

        # Check special tokens
        assert self.idx2voc[unk_id] == unk_piece
        assert self.idx2voc[sos_id] == sos_piece
        assert self.idx2voc[eos_id] == eos_piece
        assert self.idx2voc[pad_id] == pad_piece

    def __len__(self):
        return len(self.voc2idx)

    def encode(self, tokens, add_special_tokens=False):
        idxs = [self.voc2idx.get(tok, self.unk_id) for tok in tokens]
        idxs = [self.sos_id] + idxs + [self.eos_id] if add_special_tokens else idxs
        return idxs

    def decode(self, idxs, remove_special_tokens=False):
        # Remove special tokens
        if remove_special_tokens:
            try:
                # Remove <sos>
                sos_pos = idxs.index(self.sos_id)
                idxs = idxs[sos_pos+1:]
            except ValueError:
                pass
            try:
                # Remove <eos>
                eos_pos = idxs.index(self.eos_id)
                idxs = idxs[:eos_pos]
            except ValueError:
                pass

        # Decode sentences
        tokens = [self.idx2voc.get(idx, self.unk_piece) for idx in idxs]
        return tokens


class TranslationDataset(Dataset):
    def __init__(self, data_path, src_lang, trg_lang, split):
        self.src_tok = src_lang
        self.trg_tok = trg_lang

        # Read vocabs
        self.src_vocab = Vocabulary(filename=os.path.join(data_path, f"vocab.{src_lang}"), lang=src_lang)
        self.trg_vocab = Vocabulary(filename=os.path.join(data_path, f"vocab.{trg_lang}"), lang=trg_lang)

        # Read translation files
        self.src_lines = _read_file(filename=os.path.join(data_path, f"{split}.{src_lang}"))
        self.trg_lines = _read_file(filename=os.path.join(data_path, f"{split}.{trg_lang}"))

        # Checks
        assert len(self.src_lines) == len(self.trg_lines)

    def __len__(self):
        return len(self.src_lines)

    def __getitem__(self, idx):
        src_line, trg_line = self.src_lines[idx], self.trg_lines[idx]
        return src_line, trg_line

    def collate_fn(self, batch, max_tokens=None):
        x_encoded, y_encoded = [], []
        x_max_len = y_max_len = 0

        # Add elements to batch
        for i, (x, y) in enumerate(batch):
            _x = self.src_vocab.encode(x.strip().split(' '), add_special_tokens=True)
            _y = self.trg_vocab.encode(y.strip().split(' '), add_special_tokens=True)

            # Control tokens in batch
            x_max_len = max(x_max_len, len(_x))
            y_max_len = max(y_max_len, len(_y))

            # Add elements
            if max_tokens is None or (i+1)*(x_max_len+y_max_len) <= max_tokens:  # sample*size
                x_encoded.append(torch.tensor(_x, dtype=torch.long))
                y_encoded.append(torch.tensor(_y, dtype=torch.long))
            else:
                msg = "[WARNING] Dropping {:.2f}% of the batch because the maximum number of tokens ({}) was exceeded"
                drop_ratio = 1 - ((i+1)/len(batch))
                print(msg.format(drop_ratio, max_tokens))
                break

        # Pad sequence
        x_padded = pad_sequence(x_encoded, batch_first=False, padding_value=self.src_vocab.pad_id).T
        y_padded = pad_sequence(y_encoded, batch_first=False, padding_value=self.trg_vocab.pad_id).T

        # Check stuff
        assert x_padded.shape[0] == y_padded.shape[0] == len(x_encoded)  # Control samples
        assert max_tokens is None or (x_padded.numel() + y_padded.numel()) <= max_tokens  # Control max tokens
        return x_padded, y_padded
