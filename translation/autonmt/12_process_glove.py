import numpy as np
import pandas as pd
from autonmt.preprocessing import DatasetBuilder
from translation.autonmt.models.autoencoder import LitAutoEncoder

import os
import numpy as np
from autonmt.bundle import utils

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import math

emb_size = 300
embeddings_index = {}
path_to_glove_file = f"/home/scarrion/Downloads/glove.6B/glove.6B.{emb_size}d.txt"
with open(path_to_glove_file) as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, "f", sep=" ")
        embeddings_index[word] = coefs


# Create preprocessing for training
builder = DatasetBuilder(
    base_path="/home/scarrion/datasets/nn/translation",
    datasets=[
        {"name": "multi30k", "languages": ["de-en"], "sizes": [("original", None)]},
        {"name": "europarl", "languages": ["de-en"], "sizes": [("100k", 100000)]},
    ],
    subword_models=["word"],
    vocab_sizes=[250, 500, 1000, 2000, 4000, 8000],
    merge_vocabs=False,
    force_overwrite=False,
    use_cmd=True,
    eval_mode="same",
    conda_env_name="mltests",
    letter_case="lower",
).build(make_plots=False, safe=True)

# Create preprocessing for training and testing
tr_datasets = builder.get_ds()
ts_datasets = builder.get_ds(ignore_variants=True)

# Train & Score a model for each dataset
scores = []
errors = []
max_tokens = 100
device = "cuda" if torch.cuda.is_available() else "cpu"
compressor = "ae"

# Export raw embeddings
for ds in tr_datasets:
    # Read vocab
    trg_vocab = open(ds.get_vocab_file(lang=ds.trg_lang) + ".vocab", 'r').readlines()
    trg_vocab = [line.replace('▁', '').split('\t')[0].strip() for line in trg_vocab]

    # Get embeddings
    glove_emb = []
    missing = 0
    for word in trg_vocab:
        if word in embeddings_index:
            tok = embeddings_index[word]
        else:
            missing += 1
            tok = torch.nn.init.normal_(torch.empty(emb_size)).detach().cpu().numpy()
        glove_emb.append(tok)
    assert len(glove_emb) == len(trg_vocab)
    print(f"Missing: {missing}")

    # Save embeddings
    emb_dir = f".outputs/tmp/{emb_size}/{str(ds)}"
    utils.make_dir(emb_dir)
    glove_emb = np.stack(glove_emb, axis=0)
    np.save(f"{emb_dir}/raw_glove.npy", glove_emb)
    asd = 3
