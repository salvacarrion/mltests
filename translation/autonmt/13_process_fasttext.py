import pandas as pd

from autonmt.preprocessing import DatasetBuilder
from autonmt.bundle.report import generate_report

from autonmt.toolkits import AutonmtTranslator
from autonmt.modules.models import Transformer
from autonmt.vocabularies import Vocabulary

import os
import datetime
import torch
import numpy as np
from autonmt.bundle import utils
from sklearn.decomposition import PCA
import copy
from sklearn.preprocessing import StandardScaler

import json
import torch
import random
import numpy as np
from pytorch_lightning.utilities.seed import seed_everything
import io
import json

import fasttext
import fasttext.util

# Create preprocessing for training
builder_big = DatasetBuilder(
    base_path="/home/scarrion/datasets/nn/translation",
    datasets=[
        # {"name": "multi30k", "languages": ["de-en"], "sizes": [("original", None)]},
        {"name": "europarl", "languages": ["de-en"], "sizes": [("original_lc", None)]},
        # {"name": "europarl", "languages": ["de-en"], "sizes": [("100k", 100000)]},
    ],
    subword_models=["word"],
    vocab_sizes=[16000],
    merge_vocabs=False,
    force_overwrite=False,
    use_cmd=False,
    eval_mode="same",
    letter_case="lower",
).build(make_plots=False, safe=True)
big_datasets = builder_big.get_ds()
ds_ref = big_datasets[0]

base_path = "."

# Load vocabs
for lang, lang_id in [(ds_ref.src_lang, "src")]:  # ,
    # Load vocab
    vocab = Vocabulary().build_from_ds(ds_ref, lang=lang)
    tokens = [tok.replace('▁', '') for tok in vocab.get_tokens()]

    # Save tokens
    with open(f"{base_path}/{lang_id}.json", 'w') as f:
        json.dump(tokens, f)

    # Load model, reduce it and get embeddings
    ft = fasttext.load_model(f"/home/scarrion/Downloads/cc.{lang}.300.bin")
    fasttext.util.reduce_model(ft, 256)
    arr = [ft.get_word_vector(tok) for tok in tokens]
    arr = np.stack(arr, axis=0)

    # Save tensor
    np.save(f"{base_path}/{lang_id}.npy", arr)
    print(f"Saved {lang_id}!")
    asd = 3


