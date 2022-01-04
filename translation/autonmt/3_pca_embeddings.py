from autonmt.preprocessing import DatasetBuilder
from translation.autonmt.models.autoencoder import LitAutoEncoder

import os
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.decomposition import PCA


def encode_data(filename, enc_dim):
    # Save embeddings
    src_emb = np.load(os.path.join(filename, "src.npy"))
    trg_emb = np.load(os.path.join(filename, "trg.npy"))

    # Check if there is need for encoding
    if src_emb.shape[1] == trg_emb.shape[1] == enc_dim:
        new_src_emb, new_trg_emb = src_emb, trg_emb
    else:
        # Train PCA and encode data
        src_pca = PCA(n_components=enc_dim).fit(X=src_emb)
        trg_pca = PCA(n_components=enc_dim).fit(X=trg_emb)

        # Apply PCA
        new_src_emb = src_pca.transform(src_emb)
        new_trg_emb = trg_pca.transform(trg_emb)

    # Save embeddings
    np.save(os.path.join(filename, "src_enc_pca.npy"), new_src_emb)
    np.save(os.path.join(filename, "trg_enc_pca.npy"), new_trg_emb)
    print(f"Embeddings saved! ({filename})")


def main():
    # Create preprocessing for training
    builder = DatasetBuilder(
        base_path="/home/scarrion/datasets/nn/translation",
        datasets=[
            {"name": "multi30k", "languages": ["de-en"], "sizes": [("original", None)]},
            {"name": "europarl", "languages": ["de-en"], "sizes": [("100k", 100000)]},
        ],
        subword_models=["word"],
        vocab_sizes=[1000, 2000, 4000, 8000],
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

    # Save embeddings
    for ds in tr_datasets:
        print(f"Encoding data for: {str(ds)}")

        # Encode data
        encode_data(f".outputs/tmp/512/{str(ds)}", enc_dim=256)


if __name__ == "__main__":
    main()
