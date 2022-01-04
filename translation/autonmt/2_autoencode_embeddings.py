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


def ae_fit_transform(x, enc_dim, batch_size=128, max_epochs=500, patience=50):
    # Create datasets
    train_data = TensorDataset(torch.tensor(x), torch.tensor(x))
    test_data = TensorDataset(torch.tensor(x))

    # Define dataloaders
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_data, shuffle=False, batch_size=batch_size)

    # Instantiate model
    model = LitAutoEncoder(input_dim=x.shape[1], enc_dim=enc_dim)

    # Callbacks
    early_stop_callback = EarlyStopping(monitor="train_loss", patience=patience, mode="min")

    # Train model
    trainer = pl.Trainer(max_epochs=max_epochs, callbacks=[early_stop_callback], devices="auto", accelerator="auto",
                         logger=True, enable_checkpointing=False)
    trainer.fit(model, train_loader)

    # Transform data
    predictions = trainer.predict(model, dataloaders=test_loader)
    predictions = torch.concat(predictions, dim=0).detach().cpu().numpy()
    return predictions


def encode_data(filename, enc_dim):
    # Save embeddings
    src_emb = np.load(os.path.join(filename, "src.npy"))
    trg_emb = np.load(os.path.join(filename, "trg.npy"))

    # Check if there is need for encoding
    if src_emb.shape[1] == trg_emb.shape[1] == enc_dim:
        new_src_emb, new_trg_emb = src_emb, trg_emb
    else:
        # Train autoencoder and encode data
        new_src_emb = ae_fit_transform(src_emb, enc_dim=enc_dim)
        new_trg_emb = ae_fit_transform(trg_emb, enc_dim=enc_dim)

    # Save embeddings
    np.save(os.path.join(filename, "src_enc_ae.npy"), new_src_emb)
    np.save(os.path.join(filename, "trg_enc_ae.npy"), new_trg_emb)
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
