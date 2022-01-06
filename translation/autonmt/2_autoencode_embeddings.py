# from autonmt.preprocessing import DatasetBuilder
# from translation.autonmt.models.autoencoder import LitAutoEncoder
# import pandas as pd
# import os
# import numpy as np
#
# import torch
# from torch import nn
# import torch.nn.functional as F
# from torch.utils.data import DataLoader, TensorDataset
# import pytorch_lightning as pl
# from pytorch_lightning.callbacks.early_stopping import EarlyStopping
#
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import r2_score
# from sklearn.metrics import mean_squared_error
# import math
#
#
# def ae_fit_transform(x, enc_dim, batch_size=256, max_epochs=3000, patience=250, num_workers=0):
#     # Create train datasets and data loaders
#     train_data = TensorDataset(torch.tensor(x), torch.tensor(x))
#     train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, num_workers=num_workers)
#
#     # Instantiate model
#     model = LitAutoEncoder(input_dim=x.shape[1], enc_dim=enc_dim)
#
#     # Callbacks
#     early_stop_callback = EarlyStopping(monitor="train_loss", patience=patience, mode="min")
#
#     # Train model
#     trainer = pl.Trainer(max_epochs=max_epochs, callbacks=[early_stop_callback], devices="auto", accelerator="auto",
#                          logger=True, enable_checkpointing=False)
#     trainer.fit(model, train_loader)
#
#     # Transform data
#     model.mode = "encode"
#     test_enc_data = TensorDataset(torch.tensor(x))
#     test_enc_loader = torch.utils.data.DataLoader(test_enc_data, shuffle=False, batch_size=batch_size, num_workers=num_workers)
#     x_enc = trainer.predict(model, dataloaders=test_enc_loader)
#     x_enc = torch.concat(x_enc, dim=0).detach().cpu().numpy()
#
#     # Inverse transform
#     model.mode = "decode"
#     test_dec_data = TensorDataset(torch.tensor(x_enc))
#     test_dec_loader = torch.utils.data.DataLoader(test_dec_data, shuffle=False, batch_size=batch_size, num_workers=num_workers)
#     x_dec = trainer.predict(model, dataloaders=test_dec_loader)
#     x_dec = torch.concat(x_dec, dim=0).detach().cpu().numpy()
#
#     return x_enc, x_dec
#
#
# def compute_error(x, x_hat, title=""):
#     r2 = r2_score(x, x_hat)
#     rmse = math.sqrt(mean_squared_error(x, x_hat))
#     nrmse = rmse / math.sqrt(np.mean(x**2))
#
#     print(f"{title}:")
#     print(f"\t- R²: {r2}")
#     print(f"\t- RMSE: {rmse}")
#     print(f"\t- NRMSE: {nrmse}")
#
#     d = {"r2": r2, "rmse": rmse, "nrmse": nrmse}
#     return d
#
#
# def encode_data(filename, enc_dim, name=""):
#     # Save embeddings
#     src_emb = np.load(os.path.join(filename, "src.npy"))
#     trg_emb = np.load(os.path.join(filename, "trg.npy"))
#
#     # Check if there is need for encoding
#     if src_emb.shape[1] == trg_emb.shape[1] == enc_dim:
#         new_src_emb, new_trg_emb = src_emb, trg_emb
#         src_emb_rec, trg_emb_rec = src_emb, trg_emb
#
#     else:
#         # Standarized
#         src_scaler = StandardScaler().fit(src_emb)
#         trg_scaler = StandardScaler().fit(trg_emb)
#         src_emb_scaled = src_scaler.transform(src_emb)
#         trg_emb_scaled = trg_scaler.transform(trg_emb)
#
#         # Train autoencoder and encode data
#         new_src_emb, src_emb_scaled_rec = ae_fit_transform(src_emb_scaled, enc_dim=enc_dim)
#         new_trg_emb, trg_emb_scaled_rec = ae_fit_transform(trg_emb_scaled, enc_dim=enc_dim)
#
#         # Inverse scale
#         src_emb_rec = src_scaler.inverse_transform(src_emb_scaled_rec)
#         trg_emb_rec = trg_scaler.inverse_transform(trg_emb_scaled_rec)
#
#     # Reconstruction error
#     src_scores = compute_error(src_emb, src_emb_rec, title="src")
#     trg_scores = compute_error(trg_emb, trg_emb_rec, title="trg")
#
#     # Save embeddings
#     np.save(os.path.join(filename, f"src_enc_{name}.npy"), new_src_emb)
#     np.save(os.path.join(filename, f"trg_enc_{name}.npy"), new_trg_emb)
#     print(f"Embeddings saved! ({filename})")
#
#     return src_scores, trg_scores
#
#
# def main():
#     # Create preprocessing for training
#     builder = DatasetBuilder(
#         base_path="/home/scarrion/datasets/nn/translation",
#         datasets=[
#             {"name": "multi30k", "languages": ["de-en"], "sizes": [("original", None)]},
#             {"name": "europarl", "languages": ["de-en"], "sizes": [("100k", 100000)]},
#         ],
#         subword_models=["word"],
#         vocab_sizes=[250, 500, 1000, 2000, 4000, 8000],
#         merge_vocabs=False,
#         force_overwrite=False,
#         use_cmd=True,
#         eval_mode="same",
#         conda_env_name="mltests",
#         letter_case="lower",
#     ).build(make_plots=False, safe=True)
#
#     # Create preprocessing for training and testing
#     tr_datasets = builder.get_ds()
#     ts_datasets = builder.get_ds(ignore_variants=True)
#
#     # Save embeddings
#     rows = []
#     origin_emb_size = 512
#     name = "ae_non_linear_tanh"
#     for ds in tr_datasets:
#         print(f"Encoding data for: {str(ds)}")
#
#         # Encode data
#         src_scores, trg_scores = encode_data(f".outputs/tmp/{origin_emb_size}/{str(ds)}", enc_dim=256, name=name)
#
#         # Keep info
#         src_scores["emb_name"] = "src"
#         src_scores["dataset_name"] = ds.dataset_name
#         src_scores["subword_model"] = ds.subword_model
#         src_scores["vocab_size"] = ds.vocab_size
#         trg_scores["emb_name"] = "trg"
#         trg_scores["dataset_name"] = ds.dataset_name
#         trg_scores["subword_model"] = ds.subword_model
#         trg_scores["vocab_size"] = ds.vocab_size
#         rows.append(src_scores)
#         rows.append(trg_scores)
#
#     # Print results
#     df = pd.DataFrame(rows)
#     df.to_csv(f".outputs/tmp/{origin_emb_size}/{name}_{origin_emb_size}.csv", index=False)
#     print(df)
#
# if __name__ == "__main__":
#     main()
