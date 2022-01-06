# import pandas as pd
# from autonmt.preprocessing import DatasetBuilder
# from translation.autonmt.models.autoencoder import LitAutoEncoder
#
# import os
# import numpy as np
#
# import torch
# from torch import nn
# import torch.nn.functional as F
# from torch.utils.data import DataLoader, TensorDataset
# import pytorch_lightning as pl
# from pytorch_lightning.callbacks.early_stopping import EarlyStopping
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import r2_score
# from sklearn.metrics import mean_squared_error
# import math
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
# def encode_data(filename, enc_dim, name=""):
#     # Save embeddings
#     src_emb = np.load(os.path.join(filename, "src.npy"))
#     trg_emb = np.load(os.path.join(filename, "trg.npy"))
#
#     # Check if there is need for encoding
#     if src_emb.shape[1] == trg_emb.shape[1] == enc_dim:
#         new_src_emb, new_trg_emb = src_emb, trg_emb
#         src_emb_rec, trg_emb_rec = src_emb, trg_emb
#     else:
#         # Standarized
#         src_scaler = StandardScaler().fit(src_emb)
#         trg_scaler = StandardScaler().fit(trg_emb)
#         src_emb_scaled = src_scaler.transform(src_emb)
#         trg_emb_scaled = trg_scaler.transform(trg_emb)
#
#         # Train PCA and encode data
#         src_pca = PCA(n_components=enc_dim).fit(X=src_emb_scaled)
#         trg_pca = PCA(n_components=enc_dim).fit(X=trg_emb_scaled)
#
#         # Apply PCA
#         new_src_emb = src_pca.transform(src_emb)
#         new_trg_emb = trg_pca.transform(trg_emb)
#
#         # Inverse PCA + inverse scale
#         src_emb_scaled_rec = src_pca.inverse_transform(new_src_emb)
#         trg_emb_scaled_rec = trg_pca.inverse_transform(new_trg_emb)
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
#         vocab_sizes=[500, 1000, 2000, 4000, 8000],
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
#     name = "pca"
#     for ds in tr_datasets:
#         print(f"Encoding data for: {str(ds)}")
#
#         # Encode data
#         src_scores, trg_scores = encode_data(filename=f".outputs/tmp/{origin_emb_size}/{str(ds)}", enc_dim=256, name=name)
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
#
#     # Save data
# if __name__ == "__main__":
#     main()
