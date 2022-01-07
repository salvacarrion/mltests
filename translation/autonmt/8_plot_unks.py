# import os
#
# import tqdm
# from autonmt.preprocessing import DatasetBuilder
# from autonmt.bundle import utils
# import numpy as np
#
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from autonmt.bundle.utils import make_dir
#
#
# def plot_values(data, title, y_col, y_title, savepath, dpi=300):
#     make_dir(savepath)
#
#     # plt.figure(figsize=(8, 8), dpi=150)
#     # sns.set(font_scale=1.5)
#     plt.figure()
#
#     # Plot
#     data["vocab_size"] = data["vocab_size"].astype(int)
#     ax = sns.lineplot(data=data, x="vocab_size", y=y_col, marker="o", legend=False)
#
#     # Properties
#     ax.set_title(title)
#     ax.set_xlabel("Vocab. Size")
#     ax.set_ylabel(y_title)
#     plt.tight_layout()
#
#     # Save graphs
#     for ext in ["png", "pdf"]:
#         alias = title.lower().replace(' ', '_').replace('(', '').replace(')', '')
#         path = os.path.join(savepath, f"{y_col}_{alias}.{ext}")
#         plt.savefig(path, dpi=dpi)
#         print(f"Plot saved! ({path})")
#
#     # Show
#     # plt.show()
#     asd = 3
#     plt.close()
#
#
# def main():
#     sizes = [("original", None), ("100k", 100000), ("50k", 50000)]
#     variations = [("unigram", 0), ("unigram+bytes", 256)]
#
#     for size in sizes:
#         for subword_model, extra in variations:
#             # Create preprocessing for training
#             builder = DatasetBuilder(
#                 base_path="/home/scarrion/datasets/nn/translation",
#                 datasets=[
#                     # {"name": "multi30k_test", "languages": ["de-en"], "sizes": [("original", None)]},
#                     {"name": "europarl_lc", "languages": ["de-en"], "sizes": [size]},
#                 ],
#                 subword_models=[subword_model],
#                 vocab_sizes=[x+extra for x in [100, 200, 300, 400, 1000, 2000, 4000, 8000, 16000]],
#                 merge_vocabs=False,
#                 force_overwrite=False,
#                 use_cmd=True,
#                 eval_mode="same",
#                 conda_env_name="mltests",
#                 letter_case="lower",
#             ).build(make_plots=False)
#
#             # Create preprocessing for training and testing
#             tr_datasets = builder.get_ds()
#             ts_datasets = builder.get_ds(ignore_variants=True)
#
#             # Train & Score a model for each dataset
#             rows = []
#             for ds in tr_datasets:
#                 # Get paths
#                 src_enc_file_path = ds.get_encoded_path(fname=f"train.{ds.src_lang}")
#                 trg_enc_file_path = ds.get_encoded_path(fname=f"train.{ds.trg_lang}")
#                 src_vocab_path = ds.get_vocab_file(lang=ds.src_lang) + ".vocab"
#                 trg_vocab_path = ds.get_vocab_file(lang=ds.trg_lang) + ".vocab"
#
#                 # Read vocab
#                 src_words = set([x.split('\t')[0] for x in utils.read_file_lines(src_vocab_path)])
#                 trg_words = set([x.split('\t')[0] for x in utils.read_file_lines(trg_vocab_path)])
#
#                 # Read files
#                 src_enc_file = utils.read_file_lines(src_enc_file_path)
#                 trg_enc_file = utils.read_file_lines(trg_enc_file_path)
#
#                 # Compute avg tokens
#                 src_avg_tokens = sum([len(line.split(' ')) for line in tqdm.tqdm(src_enc_file, total=len(src_enc_file))])/len(src_enc_file)
#                 trg_avg_tokens = sum([len(line.split(' ')) for line in tqdm.tqdm(trg_enc_file, total=len(trg_enc_file))])/len(trg_enc_file)
#
#                 # Compute <unk>
#                 src_unk_tokens = sum([len(set(line.split(' ')).difference(src_words)) for line in tqdm.tqdm(src_enc_file, total=len(src_enc_file))])/len(src_enc_file)
#                 trg_unk_tokens = sum([len(set(line.split(' ')).difference(trg_words)) for line in tqdm.tqdm(trg_enc_file, total=len(trg_enc_file))])/len(trg_enc_file)
#
#                 # Compute unk tokens
#                 row = {"subword_model": ds.subword_model, "vocab_size": ds.vocab_size,
#                        "src_avg_tokens": src_avg_tokens, "trg_avg_tokens": trg_avg_tokens,
#                        "src_unk_tokens": src_unk_tokens, "trg_unk_tokens": trg_unk_tokens,
#                        }
#                 print(row)
#                 rows.append(row)
#
#             # Plot lines
#             ds = ts_datasets[0]
#             ds.subword_model = subword_model
#             ds.vocab_size = ''
#             alias = f"{ds.dataset_name.title()} {size[0]} ({ds.dataset_lang_pair})".replace("_Lc", "-lc")
#             savepath = os.path.join(ds.get_plots_path(), "tokens")
#             make_dir(savepath)
#
#             # Create dataframe
#             data = pd.DataFrame(rows)
#             data.to_csv(os.path.join(savepath, "data_tokens.csv"), index=False)
#
#             title = f"Tokens per Sentence\n{alias}"
#             plot_values(data, y_col="src_avg_tokens", title=f"{title} - {ds.src_lang}", savepath=savepath, y_title="Tokens/Sentence")
#             plot_values(data, y_col="trg_avg_tokens", title=f"{title} - {ds.trg_lang}", savepath=savepath, y_title="Tokens/Sentence")
#
#             title = f"Unknowns per Sentence\n{alias}"
#             plot_values(data, y_col="src_unk_tokens", title=f"{title} - {ds.src_lang}", savepath=savepath, y_title="<UNK>/Sentence")
#             plot_values(data, y_col="trg_unk_tokens", title=f"{title} - {ds.trg_lang}", savepath=savepath, y_title="<UNK>/Sentence")
#
#
# if __name__ == "__main__":
#     main()
