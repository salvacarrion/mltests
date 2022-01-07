# import os
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
# def compute_iou(vocab1, vocab2):
#     return len(set(vocab1).intersection(set(vocab2)))/len(set(vocab1).union(set(vocab2)))
#
#
# def plot_heat_map(data, labels, title, savepath, dpi=300):
#     make_dir(savepath)
#
#     plt.figure(figsize=(12, 12), dpi=dpi)
#     sns.set(font_scale=1.75)
#
#     ax = sns.heatmap(data, annot=True, cbar=False, fmt=".2f")
#     ax.set_xticklabels([x.title() for x in labels], ha='center', minor=False)
#     ax.set_yticklabels([x.title() for x in labels], va='center', minor=False)
#
#     # Properties
#     ax.invert_yaxis()
#     ax.set_title(title, fontdict={'fontsize': 30})
#     plt.tight_layout()
#
#     # Save graphs
#     for ext in ["png", "pdf"]:
#         alias = title.lower().replace(' ', '_').replace('(', '').replace(')', '').replace("_-_", "_")
#         path = os.path.join(savepath, f"heat_map_{alias}.{ext}")
#         plt.savefig(path, dpi=dpi)
#         print(f"Plot saved! ({path})")
#
#     # Show
#     # plt.show()
#     asd = 3
#
#     plt.close()
#
#
#
# def main():
#     sizes = [("100k", 100000), ("50k", 50000)]
#     variations = [("word", 0)]
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
#             vocabs = []
#             for ds in tr_datasets:
#                 # Get paths
#                 src_vocab_path = ds.get_vocab_file(lang=ds.src_lang) + ".vocab"
#                 trg_vocab_path = ds.get_vocab_file(lang=ds.trg_lang) + ".vocab"
#
#                 # Read vocab
#                 src_words = [x.split('\t')[0] for x in utils.read_file_lines(src_vocab_path)]
#                 trg_words = [x.split('\t')[0] for x in utils.read_file_lines(trg_vocab_path)]
#
#                 # Add vocabs
#                 vocabs.append((ds, src_words, trg_words))
#
#             # Compute IoUs
#             src_heat_map = np.zeros((len(vocabs), len(vocabs)))
#             trg_heat_map = np.zeros((len(vocabs), len(vocabs)))
#             for i, vocab1 in enumerate(vocabs):
#                 for j, vocab2 in enumerate(vocabs):
#                     src_heat_map[i, j] = compute_iou(vocab1[1], vocab2[1])
#                     trg_heat_map[i, j] = compute_iou(vocab1[2], vocab2[2])
#
#             # Plot heat maps
#             ds = ts_datasets[0]
#             ds.subword_model = subword_model
#             ds.vocab_size = ''
#             alias = f"{ds.dataset_name.title()} {size[0]} ({ds.dataset_lang_pair})".replace("_Lc", "-lc")
#             savepath = os.path.join(ds.get_plots_path(), "heatmap")
#             make_dir(savepath)
#
#             title = f"IoU for {alias}"
#             labels = [ds.vocab_size for ds in tr_datasets]
#             plot_heat_map(src_heat_map, labels, title=f"{title} - {ds.src_lang}", savepath=savepath)
#             plot_heat_map(trg_heat_map, labels, title=f"{title} - {ds.trg_lang}", savepath=savepath)
#
#
# if __name__ == "__main__":
#     main()
