import os
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from autonmt.bundle import utils
import pandas as pd
sns.set()

from autonmt.preprocessing import DatasetBuilder

import numpy as np
from sklearn.manifold import TSNE

from autonmt.bundle import utils

main_words = ["man", "woman", "lady", "young", "old", "children",
              "jacket", "shirt", "hat", "dress",
              "dog", "cat", "bird", "mouse",
              "running", "soccer", "player",
              "on", "over", "in", "for", "to"]
# main_words = set(main_words[:int(len(main_words)/4)])
print(f"total words: {len(main_words)}")

def main():
    file = "trg"
    path = "/home/scarrion/Documents/Programming/Python/mltests/translation/autonmt/.outputs/tmp/256/multi30k_de-en_original_word_8000/"

    # Create preprocessing for training
    builder = DatasetBuilder(
        base_path="/home/scarrion/datasets/nn/translation",
        datasets=[
            {"name": "multi30k", "languages": ["de-en"], "sizes": [("original", None)]},
            {"name": "europarl", "languages": ["de-en"], "sizes": [("100k", 100000)]},
        ],
        subword_models=["word"],
        vocab_sizes=[500, 1000, 2000, 4000, 8000],
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

    train_tsne = False
    file = "trg_enc_pca"
    for origin_emb_size in [256, 512]:
        for ds in tr_datasets:
            base_path = f".outputs/tmp/{origin_emb_size}/{str(ds)}"

            if train_tsne:
                x = np.load(os.path.join(base_path, f"{file}.npy"))
                x_embedded = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(x)
                np.save(os.path.join(base_path, f"{file}_tsne.npy"), x_embedded)
                print(f"File saved! ({str(ds)})")
            else:
                x = np.load(os.path.join(base_path, f"{file}_tsne.npy"))
                labels = utils.read_file_lines(ds.get_vocab_file("en") + ".vocab")
                labels = [l.split('\t')[0] for l in labels]
                data = pd.DataFrame(data=x, columns=["f1", "f2"])
                data["label"] = labels

                scale = 2.0
                plt.figure(figsize=(12, 12))
                sns.set(font_scale=scale)

                g = sns.scatterplot(
                    x="f1", y="f2",
                    palette=sns.color_palette("hls", 10),
                    data=data,
                    legend="full",
                    alpha=0.3
                )
                # g.set(title=str(ds).replace('_', ' ') + f"\n(source emb. {origin_emb_size})")

                for i, row in data.iterrows():
                    word = row["label"].replace('▁', '')
                    if word in main_words:
                        g.annotate(word, (row["f1"], row["f2"]), fontsize=14)
                plt.tight_layout()

                # Print plot
                savepath = os.path.join(base_path, "plots")
                utils.make_dir(savepath)
                for ext in ["png", "pdf"]:
                    path = os.path.join(savepath, f"tsne_{file}__{str(ds)}.{ext}")
                    plt.savefig(path, dpi=300)
                    print(f"Plot saved! ({path})")

                plt.show()
                plt.close()
                asdas = 3


if __name__ == "__main__":
    main()
