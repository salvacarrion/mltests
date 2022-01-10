import os
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from autonmt.bundle import utils
import pandas as pd
sns.set()


def plot_1ep(aspect_ratio=(17, 9), size=2.5, dpi=300):
    dataset_title = "Europarl-100k (de-en)"
    dataset_name = "europarl"

    for difference in [True, False]:
        for ori_emb in [256, 512]:
            savepath = ".outputs/plots/"
            data = pd.read_csv(os.path.join(f".outputs/csv/{dataset_name}_1ep.csv"))

            if difference:
                title = f"Difference w.r.t. base model: {dataset_title}\nFine-tuning of 1 epoch (emb.: {ori_emb}➔{256})"
                data = data[(data.origin_emb_size == ori_emb) & (data.encoding_algorithm != "Base (0ep)")]
                ylim = (-3, +3)
                y = "difference"
                palette = [sns.color_palette()[i] for i in [1, 2]]
            else:
                title = f"BLEU scores: {dataset_title}\nFine-tuning of 1 epoch (emb.: {ori_emb}➔{256})"
                data = data[(data.origin_emb_size == ori_emb)]
                ylim = (0, 25)
                y = "bleu"
                palette = [sns.color_palette()[i] for i in [0, 1, 2]]

            # Create subplot
            fig = plt.figure(figsize=(aspect_ratio[0] * size, aspect_ratio[1] * size))
            sns.set(font_scale=size)

            # Plot catplot
            g = sns.catplot(data=data, x="from_to", y=y, hue="encoding_algorithm", kind="bar", legend=False,
                            height=aspect_ratio[1], aspect=aspect_ratio[0] / aspect_ratio[1], palette=palette)

            # Add values
            ax = g.facet_axis(0, 0)
            for c in ax.containers:
                labels = ["{:.2f}".format(float(v.get_height())) for v in c]
                ax.bar_label(c, labels=labels, label_type='edge', fontsize=8*size)

            # Properties
            g.set(xlabel="Models", ylabel="BLEU")
            plt.legend(title="Emb. source", loc='lower right')
            g.set(ylim=ylim)
            plt.tight_layout()

            # Save graphs
            utils.make_dir(savepath)
            for ext in ["png", "pdf"]:
                alias = title.lower().replace(' ', '_').replace('(', '').replace(')', '').replace(":", "_").replace('\n', '_')
                path = os.path.join(savepath, f"{alias}.{ext}")
                plt.savefig(path, dpi=dpi)
                print(f"Plot saved! ({path})")

            # Show
            plt.show()

            plt.close()
            asd = 3


def plot_zero_shot(aspect_ratio=(17, 9), size=2.5, dpi=300):
    dataset_title = "Multi30K (de-en)"
    dataset_name = "multi30k"

    for difference in [True, False]:
        for ori_emb in [256, 512]:
            savepath = ".outputs/plots/"
            data = pd.read_csv(os.path.join(f".outputs/csv/{dataset_name}_zero_shot.csv"))

            if difference:
                title = f"Difference w.r.t. base model: {dataset_title}\nZero-shot (emb.: {ori_emb}➔{256})"
                data = data[(data.origin_emb_size == ori_emb) & (data.compressor != "Base")]
                ylim = (-3, +3)
                y = "difference"
                palette = [sns.color_palette()[i] for i in [1, 2]]
            else:
                title = f"BLEU scores: {dataset_title}\nZero-shot (emb.: {ori_emb}➔{256})"
                data = data[(data.origin_emb_size == ori_emb)]
                ylim = (0, 25)
                y = "bleu"
                palette = [sns.color_palette()[i] for i in [0, 1, 2]]

            # Create subplot
            fig = plt.figure(figsize=(aspect_ratio[0] * size, aspect_ratio[1] * size))
            sns.set(font_scale=size)

            # Plot catplot
            g = sns.catplot(data=data, x="from_to", y=y, hue="compressor", kind="bar", legend=False,
                            height=aspect_ratio[1], aspect=aspect_ratio[0] / aspect_ratio[1], palette=palette)

            # Add values
            ax = g.facet_axis(0, 0)
            for c in ax.containers:
                labels = ["{:.2f}".format(float(v.get_height())) for v in c]
                ax.bar_label(c, labels=labels, label_type='edge', fontsize=8*size)

            # Properties
            g.set(xlabel="Models", ylabel="BLEU")
            plt.legend(title="Emb. source", loc='lower right')
            g.set(ylim=ylim)
            plt.tight_layout()

            # Save graphs
            utils.make_dir(savepath)
            for ext in ["png", "pdf"]:
                alias = title.lower().replace(' ', '_').replace('(', '').replace(')', '').replace(":", "_").replace('\n', '_')
                path = os.path.join(savepath, f"{alias}.{ext}")
                plt.savefig(path, dpi=dpi)
                print(f"Plot saved! ({path})")

            # Show
            plt.show()

            plt.close()
            asd = 3



if __name__ == "__main__":
    plot_zero_shot()
