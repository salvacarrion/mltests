from autonmt.preprocessing import DatasetBuilder
from autonmt.bundle.report import generate_report

from autonmt.toolkits import AutonmtTranslator
from autonmt.modules.models import Transformer
from autonmt.vocabularies import Vocabulary
from autonmt.toolkits.fairseq import FairseqTranslator

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import datetime

import math
import torch.nn as nn
from autonmt.modules.seq2seq import LitSeq2Seq
from autonmt.modules.layers import PositionalEmbedding


def main():
    dataset = "Europarl"
    lang = "de"
    df_pca = pd.read_csv(".outputs/tmp/512/pca_512.csv")
    df_ae_linear = pd.read_csv(".outputs/tmp/512/ae_linear_512.csv")
    df_ae_non_linear = pd.read_csv(".outputs/tmp/512/ae_non_linear_tanh_512.csv")
    df_pca["model"] = "PCA"
    df_ae_linear["model"] = "AE (Linear)"
    df_ae_non_linear["model"] = "AE (Non-linear)"
    df = pd.concat([df_pca, df_ae_linear, df_ae_non_linear])

    # Plot
    data = df[(df.emb_name == "src") & (df.dataset_name == dataset.lower())]
    data = data.copy().reset_index(drop=True)
    ax = sns.lineplot(data=data, x="vocab_size", y="r2", hue="model", marker="o")

    # Properties

    ax.set_title(f"{dataset} ({lang}): Reconstruction Error (512↔256)")
    ax.set_xlabel("Vocab. size")
    ax.set_ylabel("R²")
    ax.legend(title='Models')
    plt.tight_layout()

    # Remove legend's title
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[0:], labels=labels[0:])

    # Save graphs
    for ext in ["png", "pdf"]:
        plt.savefig(f".outputs/tmp/512/{dataset.lower()}_r2_512_{lang}.{ext}", dpi=150)

    plt.show()
    asd = 3


if __name__ == "__main__":

    main()
