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
    dataset_name = "Europarl"
    langs = "de-en"
    data_europarl50k = pd.read_csv(".outputs/fairseq/europarl_50k/reports/report_summary.csv")
    data_europarl50k["train_dataset"] = f"{dataset_name} 50k"
    data = pd.concat([data_europarl50k])

    # Plot
    ax = sns.lineplot(data=data, x="vocab_size", y="fairseq_bleu", hue="train_dataset", marker="o")

    # Properties

    ax.set_title(f"{dataset_name} ({langs})")
    ax.set_xlabel("Vocab. size")
    ax.set_ylabel("BLEU")
    ax.legend(title='train_dataset')
    plt.tight_layout()

    # Remove legend's title
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[0:], labels=labels[0:])

    # Save graphs
    # for ext in ["png", "pdf"]:
    #     plt.savefig(f".outputs/fairseq/{dataset_name.lower()}_{langs}.{ext}", dpi=150)

    plt.show()
    asd = 3
    plt.close()



if __name__ == "__main__":

    main()
