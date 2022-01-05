from autonmt.preprocessing import DatasetBuilder
from autonmt.bundle.report import generate_report

from autonmt.toolkits import AutonmtTranslator
from autonmt.modules.models import Transformer
from autonmt.vocabularies import Vocabulary
from autonmt.toolkits.fairseq import FairseqTranslator

import os
import datetime

import math
import torch.nn as nn
from autonmt.modules.seq2seq import LitSeq2Seq
from autonmt.modules.layers import PositionalEmbedding


def main(fairseq_args, fairseq_args_pred):
    # Create preprocessing for training
    builder = DatasetBuilder(
        base_path="/home/scarrion/datasets/nn/translation",
        datasets=[
            {"name": "multi30k_test", "languages": ["de-en"], "sizes": [("original", None)]},
            # {"name": "europarl", "languages": ["de-en"], "sizes": [("100k", 100000)]},
        ],
        subword_models=["word"],
        vocab_sizes=[1000],
        merge_vocabs=False,
        force_overwrite=False,
        use_cmd=True,
        eval_mode="same",
        conda_env_name="mltests",
        # letter_case="lower",
    ).build(make_plots=False, safe=True)

    # Create preprocessing for training and testing
    tr_datasets = builder.get_ds()
    ts_datasets = builder.get_ds(ignore_variants=True)

    # Train & Score a model for each dataset
    scores = []
    errors = []
    for ds in tr_datasets:
        try:
            model = FairseqTranslator(force_overwrite=True, model_ds=ds, conda_fairseq_env_name="fairseq")
            model.fit(max_epochs=5, patience=10, seed=1234, num_workers=12, fairseq_args=fairseq_args)
            m_scores = model.predict(ts_datasets, metrics={"bleu"}, beams=[1], max_gen_length=150, fairseq_args=fairseq_args_pred)
            scores.append(m_scores)
        except Exception as e:
            print(str(e))
            errors += [str(e)]

    # Make report and print it
    output_path = f".outputs/fairseq/{str(datetime.datetime.now())}"
    df_report, df_summary = generate_report(scores=scores, output_path=output_path, plot_metric="beam1__sacrebleu_bleu_score")
    print("Summary:")
    print(df_summary.to_string(index=False))

    print(f"Errors: {len(errors)}")
    print(errors)


if __name__ == "__main__":
    fairseq_cmd_args = [
        "--arch transformer",
        "--encoder-embed-dim 256",
        "--decoder-embed-dim 256",
        "--encoder-layers 3",
        "--decoder-layers 3",
        "--encoder-attention-heads 8",
        "--decoder-attention-heads 8",
        "--encoder-ffn-embed-dim 512",
        "--decoder-ffn-embed-dim 512",
        "--dropout 0.1",

        "--max-tokens 4096",
        "--no-epoch-checkpoints",
        "--maximize-best-checkpoint-metric",
        "--best-checkpoint-metric bleu",
        "--eval-bleu",
        "--eval-bleu-args '{\"beam\": 5}'",
        "--eval-bleu-print-samples",
        "--scoring sacrebleu",
        "--log-format simple",
        "--task translation",
        # "--wandb-project fairseq",
    ]

    fairseq_cmd_args_pred = [
        "--bpe sentencepiece"
    ]
    main(fairseq_args=fairseq_cmd_args, fairseq_args_pred=fairseq_cmd_args_pred)
