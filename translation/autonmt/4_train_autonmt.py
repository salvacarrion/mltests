from autonmt.preprocessing import DatasetBuilder
from autonmt.bundle.report import generate_report

from autonmt.toolkits import AutonmtTranslator
from autonmt.modules.models import Transformer
from autonmt.vocabularies import Vocabulary

import os
import datetime

import math
import torch.nn as nn
from autonmt.modules.seq2seq import LitSeq2Seq
from autonmt.modules.layers import PositionalEmbedding

from models.transformer_with_adapter import TransformerWithAdapter as Transformer2


def main():
    # Create preprocessing for training
    builder = DatasetBuilder(
        base_path="/home/scarrion/datasets/nn/translation",
        datasets=[
            {"name": "multi30k", "languages": ["de-en"], "sizes": [("original", None)]},
            # {"name": "europarl", "languages": ["de-en"], "sizes": [("original", None)]},
        ],
        subword_models=["word"],
        vocab_sizes=[8000],
        merge_vocabs=False,
        force_overwrite=True,
        use_cmd=True,
        eval_mode="same",
        letter_case="lower",
    ).build(make_plots=False)

    # Create preprocessing for training and testing
    tr_datasets = builder.get_ds()
    ts_datasets = builder.get_ds(ignore_variants=True)

    # Train & Score a model for each dataset
    scores = []
    errors = []
    run_prefix = "model_mt8kemb"
    for ds in tr_datasets:
        # try:

        # Instantiate vocabs and model
        src_vocab = Vocabulary(max_tokens=120).build_from_ds(ds=ds, lang=ds.src_lang)
        trg_vocab = Vocabulary(max_tokens=120).build_from_ds(ds=ds, lang=ds.trg_lang)
        model = Transformer(src_vocab_size=len(src_vocab), trg_vocab_size=len(trg_vocab), padding_idx=src_vocab.pad_id,
                            encoder_embed_dim=256, decoder_embed_dim=256)

        # Train model
        wandb_params = dict(project="autonmt-tests", entity="salvacarrion")
        model = AutonmtTranslator(model=model, src_vocab=src_vocab, trg_vocab=trg_vocab, model_ds=ds,
                                  wandb_params=wandb_params, force_overwrite=True, run_prefix=run_prefix)
        model.fit(max_epochs=100, batch_size=128, seed=1234, num_workers=16, patience=10)
        m_scores = model.predict(ts_datasets, metrics={"bleu"}, beams=[1], max_gen_length=120, load_best_checkpoint=True)
        scores.append(m_scores)

        # except Exception as e:
        #     print(str(e))
        #     errors += [str(e)]

    # Make report and print it
    output_path = f".outputs/autonmt/{str(datetime.datetime.now())}/{run_prefix}"
    df_report, df_summary = generate_report(scores=scores, output_path=output_path, plot_metric="beam1__sacrebleu_bleu_score")
    print("Summary:")
    print(df_summary.to_string(index=False))

    print(f"Errors: {len(errors)}")
    print(errors)


if __name__ == "__main__":
    main()
