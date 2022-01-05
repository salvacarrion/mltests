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


class TransformerWithAdapter(LitSeq2Seq):
    def __init__(self,
                 src_vocab_size, trg_vocab_size,
                 encoder_embed_dim_emb=512,
                 decoder_embed_dim_emb=512,
                 encoder_embed_dim=256,
                 decoder_embed_dim=256,
                 encoder_layers=3,
                 decoder_layers=3,
                 encoder_attention_heads=8,
                 decoder_attention_heads=8,
                 encoder_ffn_embed_dim=512,
                 decoder_ffn_embed_dim=512,
                 dropout=0.1,
                 activation_fn="relu",
                 max_src_positions=1024,
                 max_trg_positions=1024,
                 padding_idx=None,
                 learned=False,
                 **kwargs):
        super().__init__(src_vocab_size, trg_vocab_size, padding_idx, **kwargs)
        self.max_src_positions = max_src_positions
        self.max_trg_positions = max_trg_positions

        # Model
        self.src_embeddings = nn.Embedding(src_vocab_size, encoder_embed_dim_emb)
        self.trg_embeddings = nn.Embedding(trg_vocab_size, decoder_embed_dim_emb)
        self.src_pos_embeddings = PositionalEmbedding(num_embeddings=max_src_positions, embedding_dim=encoder_embed_dim_emb, padding_idx=padding_idx, learned=learned)
        self.trg_pos_embeddings = PositionalEmbedding(num_embeddings=max_trg_positions, embedding_dim=decoder_embed_dim_emb, padding_idx=padding_idx, learned=learned)
        self.src_dense_emb = nn.Linear(encoder_embed_dim_emb, encoder_embed_dim)
        self.trg_dense_emb = nn.Linear(decoder_embed_dim_emb, decoder_embed_dim)
        self.transformer = nn.Transformer(d_model=encoder_embed_dim,
                                          nhead=encoder_attention_heads,
                                          num_encoder_layers=encoder_layers,
                                          num_decoder_layers=decoder_layers,
                                          dim_feedforward=encoder_ffn_embed_dim,
                                          dropout=dropout,
                                          activation=activation_fn)
        self.output_layer = nn.Linear(encoder_embed_dim, src_vocab_size)
        self.input_dropout = nn.Dropout(dropout)

        # Checks
        assert encoder_embed_dim == decoder_embed_dim
        assert encoder_attention_heads == decoder_attention_heads
        assert encoder_ffn_embed_dim == decoder_ffn_embed_dim

    def forward_encoder(self, x):
        assert x.shape[1] <= self.max_src_positions

        # Encode src
        x_pos = self.src_pos_embeddings(x)
        x_emb = self.src_embeddings(x)
        x_emb = (x_emb + x_pos)
        x_emb = self.src_dense_emb(x_emb)
        x_emb = x_emb.transpose(0, 1)

        memory = self.transformer.encoder(src=x_emb, mask=None, src_key_padding_mask=None)
        return memory

    def forward_decoder(self, y, memory):
        assert y.shape[1] <= self.max_trg_positions

        # Encode trg
        y_pos = self.trg_pos_embeddings(y)
        y_emb = self.trg_embeddings(y)
        y_emb = (y_emb + y_pos)
        y_emb = self.trg_dense_emb(y_emb)
        y_emb = y_emb.transpose(0, 1)

        # Make trg mask
        tgt_mask = self.transformer.generate_square_subsequent_mask(y_emb.shape[0]).to(y_emb.device)

        output = self.transformer.decoder(tgt=y_emb, memory=memory, tgt_mask=tgt_mask, memory_mask=None,
                                          tgt_key_padding_mask=None, memory_key_padding_mask=None)

        # Get output
        output = output.transpose(0, 1)
        output = self.output_layer(output)
        return output


def main():
    # Create preprocessing for training
    builder = DatasetBuilder(
        base_path="/home/scarrion/datasets/nn/translation",
        datasets=[
            {"name": "multi30k", "languages": ["de-en"], "sizes": [("original", None)]},
            {"name": "europarl", "languages": ["de-en"], "sizes": [("100k", 100000)]},
        ],
        subword_models=["word"],
        vocab_sizes=[250, 500],
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

    # Train & Score a model for each dataset
    scores = []
    errors = []
    run_prefix = "transformer256emb"
    for ds in tr_datasets:
        try:
            # Instantiate vocabs and model
            src_vocab = Vocabulary(max_tokens=100).build_from_ds(ds=ds, lang=ds.src_lang)
            trg_vocab = Vocabulary(max_tokens=100).build_from_ds(ds=ds, lang=ds.trg_lang)
            model = Transformer(src_vocab_size=len(src_vocab), trg_vocab_size=len(trg_vocab), padding_idx=src_vocab.pad_id)

            # Train model
            # wandb_params = None
            wandb_params = dict(project="autonmt-tests", entity="salvacarrion")
            model = AutonmtTranslator(model=model, model_ds=ds, src_vocab=src_vocab, trg_vocab=trg_vocab, run_prefix=run_prefix, wandb_params=wandb_params, force_overwrite=True)
            model.fit(max_epochs=100, batch_size=512, seed=1234, num_workers=12, patience=10, strategy="dp")
            m_scores = model.predict(ts_datasets, metrics={"bleu"}, beams=[1], max_gen_length=150)
            scores.append(m_scores)
        except Exception as e:
            print(str(e))
            errors += [str(e)]

    # Make report and print it
    output_path = f".outputs/autonmt/{str(datetime.datetime.now())}"
    df_report, df_summary = generate_report(scores=scores, output_path=output_path, plot_metric="beam1__sacrebleu_bleu_score")
    print("Summary:")
    print(df_summary.to_string(index=False))

    print(f"Errors: {len(errors)}")
    print(errors)


if __name__ == "__main__":
    main()
