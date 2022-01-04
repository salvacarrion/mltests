import pandas as pd
from autonmt.preprocessing import DatasetBuilder
from autonmt.bundle.report import generate_report

from autonmt.toolkits import AutonmtTranslator
from autonmt.modules.models import Transformer
from translation.autonmt.models.transformer_with_adapter import TransformerWithAdapter as Transformer2
from autonmt.vocabularies import Vocabulary

import os
import datetime
import torch
import numpy as np
from autonmt.bundle import utils
from sklearn.decomposition import PCA
import copy


def load_model(ds, run_prefix):
    max_length = 100
    src_vocab = Vocabulary(max_tokens=max_length).build_from_ds(ds=ds, lang=ds.src_lang)
    trg_vocab = Vocabulary(max_tokens=max_length).build_from_ds(ds=ds, lang=ds.trg_lang)
    model = Transformer(src_vocab_size=len(src_vocab), trg_vocab_size=len(trg_vocab), padding_idx=src_vocab.pad_id)

    # Load check point
    checkpoint_path = ds.get_model_checkpoints_path(toolkit="autonmt", run_name=ds.get_run_name(run_prefix),
                                                    fname="checkpoint_best.pt")
    model_state_dict = torch.load(checkpoint_path)['state_dict']
    model.load_state_dict(model_state_dict)
    return model, src_vocab, trg_vocab


def expand_model(model, small_src_vocab, small_trg_vocab, big_ds, src_emb, trg_emb):
    max_length = 100
    big_src_vocab = Vocabulary(max_tokens=max_length).build_from_ds(ds=big_ds, lang=big_ds.src_lang)
    big_trg_vocab = Vocabulary(max_tokens=max_length).build_from_ds(ds=big_ds, lang=big_ds.trg_lang)

    # Get old embedding matrix (small)
    device = model.device
    dtype = model.src_embeddings.weight.dtype
    small_src_emb = model.src_embeddings.weight.detach()
    small_trg_emb = model.trg_embeddings.weight.detach()

    # Get sizes
    src_big_voc_size, src_small_voc_size, src_voc_dim = len(big_src_vocab), small_src_emb.shape[0], small_src_emb.shape[1]
    trg_big_voc_size, trg_small_voc_size, trg_voc_dim = len(big_trg_vocab), small_trg_emb.shape[0], small_trg_emb.shape[1]

    # Match vocabularie
    words_small_src_vocab = small_src_vocab.get_tokens()
    words_small_trg_vocab = small_trg_vocab.get_tokens()
    words_big_src_vocab = big_src_vocab.get_tokens()
    words_big_trg_vocab = big_trg_vocab.get_tokens()
    words_missing_src_vocab = list(set(words_big_src_vocab).difference(set(words_small_src_vocab)))
    words_missing_trg_vocab = list(set(words_big_trg_vocab).difference(set(words_small_trg_vocab)))
    final_big_src_vocab = words_small_src_vocab + words_missing_src_vocab
    final_big_trg_vocab = words_small_trg_vocab + words_missing_trg_vocab
    src_big_sorted_missing_idxs = [big_src_vocab.voc2idx[tok] for tok in words_missing_src_vocab]
    trg_big_sorted_missing_idxs = [big_trg_vocab.voc2idx[tok] for tok in words_missing_trg_vocab]

    # Reserve space for new embeddings
    new_src_emb = torch.zeros((src_big_voc_size, src_voc_dim), device=device, dtype=dtype)
    new_trg_emb = torch.zeros((trg_big_voc_size, trg_voc_dim), device=device, dtype=dtype)

    # Copy old embeddings
    new_src_emb[:src_small_voc_size, :] = small_src_emb
    new_trg_emb[:trg_small_voc_size, :] = small_trg_emb

    # Add new embeddings
    src_idxs = src_emb[torch.tensor(src_big_sorted_missing_idxs)]
    trg_idxs = trg_emb[torch.tensor(trg_big_sorted_missing_idxs)]
    new_src_emb[src_small_voc_size:, :] = torch.tensor(src_idxs, device=device, dtype=dtype)
    new_trg_emb[trg_small_voc_size:, :] = torch.tensor(trg_idxs, device=device, dtype=dtype)

    # Convert embedding to parameter
    model.src_embeddings.weight = torch.nn.parameter.Parameter(new_src_emb)
    model.trg_embeddings.weight = torch.nn.parameter.Parameter(new_trg_emb)

    # Create new vocabs from tokens
    big_src_vocab = Vocabulary(max_tokens=max_length).build_from_tokens([(tok, 0) for tok in final_big_src_vocab])
    big_trg_vocab = Vocabulary(max_tokens=max_length).build_from_tokens([(tok, 0) for tok in final_big_trg_vocab])

    return model, big_src_vocab, big_trg_vocab


def save_embeddings_models(model, savepath):
    src_emb = model.src_embeddings.weight.detach().cpu().numpy()
    trg_emb = model.trg_embeddings.weight.detach().cpu().numpy()

    # Make path
    utils.make_dir(savepath)

    # Save embeddings
    np.save(os.path.join(savepath, "src.npy"), src_emb)
    np.save(os.path.join(savepath, "trg.npy"), trg_emb)
    print(f"Embeddings saved! ({savepath})")


def load_compressed_embeddings(filename, compressor, subword_size, src_emb, trg_emb):
    if compressor in {"random"}:
        # Initialize embeddings
        src_emb = torch.nn.init.normal_(torch.zeros(subword_size, src_emb))
        trg_emb = torch.nn.init.normal_(torch.zeros(subword_size, trg_emb))

    elif compressor in {"pca", "ae"}:
        src_emb = torch.tensor(np.load(os.path.join(filename, f"src_enc_{compressor}.npy")))
        trg_emb = torch.tensor(np.load(os.path.join(filename, f"trg_enc_{compressor}.npy")))
    return src_emb, trg_emb


def get_dataset(datasets, subword_size):
    _dss = [ds for ds in datasets if ds.vocab_size == str(subword_size)]
    return _dss[0]


def main():
    # Create preprocessing for training
    builder = DatasetBuilder(
        base_path="/home/scarrion/datasets/nn/translation",
        datasets=[
            {"name": "multi30k", "languages": ["de-en"], "sizes": [("original", None)]},
            # {"name": "europarl", "languages": ["de-en"], "sizes": [("100k", 100000)]},
        ],
        subword_models=["word"],
        vocab_sizes=[1000, 2000, 4000, 8000],
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
    max_tokens = 100
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compressor = "ae"

    # # Export raw embeddings
    # for ds in tr_datasets:
    #     # Save embeddings
    #     model, src_vocab, trg_vocab = load_model(ds, run_prefix)
    #     save_embeddings_models(model, f".outputs/tmp/256/{str(ds)}")

    pairs = [(1000, 2000), (2000, 4000), (4000, 8000)]
    compressors = ["random", "pca", "ae"]
    rows = []
    for new_emb_size in [256, 512]:
        for sw_small, sw_big in pairs:
            # Get datasets
            ds_small = get_dataset(tr_datasets, sw_small)
            ds_big = get_dataset(tr_datasets, sw_big)

            for comp in compressors:
                # Compress vector
                if comp in {"random"}:
                    src_emb, trg_emb = load_compressed_embeddings(f".outputs/tmp/{new_emb_size}/{str(ds_big)}", comp, subword_size=sw_big, src_emb=256, trg_emb=256)
                elif comp in {"pca", "ae"}:
                    src_emb, trg_emb = load_compressed_embeddings(f".outputs/tmp/{new_emb_size}/{str(ds_big)}", comp, subword_size=sw_big, src_emb=256, trg_emb=256)
                else:
                    raise ValueError("Unknown compressor")

                # Load small model and vocabs
                run_prefix = "model256emb"
                _model, _src_vocab, _trg_vocab = load_model(ds_small, run_prefix)

                # Expand model and vocabs
                model, src_vocab, trg_vocab = expand_model(_model, _src_vocab, _trg_vocab, ds_big, src_emb, trg_emb)
                model = model.to(device)

                # Test model
                model = AutonmtTranslator(model=model, model_ds=ds_small,  src_vocab=src_vocab, trg_vocab=trg_vocab,run_prefix=run_prefix,  force_overwrite=True)
                m_scores = model.predict(eval_datasets=ts_datasets, metrics={"bleu"}, beams=[1], max_gen_length=max_tokens)

                # Keep results
                bleu = m_scores[0]['beams']['beam1']['sacrebleu_bleu_score']
                row = {"sw_small": sw_small, "sw_big": sw_big, "new_emb_size": new_emb_size, "compressor": comp, "bleu": bleu}
                rows.append(row)
                print(row)

    # Create pandas dataframe
    df = pd.DataFrame(rows)
    df.to_csv("results.csv", index=False)
    print(df)


if __name__ == "__main__":
    main()