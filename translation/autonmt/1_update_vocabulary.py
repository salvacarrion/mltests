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
from sklearn.preprocessing import StandardScaler

import torch
import random
import numpy as np
from pytorch_lightning.utilities.seed import seed_everything

# Define seed

# Set seeds
seed=1234
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
seed_everything(seed)

# Tricky: https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
# torch.use_deterministic_algorithms(True)

# Test randomness
print(f"\t- [INFO]: Testing random seed ({seed}):")
print(f"\t\t- random: {random.random()}")
print(f"\t\t- numpy: {np.random.rand(1)}")
print(f"\t\t- torch: {torch.rand(1)}")


def load_model(ds, run_prefix):
    max_length = 100
    src_vocab = Vocabulary(max_tokens=max_length).build_from_ds(ds=ds, lang=ds.src_lang)
    trg_vocab = Vocabulary(max_tokens=max_length).build_from_ds(ds=ds, lang=ds.trg_lang)
    model = Transformer(src_vocab_size=len(src_vocab), trg_vocab_size=len(trg_vocab), padding_idx=src_vocab.pad_id,
                            encoder_embed_dim=256, decoder_embed_dim=256)

    # Load check point
    if run_prefix:
        checkpoint_path = ds.get_model_checkpoints_path(toolkit="autonmt", run_name=ds.get_run_name(run_prefix),
                                                        fname="checkpoint_best.pt")
        model_state_dict = torch.load(checkpoint_path)['state_dict']
        model.load_state_dict(model_state_dict)

        # Change vocabularies
        src_tokens = [tok.replace('▁', '') for tok in src_vocab.get_tokens()]
        trg_tokens = [tok.replace('▁', '') for tok in trg_vocab.get_tokens()]
        src_tokens[0] = '⁇'
        trg_tokens[0] = '⁇'
        src_vocab = Vocabulary(max_tokens=max_length, unk_piece='⁇').build_from_tokens([(tok, 0) for tok in src_tokens])
        trg_vocab = Vocabulary(max_tokens=max_length, unk_piece='⁇').build_from_tokens([(tok, 0) for tok in trg_tokens])
    return model, src_vocab, trg_vocab


def expand_model(model, small_src_vocab, small_trg_vocab, big_ds, src_emb, trg_emb, comp):
    max_length = 100
    big_src_vocab = Vocabulary(max_tokens=max_length).build_from_ds(ds=big_ds, lang=big_ds.src_lang)
    big_trg_vocab = Vocabulary(max_tokens=max_length).build_from_ds(ds=big_ds, lang=big_ds.trg_lang)

    # Get old embedding matrix (small)
    device = model.device
    dtype = model.src_embeddings.weight.dtype
    small_src_emb = model.src_embeddings.weight.detach()
    small_trg_emb = model.trg_embeddings.weight.detach()

    # Compute mean and std
    src_small_scaler = StandardScaler().fit(small_src_emb.numpy())
    trg_small_scaler = StandardScaler().fit(trg_emb.numpy())

    # Get sizes
    src_big_voc_size, src_small_voc_size, src_voc_dim = len(big_src_vocab), small_src_emb.shape[0], small_src_emb.shape[1]
    trg_big_voc_size, trg_small_voc_size, trg_voc_dim = len(big_trg_vocab), small_trg_emb.shape[0], small_trg_emb.shape[1]

    # Match vocabularies
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
    src_idxs = src_emb[torch.tensor(src_big_sorted_missing_idxs).long()]
    trg_idxs = trg_emb[torch.tensor(trg_big_sorted_missing_idxs).long()]
    src_big_tmp = torch.tensor(src_idxs, device=device, dtype=dtype)
    trg_big_tmp = torch.tensor(trg_idxs, device=device, dtype=dtype)

    # Re-scale new tensors (if needed)
    if comp in {"random"}:  # Do not scale for random
        src_big_rescaled, trg_big_rescaled = src_big_tmp, trg_big_tmp
    else:
        # Standarize new tensors (it's already standarize, although since X values have been select, its stats are shifted)
        # src_big_tmp = StandardScaler().fit_transform(src_big_tmp.numpy())
        # trg_big_tmp = StandardScaler().fit_transform(trg_big_tmp.numpy())

        src_big_rescaled, trg_big_rescaled = src_big_tmp, trg_big_tmp
        # Rescale new tensors
        # src_big_rescaled = src_small_scaler.inverse_transform(src_big_tmp)
        # trg_big_rescaled = trg_small_scaler.inverse_transform(trg_big_tmp)

    # Inverse transform but with the previous model stats
    new_src_emb[src_small_voc_size:, :] = torch.tensor(src_big_rescaled, device=device, dtype=dtype)
    new_trg_emb[trg_small_voc_size:, :] = torch.tensor(trg_big_rescaled, device=device, dtype=dtype)

    # Convert embedding to parameter
    model.src_embeddings.weight = torch.nn.parameter.Parameter(new_src_emb)
    model.trg_embeddings.weight = torch.nn.parameter.Parameter(new_trg_emb)

    # Modify output layer
    new_output = torch.nn.Linear(model.output_layer.in_features, trg_big_voc_size, device=device, dtype=dtype)
    new_output_weights = new_output.weight.detach()
    new_output_bias = new_output.bias.detach()
    new_output_weights[:src_small_voc_size, :] = model.output_layer.weight.detach()
    new_output_bias[:src_small_voc_size] = model.output_layer.bias.detach()
    new_output.weight = torch.nn.parameter.Parameter(new_output_weights)
    new_output.bias = torch.nn.parameter.Parameter(new_output_bias)
    model.output_layer = new_output

    # if comp != "random":
    #     print("******************* Freezing embedding layers *********************")
    #     if comp != "glove":
    #         for p in model.src_embeddings.parameters():
    #             p.requires_grad = False
    #     for p in model.trg_embeddings.parameters():
    #         p.requires_grad = False

    # Create new vocabs from tokens
    final_big_src_vocab[0] = '⁇'
    final_big_trg_vocab[0] = '⁇'
    big_src_vocab = Vocabulary(max_tokens=max_length, unk_piece='⁇').build_from_tokens([(tok.replace('▁', ''), 0) for tok in final_big_src_vocab])
    big_trg_vocab = Vocabulary(max_tokens=max_length, unk_piece='⁇').build_from_tokens([(tok.replace('▁', ''), 0) for tok in final_big_trg_vocab])

    # Reset model
    # model.apply(weight_reset)

    return model, big_src_vocab, big_trg_vocab


def model_with_embeddings(src_emb, trg_emb):
    # Instantiate model
    model = Transformer(src_vocab_size=len(src_emb), trg_vocab_size=len(trg_emb), padding_idx=3,
                        encoder_embed_dim=256, decoder_embed_dim=256)
    device = model.device
    dtype = model.src_embeddings.weight.dtype

    # Numpy2Tensor
    src_emb = torch.tensor(src_emb, dtype=dtype, device=device)
    trg_emb = torch.tensor(trg_emb, dtype=dtype, device=device)

    # Convert embedding to parameter
    model.src_embeddings.weight = torch.nn.parameter.Parameter(src_emb)
    model.trg_embeddings.weight = torch.nn.parameter.Parameter(trg_emb)

    # Freeze layer
    for p in model.src_embeddings.parameters():
        p.requires_grad = False
    for p in model.trg_embeddings.parameters():
        p.requires_grad = False
    return model


def weight_reset(m):
    reset_parameters = getattr(m, "reset_parameters", None)
    if callable(reset_parameters):
        m.reset_parameters()


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
    elif compressor in {"glove"}:
        src_emb = torch.nn.init.normal_(torch.zeros(subword_size, src_emb))
        trg_emb = torch.tensor(np.load(os.path.join(filename, f"trg_enc_glove_pca.npy")))
    elif compressor in {None, "none"}:
        src_emb = torch.tensor(np.load(os.path.join(filename, f"src.npy")))
        trg_emb = torch.tensor(np.load(os.path.join(filename, f"trg.npy")))
    else:  # standarized
        src_emb = torch.tensor(np.load(os.path.join(filename, f"src_enc_{compressor}.npy")))
        trg_emb = torch.tensor(np.load(os.path.join(filename, f"trg_enc_{compressor}.npy")))
    return src_emb, trg_emb


def get_ref_vocabs(ds_ref, limit, max_length=100):
    src_vocab = Vocabulary(max_tokens=max_length).build_from_ds(ds_ref, lang=ds_ref.src_lang)
    trg_vocab = Vocabulary(max_tokens=max_length).build_from_ds(ds_ref, lang=ds_ref.trg_lang)

    # Create new vocabs from tokens
    src_tokens = [tok.replace('▁', '') for tok in src_vocab.get_tokens()[:limit]]
    trg_tokens = [tok.replace('▁', '') for tok in trg_vocab.get_tokens()[:limit]]
    src_tokens[0] = '⁇'
    trg_tokens[0] = '⁇'
    src_vocab = Vocabulary(max_tokens=max_length, unk_piece='⁇').build_from_tokens([(tok, 0) for tok in src_tokens])
    trg_vocab = Vocabulary(max_tokens=max_length, unk_piece='⁇').build_from_tokens([(tok, 0) for tok in trg_tokens])

    return src_vocab, trg_vocab


def get_ref_vocabs2(base_path, limit, max_length=100):
    src_tokens = utils.load_json(os.path.join(base_path, "src.json"))
    trg_tokens = utils.load_json(os.path.join(base_path, "trg.json"))
    src_vocab = Vocabulary(max_tokens=max_length).build_from_tokens([(tok, 0) for tok in src_tokens])
    trg_vocab = Vocabulary(max_tokens=max_length).build_from_tokens([(tok, 0) for tok in trg_tokens])

    # Create new vocabs from tokens
    src_tokens = [tok.replace('▁', '') for tok in src_vocab.get_tokens()[:limit]]
    trg_tokens = [tok.replace('▁', '') for tok in trg_vocab.get_tokens()[:limit]]
    src_tokens[0] = '⁇'
    trg_tokens[0] = '⁇'
    src_vocab = Vocabulary(max_tokens=max_length, unk_piece='⁇').build_from_tokens([(tok, 0) for tok in src_tokens])
    trg_vocab = Vocabulary(max_tokens=max_length, unk_piece='⁇').build_from_tokens([(tok, 0) for tok in trg_tokens])

    return src_vocab, trg_vocab

def get_dataset(datasets, subword_size):
    _dss = [ds for ds in datasets if ds.vocab_size == str(subword_size)]
    return _dss[0]


def main():
    # Create preprocessing for training
    builder_big = DatasetBuilder(
        base_path="/home/scarrion/datasets/nn/translation",
        datasets=[
            # {"name": "multi30k", "languages": ["de-en"], "sizes": [("original", None)]},
            {"name": "europarl", "languages": ["de-en"], "sizes": [("original_lc", None)]},
            # {"name": "europarl", "languages": ["de-en"], "sizes": [("100k", 100000)]},
        ],
        subword_models=["word"],
        vocab_sizes=[16000],
        merge_vocabs=False,
        force_overwrite=False,
        use_cmd=False,
        eval_mode="same",
        letter_case="lower",
    ).build(make_plots=False, safe=True)
    big_datasets = builder_big.get_ds()
    ds_ref = big_datasets[0]

    # Create preprocessing for training
    builder = DatasetBuilder(
        base_path="/home/scarrion/datasets/nn/translation",
        datasets=[
            # {"name": "multi30k", "languages": ["de-en"], "sizes": [("original", None)]},
            # {"name": "europarl", "languages": ["de-en"], "sizes": [("original_lc", None)]},
            {"name": "europarl", "languages": ["de-en"], "sizes": [("100k", 100000)]},
        ],
        subword_models=["word"],
        vocab_sizes=[250, 500, 1000, 2000, 4000, 8000],
        merge_vocabs=False,
        force_overwrite=False,
        use_cmd=False,
        eval_mode="same",
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

    # Export raw embeddings
    # run_prefix = "model"
    # for ds in tr_datasets:
    #     # Save embeddings
    #     model, src_vocab, trg_vocab = load_model(ds, run_prefix)
    #     save_embeddings_models(model, f".outputs/tmp/256/{str(ds).replace('_test', '')}")  #⁇

    # pairs = [(250, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 8000)]  #  (500, 1000), (1000, 2000), (2000, 4000), (4000, 8000), (8000, 8000)
    pairs = [(8000, 16000), (4000, 8000), (2000, 4000)]  #  (500, 1000), (1000, 2000), (2000, 4000), (4000, 8000), (8000, 8000)
    compressors = ["none"]
    rows = []
    batch_size = 64
    for origin_emb_size in [256]:
        for sw_small, sw_big in pairs:
            # Get embeddings/vocabs
            # src_emb, trg_emb = load_compressed_embeddings(f".outputs/tmp/256/{str(ds_ref)}", compressor="none", subword_size=sw_big, src_emb=256, trg_emb=256)
            # src_vocab, trg_vocab = get_ref_vocabs(ds_ref, limit=16000)

            src_emb, trg_emb = load_compressed_embeddings(f".outputs/tmp/256/fasttext256_europarl_lc_16k", compressor="none", subword_size=sw_big, src_emb=256, trg_emb=256)
            src_vocab, trg_vocab = get_ref_vocabs(ds_ref, limit=sw_small)
            # src_vocab2, trg_vocab2 = get_ref_vocabs2(f".outputs/tmp/256/fasttext256_europarl_lc_16k", limit=16000)

            # Load small model and vocabs
            ds_small = get_dataset(tr_datasets, sw_small)
            run_prefix = "model_fasttext256_16k__europarl100k"  #f"model_eu8kemb"  # model_mt8kemb
            model = model_with_embeddings(src_emb, trg_emb)

            # Load checkpoint (to evaluate)
            # ds_small.subword_model = "none"
            # checkpoint_path = ds_small.get_model_checkpoints_path(toolkit="autonmt", run_name=ds_small.get_run_name(run_prefix), fname="checkpoint_best.pt")
            # model_state_dict = torch.load(checkpoint_path)['state_dict']
            # model.load_state_dict(model_state_dict)

            model = model.to(device)

            # Test model
            ds_small.subword_model = "none"
            wandb_params = None  #dict(project="autonmt-tests", entity="salvacarrion")
            model = AutonmtTranslator(model=model, model_ds=ds_small,  src_vocab=src_vocab, trg_vocab=trg_vocab, wandb_params=wandb_params, run_prefix=run_prefix, force_overwrite=True)
            model.fit(max_epochs=30, learning_rate=0.001, optimizer="adam", batch_size=batch_size, seed=1234, num_workers=0, patience=10)
            m_scores = model.predict(eval_datasets=ts_datasets, metrics={"bleu"}, beams=[1], max_gen_length=max_tokens, batch_size=batch_size)
            ds_small.subword_model = "word"

            # Keep results
            bleu = m_scores[0]['beams']['beam1']['sacrebleu_bleu_score']
            row = {"dataset_name": ds_small.dataset_name, "subword_model": ds_small.subword_model, "from_to": f"{sw_small}", "origin_emb_size": origin_emb_size, "compressor": "none", "bleu": bleu}
            rows.append(row)
            print(row)
            asd = 3

    # Create pandas dataframe
    df = pd.DataFrame(rows)
    df.to_csv("europarl2M.csv", index=False)
    print(df)


if __name__ == "__main__":
    main()
