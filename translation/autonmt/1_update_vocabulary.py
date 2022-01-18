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
    model = Transformer(src_vocab_size=len(src_vocab), trg_vocab_size=len(trg_vocab), padding_idx=src_vocab.pad_id)

    # Load check point
    checkpoint_path = ds.get_model_checkpoints_path(toolkit="autonmt", run_name=ds.get_run_name(run_prefix),
                                                    fname="checkpoint_best.pt")
    model_state_dict = torch.load(checkpoint_path)['state_dict']
    model.load_state_dict(model_state_dict)
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
    src_idxs = src_emb[torch.tensor(src_big_sorted_missing_idxs)]
    trg_idxs = trg_emb[torch.tensor(trg_big_sorted_missing_idxs)]
    src_big_tmp = torch.tensor(src_idxs, device=device, dtype=dtype)
    trg_big_tmp = torch.tensor(trg_idxs, device=device, dtype=dtype)

    # Re-scale new tensors (if needed)
    if comp in {"random"}:  # Do not scale for random
        src_big_rescaled, trg_big_rescaled = src_big_tmp, trg_big_tmp
    else:
        # Standarize new tensors (it's already standarize, although since X values have been select, its stats are shifted)
        # src_big_tmp = StandardScaler().fit_transform(src_big_tmp.numpy())
        # trg_big_tmp = StandardScaler().fit_transform(trg_big_tmp.numpy())

        # Rescale new tensors
        src_big_rescaled = src_small_scaler.inverse_transform(src_big_tmp)
        trg_big_rescaled = trg_small_scaler.inverse_transform(trg_big_tmp)

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

    if comp != "random":
        print("******************* Freezing embedding layers *********************")
        for p in model.src_embeddings.parameters():
            p.requires_grad = False
        for p in model.trg_embeddings.parameters():
            p.requires_grad = False

    # Create new vocabs from tokens
    final_big_src_vocab[0] = '⁇'
    final_big_trg_vocab[0] = '⁇'
    big_src_vocab = Vocabulary(max_tokens=max_length, unk_piece='⁇').build_from_tokens([(tok.replace('▁', ''), 0) for tok in final_big_src_vocab])
    big_trg_vocab = Vocabulary(max_tokens=max_length, unk_piece='⁇').build_from_tokens([(tok.replace('▁', ''), 0) for tok in final_big_trg_vocab])

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
    elif compressor in {"glove"}:
        src_emb = torch.nn.init.normal_(torch.zeros(subword_size, src_emb))
        trg_emb = torch.tensor(np.load(os.path.join(filename, f"trg_enc_glove_pca.npy")))
    else:  # standarized
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
            # {"name": "multi30k", "languages": ["de-en"], "sizes": [("original", None)]},
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

    # # Export raw embeddings
    # run_prefix = "transformer256emb"
    # for ds in tr_datasets:
    #     # Save embeddings
    #     model, src_vocab, trg_vocab = load_model(ds, run_prefix)
    #     save_embeddings_models(model, f".outputs/tmp/256/{str(ds)}")  #⁇

    pairs = [(250, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 8000), (8000, 8000)]
    compressors = ["random"]
    rows = []
    for origin_emb_size in [256]:
        for sw_small, sw_big in pairs:
            # Get datasets
            ds_small = get_dataset(tr_datasets, sw_small)
            ds_big = get_dataset(tr_datasets, sw_big)
            assert ds_small.dataset_name == ds_big.dataset_name
            assert ds_small.subword_model == ds_big.subword_model

            for comp in compressors:
                # Compress vector
                src_emb, trg_emb = None, None
                if comp not in {None, "none"}:
                    src_emb, trg_emb = load_compressed_embeddings(f".outputs/tmp/{origin_emb_size}/{str(ds_big)}", comp, subword_size=sw_big, src_emb=256, trg_emb=256)

                # Load small model and vocabs
                run_prefix = f"transformer256emb"  # Don't change it
                model, src_vocab, trg_vocab = load_model(ds_small, run_prefix)
                # run_prefix += f"_zr_oes{origin_emb_size}_c{comp}_{sw_small}-{sw_big}"

                # Expand model and vocabs
                # model, src_vocab, trg_vocab = expand_model(model, src_vocab, trg_vocab, ds_big, src_emb, trg_emb, comp)
                model = model.to(device)

                # Test model
                # ds_small.subword_model = "none"
                wandb_params = None  #dict(project="autonmt-tests", entity="salvacarrion")
                model = AutonmtTranslator(model=model, model_ds=ds_small,  src_vocab=src_vocab, trg_vocab=trg_vocab, wandb_params=wandb_params, run_prefix=run_prefix, force_overwrite=True)
                # model.fit(max_epochs=1, learning_rate=0.0001, optimizer="sgd", batch_size=128, seed=1234, num_workers=0, patience=10)
                m_scores = model.predict(eval_datasets=ts_datasets, metrics={"bleu"}, beams=[1], max_gen_length=max_tokens)
                # ds_small.subword_model = "word"

                # Keep results
                bleu = m_scores[0]['beams']['beam1']['sacrebleu_bleu_score']
                row = {"dataset_name": ds_small.dataset_name, "subword_model": ds_small.subword_model, "from_to": f"{sw_small}➔{sw_big}", "origin_emb_size": origin_emb_size, "compressor": comp, "bleu": bleu}
                rows.append(row)
                print(row)
                asd = 3

    # Create pandas dataframe
    df = pd.DataFrame(rows)
    df.to_csv("europarl.csv", index=False)
    print(df)


if __name__ == "__main__":
    main()
