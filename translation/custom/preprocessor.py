import os
import shutil
from pathlib import Path


def check_file_exist(files):
    for p in files:
        if os.path.exists(str(p)):
            raise IOError(f"Missing file: {p}")


def preprocess(destdir, src_lang, trg_lang, trainpref="train", validpref="val", testpref="test",
               src_vocab=None, trg_vocab=None, src_spm_model=None, trg_spm_model=None):

    # Create path
    path = Path(destdir)
    path.mkdir(parents=True, exist_ok=True)

    # Copy split files
    for split_file_pref, split_name in [(trainpref, "train"), (validpref, "val"), (testpref, "test")]:
        if split_file_pref is None:  # Allow for empty values (eg.: testing)
            continue

        for lang in [src_lang, trg_lang]:
            filename = f"{split_file_pref}.{lang}"
            dest_file = os.path.join(destdir, f"{split_name}.{lang}")

            if filename and os.path.exists(filename):
                if not os.path.exists(dest_file):  # Check if the dest file exists
                    shutil.copyfile(filename, dest_file)
                    print(f"\t=> Split file copied: {split_name}.{lang}")
            else:
                raise IOError(f"Missing split file: {filename}")

    # Copy vocabs (check if there is just one) ********************
    trg_vocab = src_vocab if (src_vocab == trg_vocab) else trg_vocab  # Single vocab
    vocab_files = [(src_vocab, f"vocab.{src_lang}"), (trg_vocab, f"vocab.{trg_lang}")]

    # Copy vocab files
    for vocab_file, fname in vocab_files:
        dest_file = os.path.join(destdir, fname)

        if vocab_file and os.path.exists(vocab_file):
            if not os.path.exists(dest_file):  # Check if the dest file exists
                shutil.copyfile(vocab_file, dest_file)
                print(f"\t=> Vocab file copied: {fname}")
        else:
            raise IOError(f"Missing file: {vocab_file}")

    # Copy spm models (check if there is just one) *******************
    if src_spm_model:
        trg_spm_model = src_spm_model if (src_spm_model == trg_spm_model) else trg_spm_model  # Single model
        spm_models_files = [(src_spm_model, f"vocab_{src_lang}.model"), (trg_spm_model, f"spm_{trg_lang}.model")]

        # Copy spm models files
        for spm_file, fname in spm_models_files:
            dest_file = os.path.join(destdir, fname)

            if spm_file and os.path.exists(spm_file):
                if not os.path.exists(dest_file):  # Check if the dest file exists
                    shutil.copyfile(spm_file, dest_file)
                    print(f"\t=> SPM file copied: {fname}")
            else:
                raise IOError(f"Missing spm model: {spm_file}")
    asd = 3