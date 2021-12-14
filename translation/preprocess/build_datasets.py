import os
from pathlib import Path
from itertools import islice
from collections import defaultdict
import random
random.seed(123)

from translation.preprocess.utils import *
from translation.autonmt import commands

CONDA_ENVNAME = "mltests"


def create_splits(base_path, datasets, val_size, test_size, shuffle, force_overwrite):
    print("Creating splits...")

    # LEVEL 0: Dataset names
    for ds in datasets:  # Dataset
        ds_name = ds["name"]

        # LEVEL 1: Sizes
        for ds_size_name, ds_max_lines in [("original", None)]:  # Lengths

            # LEVEL 2: Languages
            for lang_pair in ds["languages"]:  # Languages
                src_lang, trg_lang = lang_pair.split("-")

                # Check if the split folder exists
                splits_path = os.path.join(base_path, ds_name, ds_size_name, lang_pair, "data", "splits")
                if not force_overwrite and os.path.exists(splits_path):
                    print("\t\t=> Skipping split creation as the folder already exists")
                    continue

                # Create splits folder
                path = Path(splits_path)
                path.mkdir(parents=True, exist_ok=True)

                # Read raw files
                raw_path = os.path.join(base_path, ds_name, ds_size_name, lang_pair, "data", "raw")
                print(f"\t=> Processing raw files: {raw_path}")
                with open(os.path.join(raw_path, f"data.{src_lang}"), 'r') as f:
                    src_lines = f.readlines()
                with open(os.path.join(raw_path, f"data.{trg_lang}"), 'r') as f:
                    trg_lines = f.readlines()

                # Clean lines
                lines = preprocess_pairs(src_lines, trg_lines, shuffle=shuffle)

                # Check size type
                val_size = int(val_size*len(lines)) if isinstance(val_size, float) else val_size
                test_size = int(test_size*len(lines)) if isinstance(test_size, float) else test_size

                # Create partitions
                assert len(lines) > (val_size+test_size)*2  # At least 50% of training data
                train_lines = lines[:-(val_size+test_size)]
                val_lines = lines[-(val_size+test_size):-test_size]
                test_lines = lines[-test_size:]
                splits = [(train_lines, "train"), (val_lines, "val"), (test_lines, "test")]

                # Save partitions
                for split_lines, split_name in splits:
                    for i, split_lang in enumerate([src_lang, trg_lang]):  # Languages
                        savepath = os.path.join(splits_path, f"{split_name}.{split_lang}")
                        with open(savepath, 'w') as fs:
                            lines = [line[i] + '\n' for line in split_lines]  # split_lines is a tuple (src, trg)
                            fs.writelines(lines)


def create_reduced_versions(base_path, datasets, autofix):
    print("Creating reduced versions...")

    # LEVEL 0: Dataset names
    for ds in datasets:  # Dataset
        ds_name = ds["name"]

        # LEVEL 1: Sizes
        for ds_size_name, ds_max_lines in ds["sizes"]:  # Lengths

            # LEVEL 2: Languages
            for lang_pair in ds["languages"]:  # Languages
                src_lang, trg_lang = lang_pair.split("-")
                trans_files = get_translation_files(src_lang, trg_lang)

                if ds_size_name == "original":  # Check for splits
                    for trans_fname in trans_files:
                        # Check level 3.1: Translation files
                        ds_level = os.path.join(ds_name, ds_size_name, lang_pair, "data", "splits", trans_fname)
                        fname = os.path.join(base_path, ds_level)
                        if not os.path.exists(fname):
                            print(f"- Checking file '{ds_level}' => Failed")
                            continue

                else:  # Create folder
                    # Create folders (recursively)
                    ds_level = os.path.join(ds_name, ds_size_name, lang_pair, "data", "splits")
                    ds_level_path = os.path.join(base_path, ds_level)

                    if not os.path.exists(ds_level_path):
                        if not autofix:
                            print(f"- Checking dataset '{ds_level}' => Failed")
                            continue
                        else:
                            # Create folder recursively
                            if not os.path.exists(ds_level_path):
                                print(f"- Autofix for '{ds_level}': Creating directories...")
                                path = Path(ds_level_path)
                                path.mkdir(parents=True, exist_ok=True)

                            # Add trunc file names
                            for trans_fname in trans_files:
                                # Check level 3.1: Translation files
                                ori_filename = os.path.join(base_path, ds_name, "original", lang_pair, "data", "splits", trans_fname)
                                new_filename = os.path.join(base_path, ds_name, ds_size_name, lang_pair, "data", "splits", trans_fname)

                                # Copy n lines efficiently
                                if not os.path.exists(new_filename):
                                    with open(ori_filename, 'r') as fin, open(new_filename, 'w') as fout:
                                        lines = list(islice(fin, ds_max_lines))
                                        fout.writelines(lines)
                                        print(f"- Autofix for '{ds_level}': Creating {trans_fname}...")


def pretokenize(base_path, datasets, force_overwrite):
    print("Pretokenizing splits...")

    # 1. Normalization => Whitespace, Strip accents, Unicode normalization,...
    # 2. Pre-Tokenization => Split text into "words" (hello world !)
    # 3. (subword) Tokenizer => BPE, Unigram, WordPiece,...
    # 4. Post-Processing => <unk>, <sos>, <eos>, max lengths,...
    # Here: Normalization (Python) + Pre-Tokenization (sacremoses) + BPE (fastBPE vs. subword-nmt)
    for ds in datasets:  # Dataset
        ds_name = ds["name"]
        for ds_size_name, ds_max_lines in ds["sizes"]:  # Lengths
            for lang_pair in ds["languages"]:  # Languages
                src_lang, trg_lang = lang_pair.split("-")
                trans_files = get_translation_files(src_lang, trg_lang)

                # Create folder
                pretokenize_path = os.path.join(base_path, ds_name, ds_size_name, lang_pair, "data", "pretokenized")
                path = Path(pretokenize_path)
                path.mkdir(parents=True, exist_ok=True)

                # Process each file
                print(f"\t=> Pretokenizing: {pretokenize_path}")
                for trans_fname in trans_files:
                    file_lang = trans_fname.split(".")[-1]
                    ori_filename = os.path.join(base_path, ds_name, ds_size_name, lang_pair, "data", "splits", trans_fname)
                    new_filename = os.path.join(pretokenize_path, trans_fname)

                    # Check if the pretokenized files already exists
                    if not force_overwrite and os.path.exists(new_filename):
                        print("\t\t=> Skipping pretokenization as this file already exists")
                    else:
                        # Tokenize (sacremoses): https://github.com/alvations/sacremoses
                        commands.moses_tokenizer(lang=file_lang, input_file=ori_filename, output_file=new_filename)


def build_vocab(base_path, datasets, encoding_mode, subword_model, vocab_size, character_coverage, merge_trains, force_overwrite):
    print(f"- Building vocabs: (encoding_mode={encoding_mode}; subword_model={subword_model}; vocab_size={vocab_size})")

    for ds in datasets:  # Dataset
        ds_name = ds["name"]
        for ds_size_name, ds_max_lines in ds["sizes"]:  # Lengths
            for lang_pair in ds["languages"]:  # Languages
                src_lang, trg_lang = lang_pair.split("-")
                trans_files = get_translation_files(src_lang, trg_lang)

                # Create dirs
                vocab_path = os.path.join(base_path, ds_name, ds_size_name, lang_pair, "vocabs", "spm")
                vocab_data_path = os.path.join(vocab_path, "data")
                model_vocab_path = os.path.join(vocab_path, subword_model, str(vocab_size))
                print(f"\t=> Build vocab for: {vocab_path}")
                for p in [vocab_path, vocab_data_path, model_vocab_path]:
                    path = Path(p)
                    path.mkdir(parents=True, exist_ok=True)

                # Concatenate train files
                if not merge_trains:
                    raise NotImplementedError("Only merge train files is allowed")
                else:
                    # Concat training sets
                    train_concat_fname = "train_pretok.txt" if (encoding_mode == "pretokenized" or subword_model == "word") else "train_raw.txt"
                    new_filename = os.path.join(vocab_data_path, train_concat_fname)

                    # Check if concatenated train file exists
                    if not force_overwrite and os.path.isfile(new_filename):
                        print("\t\t=> Skipping concatenated train file as it already exists")
                        # raise IOError("Concatenated train file exists")
                    else:
                        # Concat train files
                        with open(new_filename, 'w') as outfile:
                            for trans_fname in [f"train.{src_lang}", f"train.{trg_lang}"]:
                                raw_folder = "pretokenized" if (encoding_mode == "pretokenized" or subword_model == "word") else "splits"
                                ori_filename = os.path.join(base_path, ds_name, ds_size_name, lang_pair, "data", raw_folder, trans_fname)
                                with open(ori_filename) as infile:
                                    outfile.write(infile.read())

                # Train model
                model_prefix = os.path.join(model_vocab_path, f"spm_{src_lang}-{trg_lang}")  # without .model
                if not force_overwrite and os.path.exists(model_prefix):
                    print("\t\t=> Skipping spm training as it already exists")
                else:
                    commands.spm_train(input_file=new_filename, model_prefix=model_prefix, vocab_size=vocab_size, character_coverage=character_coverage, subword_model=subword_model)


def encode_datasets(base_path, datasets, encoding_mode, subword_model, vocab_size, min_vocab_frequency, export_frequencies, force_overwrite):
    print(f"- Encoding datasets: (encoding_mode={encoding_mode}; min_vocab_frequency={min_vocab_frequency})")

    # Sentenpiece restrictions (I don't know why)
    if min_vocab_frequency > 1 and subword_model == "word":
        raise ValueError("Vocabulary constraint is only enabled in subword units.")
    elif min_vocab_frequency > 1:
        print("\t\t=> Ignoring 'min_vocab_frequency'. It does not work as expected.")

    for ds in datasets:  # Dataset
        ds_name = ds["name"]
        for ds_size_name, ds_max_lines in ds["sizes"]:  # Lengths
            for lang_pair in ds["languages"]:  # Languages
                src_lang, trg_lang = lang_pair.split("-")
                trans_files = get_translation_files(src_lang, trg_lang)

                # Get vocab dir
                vocab_dir = os.path.join(base_path, ds_name, ds_size_name, lang_pair, "vocabs", "spm", subword_model, str(vocab_size))
                spm_model_path = os.path.join(vocab_dir, f"spm_{src_lang}-{trg_lang}.model")

                # Create encoded path
                encoded_path = os.path.join(base_path, ds_name, ds_size_name, lang_pair, "data", "encoded")
                path = Path(encoded_path)
                path.mkdir(parents=True, exist_ok=True)

                # Encode files
                new_filename_dir = os.path.join(base_path, ds_name, ds_size_name, lang_pair, "data", "encoded", subword_model, str(vocab_size))
                raw_folder = "pretokenized" if (encoding_mode == "pretokenized" or subword_model == "word") else "splits"
                for fname in trans_files:
                    ori_filename = os.path.join(base_path, ds_name, ds_size_name, lang_pair, "data", raw_folder, fname)
                    new_filename = os.path.join(new_filename_dir, fname)

                    # Create new path
                    path = Path(new_filename_dir)
                    path.mkdir(parents=True, exist_ok=True)

                    # # Encode files
                    # if not force_overwrite and os.path.exists(new_filename):
                    #     print(f"\t\t=> Skipping encoded file as it already exists: ({fname})")
                    # else:
                    #     commands.spm_encode(spm_model_path=spm_model_path, input_file=ori_filename, output_file=new_filename)

                # Export vocab frequencies
                if export_frequencies:
                    export_vocab_frequencies(encoded_dir=new_filename_dir, vocab_dir=vocab_dir, src_lang=src_lang, trg_lang=trg_lang, force_overwrite=force_overwrite)


def export_vocab_frequencies(encoded_dir, vocab_dir, src_lang, trg_lang, force_overwrite):
    vocab_path = os.path.join(vocab_dir, f"spm_{src_lang}-{trg_lang}.vocab")
    vocab_freq_path = os.path.join(vocab_dir, f"spm_{src_lang}-{trg_lang}.vocabf")

    if not force_overwrite and os.path.exists(vocab_freq_path):
        print(f"\t\t=> Skipping exporting vocab frequencies as the file already exists: ({vocab_freq_path})")
    else:
        print(f"Exporting vocab frequencies...")
        # Load vocab
        vocabs = {l.strip().split('\t')[0] for l in open(vocab_path, 'r').readlines()}

        # Count tokens
        vocab_frequencies = defaultdict(int)
        for fname in [f"train.{src_lang}", f"train.{trg_lang}"]:
            with open(os.path.join(encoded_dir, fname), 'r') as f:
                for line in tqdm(f):
                    tokens = line.strip().split(' ')
                    for tok in tokens:
                        if tok in vocabs:  # Count only the tokens that exists in the vocab
                            vocab_frequencies[tok] += 1

        # Save file
        vocab_frequencies = sorted(list(vocab_frequencies.items()), key=lambda x: x[1], reverse=True)  # Descending order
        with open(vocab_freq_path, 'w') as f:
            f.writelines([f"{pair[0]}\t{pair[1]}\n" for pair in vocab_frequencies])


def main(base_path, datasets, encoding_mode, subword_models, vocab_sizes, min_vocab_frequency, force_overwrite, split_raw_data=False):
    # Checks
    if encoding_mode not in {"pretokenized", "encoded", "splits"}:
        raise ValueError(f"'encoded_mode' not valid.")
    if encoding_mode not in {"pretokenized", "encoded", "splits"}:
        raise ValueError(f"'encoded_mode' not valid.")

    # Split raw data
    if split_raw_data:
        create_splits(base_path, datasets, val_size=5000, test_size=5000, shuffle=True, force_overwrite=force_overwrite)

    # Create reduced versions
    create_reduced_versions(base_path, datasets, autofix=True)

    # # Pretokenize (sacremoses)
    # if encoding_mode == "pretokenized" or "word" in set(subword_models):
    #     pretokenize(base_path, datasets, force_overwrite=force_overwrite)

    # Build vocabs
    for sw_model in subword_models:  # unigram, bpe, char, or word
        for voc_size in vocab_sizes:
            # # Build vocab
            # build_vocab(base_path=base_path, datasets=datasets, encoding_mode=encoding_mode,
            #             subword_model=sw_model, vocab_size=voc_size, character_coverage=1.0, merge_trains=True,
            #             force_overwrite=force_overwrite)

            # Encode datasets
            if encoding_mode == "encoded":
                encode_datasets(base_path=base_path, datasets=datasets, encoding_mode=encoding_mode,
                                subword_model=sw_model, vocab_size=voc_size, min_vocab_frequency=min_vocab_frequency,
                                export_frequencies=True, force_overwrite=force_overwrite)


if __name__ == "__main__":
    # Get base path
    if os.environ.get("LOCAL_GPU"):
        BASE_PATH = "/home/salva/Documents/datasets/nn/translation"
    else:
        BASE_PATH = "/home/scarrion/datasets/nn/translation"

    ENCODING_MODE = "encoded"  # splits (raw), encoded (spm), [pretokenized (moses) => Force moses tokenization for everything]
    SUBWORD_MODELS = ["word", "char", "unigram"]  # unigram, bpe, char, or word
    VOCAB_SIZE = [16000]
    MIN_VOCAB_FREQUENCY = 1  # Doesn't work
    FORCE_OVERWRITE = True

    DATASETS = [
        {"name": "multi30k", "sizes": [("original", None)], "languages": ["de-en"]},
        # {"name": "europarl", "sizes": [("50k", 50000)], "languages": ["es-en"]},

        # {"name": "ccaligned", "sizes": [("original", None)], "languages": ["or-en", "ti-en"]},
        # {"name": "wikimatrix", "sizes": [("original", None)], "languages": ["ar-en", "ja-en", "ko-en"]},

        # {"name": "commoncrawl", "sizes": [("original", None), ("100k", 100000), ("50k", 50000)], "languages": ["es-en"]},
        # {"name": "europarl", "sizes": [("original", None), ("100k", 100000), ("50k", 50000)], "languages": ["cs-en", "de-en", "es-en", "fr-en"]},
        # {"name": "iwlst16", "sizes": [("original", None)], "languages": ["de-en"]},
        # {"name": "newscommentary", "sizes": [("original", None), ("100k", 100000), ("50k", 50000)], "languages": ["es-en"]},
        # {"name": "scielo/health", "sizes": [("original", None), ("100k", 100000), ("50k", 50000)], "languages": ["es-en", "pt-en"]},
        # {"name": "scielo/biological", "sizes": [("original", None), ("100k", 100000), ("50k", 50000)], "languages": ["es-en", "pt-en"]},
    ]

    # Create datasets
    main(base_path=BASE_PATH, datasets=DATASETS, encoding_mode=ENCODING_MODE, subword_models=SUBWORD_MODELS,
         vocab_sizes=VOCAB_SIZE, min_vocab_frequency=MIN_VOCAB_FREQUENCY, force_overwrite=FORCE_OVERWRITE)

    print("Done!")

