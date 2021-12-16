import json
import os
from pathlib import Path
from itertools import islice
from collections import defaultdict
import random

import pandas as pd
import tqdm

import numpy as np

random.seed(123)

from translation.preprocess import utils, plots
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
                lines = utils.preprocess_pairs(src_lines, trg_lines, shuffle=shuffle)

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
                trans_files = utils.get_translation_files(src_lang, trg_lang)

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
                trans_files = utils.get_translation_files(src_lang, trg_lang)

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
                trans_files = utils.get_translation_files(src_lang, trg_lang)

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
                trans_files = utils.get_translation_files(src_lang, trg_lang)

                # Get vocab dir
                vocab_dir = os.path.join(base_path, ds_name, ds_size_name, lang_pair, "vocabs", "spm", subword_model, str(vocab_size))
                spm_model_path = os.path.join(vocab_dir, f"spm_{src_lang}-{trg_lang}.model")

                # Create encoded path
                encoded_path = os.path.join(base_path, ds_name, ds_size_name, lang_pair, "data", "encoded")
                path = Path(encoded_path)
                path.mkdir(parents=True, exist_ok=True)

                # Encode files
                print(f"\t=> Encoding: {encoded_path}")
                new_filename_dir = os.path.join(base_path, ds_name, ds_size_name, lang_pair, "data", "encoded", subword_model, str(vocab_size))
                raw_folder = "pretokenized" if (encoding_mode == "pretokenized" or subword_model == "word") else "splits"
                for fname in trans_files:
                    ori_filename = os.path.join(base_path, ds_name, ds_size_name, lang_pair, "data", raw_folder, fname)
                    new_filename = os.path.join(new_filename_dir, fname)

                    # Create new path
                    path = Path(new_filename_dir)
                    path.mkdir(parents=True, exist_ok=True)

                    # Encode files
                    if not force_overwrite and os.path.exists(new_filename):
                        print(f"\t\t=> Skipping encoded file as it already exists: ({fname})")
                    else:
                        commands.spm_encode(spm_model_path=spm_model_path, input_file=ori_filename, output_file=new_filename)

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
                for line in tqdm.tqdm(f):
                    tokens = line.strip().split(' ')
                    for tok in tokens:
                        if tok in vocabs:  # Count only the tokens that exists in the vocab
                            vocab_frequencies[tok] += 1

        # Save file
        vocab_frequencies = sorted(list(vocab_frequencies.items()), key=lambda x: x[1], reverse=True)  # Descending order
        with open(vocab_freq_path, 'w') as f:
            f.writelines([f"{pair[0]}\t{pair[1]}\n" for pair in vocab_frequencies])


def plot_datasets(base_path, datasets, subword_model, vocab_size, force_overwrite):
    print(f"- Plotting datasets...")
    print(f"- [WARNING]: Matplotlib might miss some images if the loop is too fast")

    SAVE_FIGURES=True
    SHOW_FIGURES=False

    for ds in datasets:  # Dataset
        ds_name = ds["name"]
        for ds_size_name, ds_max_lines in ds["sizes"]:  # Lengths
            for lang_pair in ds["languages"]:  # Languages
                src_lang, trg_lang = lang_pair.split("-")
                trans_files = utils.get_translation_files(src_lang, trg_lang)

                # Set base path
                base_fname = f"{ds_name}_{ds_size_name}_{lang_pair}__{subword_model}_{str(vocab_size)}"
                print(f"\t=> Creating plots for: {base_fname}")

                # Get dirs
                vocab_dir = os.path.join(base_path, ds_name, ds_size_name, lang_pair, "vocabs", "spm", subword_model, str(vocab_size))
                encoded_path = os.path.join(base_path, ds_name, ds_size_name, lang_pair, "data", "encoded", subword_model, str(vocab_size))

                # Create plot paths
                plots_vocabs_path = os.path.join(base_path, ds_name, ds_size_name, lang_pair, "plots", "vocabs", subword_model, str(vocab_size))
                plots_encoded_path = os.path.join(base_path, ds_name, ds_size_name, lang_pair, "plots", "data", "encoded", subword_model, str(vocab_size))
                for p in [plots_vocabs_path, plots_encoded_path]:
                    path = Path(p)
                    path.mkdir(parents=True, exist_ok=True)

                print(f"\t\t=> Creating 'Sentence length distribution' plots...")
                split_stats = {}
                for fname in trans_files:
                    split_name, split_lang = fname.split('.')
                    tokens_by_sentence = np.array(plots.get_tokens_by_sentence(filename=os.path.join(encoded_path, fname)))

                    row = {
                        "total_sentences": len(tokens_by_sentence),
                        "total_tokens": int(tokens_by_sentence.sum()),
                        "max_tokens": int(np.max(tokens_by_sentence)),
                        "min_tokens": int(np.min(tokens_by_sentence)),
                        "avg_tokens": float(np.average(tokens_by_sentence)),
                        "std_tokens": float(np.std(tokens_by_sentence)),
                        "percentile5_tokens": int(np.percentile(tokens_by_sentence, 5)),
                        "percentile50_tokens": int(np.percentile(tokens_by_sentence, 50)),
                        "percentile95_tokens": int(np.percentile(tokens_by_sentence, 95)),
                        "split": split_name.title(),
                        "lang": split_lang,
                        "alias": f"{split_name.title()} ({split_lang})",
                    }
                    split_stats[fname] = row

                    # Plot sentence length distribution (by tokens' length): 3x2
                    df = pd.DataFrame(tokens_by_sentence, columns=["frequency"])
                    title = f"Sentence length distribution"
                    p_fname = f"sent_distr_{split_name}_{split_lang}__{base_fname}"
                    plots.histogram(data=df, x="frequency", output_dir=plots_encoded_path, fname=p_fname,
                                    title=title, xlabel="Tokens per sentence", ylabel="Frequency", bins=100,
                                    aspect_ratio=(6, 4), size=1.5, save_fig=SAVE_FIGURES, show_fig=SHOW_FIGURES)

                # Save statistical data
                with open(os.path.join(plots_encoded_path, f"stats__{base_fname}.json"), 'w') as f:
                    json.dump(split_stats, f)

                # Get data
                df = pd.DataFrame(list(split_stats.values()))

                # Plot split size (by its sentence number): 1
                print(f"\t\t=> Creating 'Split sizes' plots...")
                title = f"Split sizes (by sentences)"
                p_fname = f"split_size_sent__{base_fname}"
                plots.catplot(data=df, x="split", y="total_sentences",  hue="lang",
                              title=title, xlabel="Dataset partitions", ylabel="Num. of sentences", leyend_title=None,
                              output_dir=plots_encoded_path, fname=p_fname,  aspect_ratio=(8, 4), size=1.0, save_fig=SAVE_FIGURES, show_fig=SHOW_FIGURES)

                # Plot split size (by token number): 1
                title = f"Split sizes (by tokens)"
                p_fname = f"split_size_tok__{base_fname}"
                plots.catplot(data=df, x="split", y="total_tokens",  hue="lang",
                              title=title, xlabel="Dataset partitions", ylabel="Num. of tokens", leyend_title=None,
                              output_dir=plots_encoded_path, fname=p_fname,  aspect_ratio=(8, 4), size=1.0, save_fig=SAVE_FIGURES, show_fig=SHOW_FIGURES)

                # Plot vocabulary frequency: 1
                print(f"\t\t=> Creating 'Vocabulary distribution' plots...")

                # Load vocabulary
                vocab_freq_path = os.path.join(vocab_dir, f"spm_{src_lang}-{trg_lang}.vocabf")
                with open(vocab_freq_path, 'r') as f:
                    rows = [line.split('\t') for line in f.readlines()]
                    df = pd.DataFrame(rows, columns=["token", "frequency"])
                    df["frequency"] = df["frequency"].astype(int)
                    df = df.sort_values(by='frequency', ascending=False, na_position='last')

                for top_k in [100, 150]:
                    title = f"Vocabulary distribution (top {str(top_k)})"
                    p_fname = f"vocab_distr_top{str(top_k)}__{base_fname}"
                    plots.barplot(data=df.head(top_k), x="token", y="frequency",
                                  output_dir=plots_vocabs_path, fname=p_fname,
                                  title=title, xlabel="Tokens", ylabel="Frequency",
                                  aspect_ratio=(12, 4), size=1.5, save_fig=SAVE_FIGURES, show_fig=SHOW_FIGURES)


def main(base_path, datasets, encoding_mode, subword_models, vocab_sizes, min_vocab_frequency, make_plots, force_overwrite, split_raw_data=False):
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

    # Pretokenize (sacremoses)
    if encoding_mode == "pretokenized" or "word" in set(subword_models):
        pretokenize(base_path, datasets, force_overwrite=force_overwrite)

    # Build vocabs
    for sw_model in subword_models:  # unigram, bpe, char, or word
        for voc_size in vocab_sizes:
            # Build vocab
            build_vocab(base_path=base_path, datasets=datasets, encoding_mode=encoding_mode,
                        subword_model=sw_model, vocab_size=voc_size, character_coverage=1.0, merge_trains=True,
                        force_overwrite=force_overwrite)

            # Encode datasets
            encode_datasets(base_path=base_path, datasets=datasets, encoding_mode=encoding_mode,
                            subword_model=sw_model, vocab_size=voc_size, min_vocab_frequency=min_vocab_frequency,
                            export_frequencies=True, force_overwrite=force_overwrite)

            # Plots
            if make_plots:
                plot_datasets(base_path=base_path, datasets=datasets, subword_model=sw_model, vocab_size=voc_size,
                              force_overwrite=force_overwrite)


if __name__ == "__main__":
    # Get base path
    if os.environ.get("LOCAL_GPU"):
        BASE_PATH = "/home/salva/Documents/datasets/nn/translation"
    else:
        BASE_PATH = "/home/scarrion/datasets/nn/translation"

    ENCODING_MODE = "encoded"  # splits (raw), encoded (spm), [pretokenized (moses) => Force moses tokenization for everything]
    SUBWORD_MODELS = ["word", "unigram", "char"]  # unigram, bpe, char, or word
    VOCAB_SIZE = [16000]
    MIN_VOCAB_FREQUENCY = 1  # Doesn't work
    MAKE_PLOTS = True
    FORCE_OVERWRITE = True

    DATASETS = [
        {"name": "multi30k", "sizes": [("original", None)], "languages": ["de-en"]},
        {"name": "europarl", "sizes": [("original", None), ("100k", 100000), ("50k", 50000)], "languages": ["es-en"]},

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
         vocab_sizes=VOCAB_SIZE, min_vocab_frequency=MIN_VOCAB_FREQUENCY, make_plots=MAKE_PLOTS,
         force_overwrite=FORCE_OVERWRITE)

    print("Done!")

