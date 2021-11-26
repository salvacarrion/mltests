import os
from pathlib import Path
from itertools import islice


def get_translation_files(src_lang, trg_lang):
    files = []
    for split in ["train", "val", "test"]:
        for lang in [src_lang, trg_lang]:
            files.append(f"{split}.{lang}")
    return files


def check_datasets(base_path, datasets, autofix=False):
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
                        ds_level = os.path.join(ds_name, ds_size_name, lang_pair, "splits", trans_fname)
                        fname = os.path.join(base_path, ds_level)
                        if not os.path.exists(fname):
                            print(f"- Checking file '{ds_level}' => Failed")
                            continue

                else:  # Create folder
                    # Create folders (recursively)
                    ds_level = os.path.join(ds_name, ds_size_name, lang_pair, "splits")
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
                                ori_filename = os.path.join(base_path, ds_name, "original", lang_pair, "splits", trans_fname)
                                new_filename = os.path.join(base_path, ds_name, ds_size_name, lang_pair, "splits", trans_fname)

                                # Copy n lines efficiently
                                if not os.path.exists(new_filename):
                                    with open(ori_filename, 'r') as fin, open(new_filename, 'w') as fout:
                                        lines = list(islice(fin, ds_max_lines))
                                        fout.writelines(lines)
                                        print(f"- Autofix for '{ds_level}': Creating {trans_fname}...")

    return True


def tokenize_files(base_path, datasets):
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
                ds_level_path = os.path.join(base_path, ds_name, ds_size_name, lang_pair, "cleaned")
                path = Path(ds_level_path)
                path.mkdir(parents=True, exist_ok=True)

                # Process each file
                for trans_fname in trans_files:
                    ori_filename = os.path.join(base_path, ds_name, ds_size_name, lang_pair, "splits", trans_fname)
                    new_filename = os.path.join(base_path, ds_name, ds_size_name, lang_pair, "cleaned", trans_fname)
                    asd = 3

                    raise NotImplementedError()


def main(base_path, datasets):
    # Check and create splits (if needed)
    check_datasets(base_path, datasets, autofix=True)

    # Cleaned files
    tokenize_files(base_path, datasets)


if __name__ == "__main__":
    BASE_PATH = "/Users/salvacarrion/Documents/Programming/Datasets/nn/translation"
    DATASETS = [
        {"name": "commoncrawl", "sizes": [("original", None), ("100k", 100000), ("50k", 50000)], "languages": ["es-en"]},
        {"name": "europarl", "sizes": [("original", None), ("100k", 100000), ("50k", 50000)], "languages": ["cs-en", "de-en", "es-en", "fr-en"]},
        {"name": "iwlst16", "sizes": [("original", None)], "languages": ["de-en"]},
        {"name": "multi30k", "sizes": [("original", None)], "languages": ["de-en"]},
        {"name": "newscommentary", "sizes": [("original", None), ("100k", 100000), ("50k", 50000)], "languages": ["es-en"]},
        {"name": "scielo/health", "sizes": [("original", None), ("100k", 100000), ("50k", 50000)], "languages": ["es-en", "pt-en"]},
        {"name": "scielo/biological", "sizes": [("original", None), ("100k", 100000), ("50k", 50000)], "languages": ["es-en", "pt-en"]},
    ]

    # Create datasets
    main(base_path=BASE_PATH, datasets=DATASETS)
    print("Done!")
