# AutoNMT


Library to build seq2seq models effortless with almost no-code.

**What can AutoNMT for me?**

- Provides a robust layout for working with NMT tasks (see layout below)
- Handles dataset preparation: 
  1) Create splits from raw files: `original/raw/data.[src,trg]` => `dataset/original/splits/[train, val, test].[src,trg]`
  2) Create versions of the original dataset: Size, tokenizer type (words, subwords, chars) and vocab size
  3) Pretokenize the splits: `original/data/pretokenized/[train, val, test].[src,trg]`
  4) Train subword models: `original/vocab/spm/unigram/16000/[spm_src-trg.[model, vocab, vocabf]]`
  5) Encode splits using the train tokenizer: `original/data/encoded/unigram/16000/[train, val, test].[src,trg]`
  6) Plot typical charts about the datasets
  7) Store stats
- Trains models for each of the given datasets
- Evaluates models for each of the given datasets
- Generates translations and postprocess them accordinly
- Score models using multiple metrics such as bleu, chrf, ter, bertscore, comet, beer,...
- Creates loggin a summarization files for you
- Provides toolkit abstraction


## Usage

### Building datasets

This code will create a total of `4*3*3*4=144` datasets (+subword models, vocabularies, plots and statistics)

```python
SUBWORD_MODELS = ["word", "unigram", "bpe", "char"] 
VOCAB_SIZE = [4000, 8000, 16000]
MAKE_PLOTS = False
FORCE_OVERWRITE = False

# Original dataset => ("original", None) // Reserved name
# New datasets => (dataset_name, size)
DATASETS = [
    {"name": "europarl", "sizes": [("original", None), ("100k", 100000), ("50k", 50000)], "languages": ["cs-en", "de-en", "es-en", "fr-en"]},
]

# Create datasets
build_datasets(base_path=BASE_PATH, datasets=DATASETS, subword_models=SUBWORD_MODELS,
               vocab_sizes=VOCAB_SIZE, make_plots=MAKE_PLOTS, force_overwrite=FORCE_OVERWRITE)
```

### Train & Evaluate models

```python
SUBWORD_MODELS = ["word", "unigram", "bpe", "char"]
VOCAB_SIZE = [4000, 8000, 16000]
FORCE_OVERWRITE = False  # Overwrite whatever that already exists
INTERACTIVE = True  # To interact with the shell if something already exists
NUM_GPUS = 'all'  # all, 1gpu=[0]; 2gpu=[0,1];...
TOOLKIT = "fairseq"  # or custom
BEAMS = [1, 5]
METRICS = {"bleu", "chrf", "ter", "bertscore", "comet"}
RUN_NAME = "mytransformer"

# Datasets for which to train a model
TRAIN_DATASETS = [
    {"name": "europarl", "sizes": [("original", None), ("100k", 100000), ("50k", 50000)], "languages": ["es-en", "pt-en"]},
    {"name": "scielo/health", "sizes": [("original", None), ("100k", 100000), ("50k", 50000)], "languages": ["es-en", "pt-en"]},
    {"name": "scielo/biological", "sizes": [("original", None), ("100k", 100000), ("50k", 50000)], "languages": ["es-en", "pt-en"]},
]

# Datasets in which evaluate the different models
EVAL_DATASETS = [
    {"name": "scielo/biological", "sizes": [("original", None), ("100k", 100000), ("50k", 50000)], "languages": ["es-en", "pt-en"]},
    {"name": "scielo/health", "sizes": [("original", None), ("100k", 100000), ("50k", 50000)], "languages": ["es-en", "pt-en"]},
]

# Train and Score
train_and_score(base_path=BASE_PATH, train_datasets=TRAIN_DATASETS, eval_datasets=EVAL_DATASETS, run_name=RUN_NAME,
                subword_models=SUBWORD_MODELS, vocab_size=VOCAB_SIZE,
                force_overwrite=FORCE_OVERWRITE, interactive=INTERACTIVE,
                toolkit=TOOLKIT, num_gpus=NUM_GPUS, beams=BEAMS, metrics=METRICS)
```


## Layout for NLP tasks

Example of the layout a dataset with multiple size versions, tokenizations, vocabulary sizes, models and evaluations.
It is automatically handle!

```text
- dataset/ => name/domain*/version*
    - original/
    - 100k/
    - 50k/
        - es-en/
        - ti-en/
            - data/
                - raw/  => data.es, data.en
                - splits/  => train.[es,en], val.[es,en], test.[es,en]
                - pretokenized/  => moses tokenization
                - encoded/  => subword model applied (if model_type=words, the "encoded/" won't be the same as the "pretokenized/" unless the min_word_frequency is 1)
                    - unigram/
                    - word/
                        - 1000/
                        - 8000/  => train.[es,en], val.[es,en], test.[es,en]
            - vocabs/
                - bpe/
                - spm/
                    - char/
                    - word/
                    - unigram/
                        - 4000/ => vocab size  (not the same as the number of merge operations)
                        - 16000/
                            - spm_es-en.[model, vocab]
            - models/
                - custom/
                - opennmt/
                - fairseq/
                    - data-bin/
                        - unigram/
                        - word/
                            - 1000/
                            - 8000/
                    - runs/
                        - mymodel1/
                        - mymodel2/
                            - logs/
                            - checkpoints/
                                - checkpoint_best.pt
                                - checkpoint_last.pt
                            - eval/
                                - dataset1/
                                - dataset2/
                                    - beams/
                                        - beam_1/
                                        - beam_5/
                                            - hyp.[tok,txt]
                                            - ref.[tok,txt]
                                            - scores/
                                                - sacrebleu_scores.json
                                                - bert_scores.txt
                                                - comet_scores.txt

```

## Installation

**Main environment:**

```
conda create --name mltests python=3.8
conda activate mltests
pip install -r requirements.txt
```

**Fairseq environment:**

> Note: Fairseq has a lot of incompatibilities and must be run separately

```
conda create --name fairseq python=3.8
conda activate fairseq
pip install -r requirements-fairseq.txt
```

**Remove environments:**

```
conda remove --name mltests --all
conda remove --name fairseq --all
```

## Fix problems on ssh

```
Git conflict:
Please move or remove them before you merge.
Aborting

Fix:
git clean -d -f
git reset --hard
git pull
```
