# mltests

Deep learning and stuff


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

## Layout for NLP tasks


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