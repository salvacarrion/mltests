# mltests
Deep learning and stuff


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