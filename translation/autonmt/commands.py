import os
from pathlib import Path
import subprocess
import random
random.seed(123)

CONDA_ENVNAME = "mltests"


def score_test_files(data_path, src_lang, trg_lang, force_overwrite=True, bleu=True, chrf=True, ter=True, bertscore=True, comet=True):
    # Create path (if needed)
    score_path = os.path.join(data_path, "scores")
    path = Path(score_path)
    path.mkdir(parents=True, exist_ok=True)

    # Test files (cleaned)
    src_file_path = os.path.join(data_path, "src.txt")
    ref_file_path = os.path.join(data_path, "ref.txt")
    hyp_file_path = os.path.join(data_path, "hyp.txt")

    # Sacrebleu
    metrics = ""
    metrics += "bleu " if bleu else ""
    metrics += "chrf " if chrf else ""
    metrics += "ter " if ter else ""
    if metrics:
        print(f"=> Scoring with Sacrebleu...")
        sacrebleu_scores_path = os.path.join(score_path, "sacrebleu_scores.json")
        cmd = f"sacrebleu {ref_file_path} -i {hyp_file_path} -m {metrics} -w 5 > {sacrebleu_scores_path}"  # bleu chrf ter
        subprocess.call(['/bin/bash', '-i', '-c', f"conda activate {CONDA_ENVNAME} && {cmd}"])

    # BertScore
    if bertscore:
        print(f"=> Scoring with BertScore...")
        bertscore_scores_path = os.path.join(score_path, "bert_scores.txt")
        cmd = f"bert-score -r {ref_file_path} -c {hyp_file_path} --lang {trg_lang} > {bertscore_scores_path}"
        subprocess.call(['/bin/bash', '-i', '-c', f"conda activate {CONDA_ENVNAME} && {cmd}"])

    # Comet
    if comet:
        print(f"=> Scoring with Comet...")
        comet_scores_path = os.path.join(score_path, "comet_scores.txt")
        cmd = f"comet-score -s {src_file_path} -t {hyp_file_path} -r {ref_file_path} > {comet_scores_path}"
        subprocess.call(['/bin/bash', '-i', '-c', f"conda activate {CONDA_ENVNAME} && {cmd}"])


def spm_encode(spm_model_path, input_file, output_file):
    cmd = f"spm_encode --model={spm_model_path} --output_format=piece < {input_file} > {output_file}"  # --vocabulary={spm_model_path}.vocab --vocabulary_threshold={min_vocab_frequency}
    subprocess.call(['/bin/bash', '-i', '-c', f"conda activate {CONDA_ENVNAME} && {cmd}"])


def spm_decode(spm_model_path, input_file, output_file):
    cmd = f"spm_decode --model={spm_model_path} --input_format=piece < {input_file} > {output_file}"
    subprocess.call(['/bin/bash', '-i', '-c', f"conda activate {CONDA_ENVNAME} && {cmd}"])


def spm_train(input_file, model_prefix, vocab_size, character_coverage, subword_model, input_sentence_size=1000000):
    # https://github.com/google/sentencepiece/blob/master/doc/options.md
    cmd = f"spm_train --input={input_file} --model_prefix={model_prefix} --vocab_size={vocab_size} --character_coverage={character_coverage} --model_type={subword_model} --input_sentence_size={input_sentence_size} --pad_id=3"
    subprocess.call(['/bin/bash', '-i', '-c', f"conda activate {CONDA_ENVNAME} && {cmd}"])


def moses_tokenizer(lang, input_file, output_file):
    cmd = f"sacremoses -l {lang} -j$(nproc) tokenize < {input_file} > {output_file}"
    subprocess.call(['/bin/bash', '-i', '-c', f"conda activate {CONDA_ENVNAME} && {cmd}"])


def moses_detokenizer(lang, input_file, output_file):
    cmd = f"sacremoses -l {lang} -j$(nproc) detokenize < {input_file} > {output_file}"
    subprocess.call(['/bin/bash', '-i', '-c', f"conda activate {CONDA_ENVNAME} && {cmd}"])


def replace_in_file(search_string, replace_string, filename):
    cmd = f"sed -i 's/{search_string}/{replace_string}/' {filename}"
    subprocess.call(['/bin/bash', '-i', '-c', f"{cmd}"])
