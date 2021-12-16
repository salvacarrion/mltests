import os
from pathlib import Path
import subprocess
import random
random.seed(123)

CONDA_ENVNAME = "mltests"


def score_test_files(data_path, trg_lang, metrics=None, src_lang=None, force_overwrite=True):
    if metrics is None:
        metrics = {'bleu'}  # {"bleu", "chrf", "ter", "bertscore", "comet"}

    # Check supported metrics
    metrics_diff = metrics.difference({"bleu", "chrf", "ter", "bertscore", "comet"})
    if metrics_diff:
        print(f"[WARNING] These metrics are not supported: {str(metrics_diff)}")
        if metrics == metrics_diff:
            print(f"- Skipping evaluation. No valid metrics were found")
            return

    # Create path (if needed)
    score_path = os.path.join(data_path, "scores")
    path = Path(score_path)
    path.mkdir(parents=True, exist_ok=True)

    # Test files (cleaned)
    src_file_path = os.path.join(data_path, "src.txt")
    ref_file_path = os.path.join(data_path, "ref.txt")
    hyp_file_path = os.path.join(data_path, "hyp.txt")

    # Sacrebleu
    sb_m = ""
    sb_m += "bleu " if "bleu" in metrics else ""
    sb_m += "chrf " if "chrf" in metrics else ""
    sb_m += "ter " if "ter" in metrics else ""
    if sb_m:
        print(f"\t=> Sacrebleu scoring...")
        sacrebleu_scores_path = os.path.join(score_path, "sacrebleu_scores.json")
        cmd = f"sacrebleu {ref_file_path} -i {hyp_file_path} -m {sb_m} -w 5 > {sacrebleu_scores_path}"  # bleu chrf ter
        subprocess.call(['/bin/bash', '-i', '-c', f"conda activate {CONDA_ENVNAME} && {cmd}"])

    # BertScore
    if 'bertscore' in metrics:
        print(f"\t=> BertScore scoring...")
        bertscore_scores_path = os.path.join(score_path, "bert_scores.txt")
        cmd = f"bert-score -r {ref_file_path} -c {hyp_file_path} --lang {trg_lang} > {bertscore_scores_path}"
        subprocess.call(['/bin/bash', '-i', '-c', f"conda activate {CONDA_ENVNAME} && {cmd}"])

    # Comet
    if 'comet' in metrics:
        print(f"\t=> Comet scoring...")
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
