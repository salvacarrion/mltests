import os
import subprocess

if os.environ.get('MACHINE') == "HOME":
    print("Local")
    BASE_PATH = "/home/salvacarrion/Documents/Programming/Datasets/Scielo/fairseq"
    FAST_PATH = "/home/salvacarrion/Documents/packages/fastBPE/fast"
else:
    print("Remote")
    BASE_PATH = "/home/scarrion/datasets/scielo/fairseq"
    FAST_PATH = "/home/scarrion/packages/fastBPE/fast"

for SRC_LANG, TRG_LANG in [("es", "en"), ("pt", "en")]:
    for domain in ["health", "biological", "merged"]:
        dataset = f"scielo_{domain}_{SRC_LANG}_{TRG_LANG}"
        path = os.path.join(BASE_PATH, dataset)
        subprocess.call(['sh', './scripts/1_tokenize.sh', SRC_LANG, TRG_LANG, path])
