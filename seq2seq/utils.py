import time
import json
import math
import pickle
import re

import torch
from torchtext import data
from torchtext.data.metrics import bleu_score

from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
# import seaborn as sns
# sns.set() Problems with attention

from nltk.tokenize.treebank import TreebankWordDetokenizer


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def gpu_info():
    if torch.cuda.is_available():
        s = f"- Using GPU: {torch.cuda.is_available()}\n" \
            f"- No. devices: {torch.cuda.device_count()}\n" \
            f"- Device name (0): {torch.cuda.get_device_name(0)}"
    else:
        s = "- Using CPU"
    return s


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def display_attention(sentence, translation, attention, savepath=None, title="Attention"):
    fig = plt.figure(figsize=(10,10), dpi=100)
    ax = fig.add_subplot(111)

    cax = ax.matshow(attention, cmap='bone')

    ax.tick_params(labelsize=15)
    ax.set_xticklabels([''] + sentence, rotation=45)
    ax.set_yticklabels([''] + translation)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    ax.set_title(title)
    ax.set_xlabel("Source")
    ax.set_ylabel("Translation")

    # Save figure
    if savepath:
        plt.savefig(savepath)
        print("save attention!")

    plt.show()
    plt.close()


def save_dataset_examples(dataset, savepath):
    start = time.time()

    total = len(dataset.examples)
    with open(savepath, 'w') as f:
        # Save num. elements
        f.write(json.dumps(total))
        f.write("\n")

        # Save elements
        for pair in tqdm(dataset.examples, total=total):
            data = [pair.src, pair.trg]
            f.write(json.dumps(data))
            f.write("\n")

    end = time.time()
    print(f"Save dataset examples: [Total time= {end - start}; Num. examples={total}]")


def load_dataset(filename, fields, ratio=1.0):
    start = time.time()

    examples = []
    with open(filename, 'rb') as f:
        # Read num. elements
        line = f.readline()
        total = json.loads(line)

        # Load elements
        limit = int(total * ratio)
        for i in tqdm(range(limit), total=limit):
            line = f.readline()
            example = json.loads(line)
            example = data.Example().fromlist(example, fields)  # Create Example obj.
            examples.append(example)

    # Build dataset ("examples" passed by reference)
    dataset = data.Dataset(examples, fields)

    end = time.time()
    print(f"Load dataset: [Total time= {end - start}; Num. examples={len(dataset.examples)}]")
    return dataset


def load_vocabulary(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def save_vocabulary(field, savepath):
    with open(savepath, 'wb') as f:
        pickle.dump(field.vocab, f)


def calculate_bleu(model, data_iter, max_trg_len, beam_width, packed_pad=False):
    trgs = []
    trg_pred = []

    model.eval()
    data_iter.batch_size = 1

    for batch in tqdm(data_iter, total=len(data_iter)):
        # Get output
        if packed_pad:
            (src, src_len), (trg, trg_len) = batch.src, batch.trg
            trg_indexes, _ = model.translate_sentence(src, src_len, max_trg_len)
        else:  # RNN, Transformers
            src, trg = batch.src, batch.trg
            trg_indexes = model.translate_sentence(src, max_trg_len, beam_width)

        # Get best
        trg_indexes_best, trg_prob_best, _ = trg_indexes[0]  # Predictions must be sorted by probability
        trg_indexes = trg_indexes_best

        # Convert predicted indices to tokens
        trg_pred_tokens = [model.trg_field.vocab.itos[i] for i in trg_indexes]
        trg_tokens = [model.trg_field.vocab.itos[i] for i in trg.detach().cpu().int().flatten()]

        # Remove special tokens
        trg_pred_tokens = trg_pred_tokens[1:-1]  # Remove <sos> and <eos>
        trg_tokens = trg_tokens[1:-1]  # Remove <sos> and <eos>

        # Add predicted token
        trg_pred.append(trg_pred_tokens)
        trgs.append([trg_tokens])

    # Compute score
    score = bleu_score(trg_pred, trgs)
    return score


def detokenize(text):
    text = [x for x in text if x not in {"<unk>", "<pad>", "<sos>", "<eos>"}]
    tbwd = TreebankWordDetokenizer()
    text = tbwd.detokenize(text)
    return text


def truecasing(text, nlp):
    # Autocapitalize
    doc = nlp(text)
    tagged_sent = [(w.text, w.tag_) for w in doc]
    normalized_sent = [w.capitalize() if t in ["NN", "NNS"] else w for (w, t) in tagged_sent]
    normalized_sent[0] = normalized_sent[0].capitalize()
    text = re.sub(r" (?=[\.,'!?:;])", "", ' '.join(normalized_sent))
    return text


def postprocess(text, nlp=None):
    # Detokenize
    text = detokenize(text)

    # Basic post-processing
    text = text.lower().strip()

    # Autocapitalize
    if nlp:
        text = truecasing(text, nlp)
    return text


def show_translation_pair(src_tokens, trans_tokens, nlp_src=None, nlp_trg=None):
    # Source
    src = postprocess(src_tokens, nlp_src)
    print(f"- Original: '{src}'")

    # Target
    trgs = []
    for i, trans in enumerate(trans_tokens):
        trg = postprocess(trans, nlp_trg)
        trgs.append(trgs)
        print(f"\t=> Translation #{i+1}: '{trg}'")

    return src, trgs
