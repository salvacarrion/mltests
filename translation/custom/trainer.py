import os
from pathlib import Path

import torch
from translation.custom.dataset import TranslationDataset
from translation.custom import models, search_algorithms


def train(data_path, src_lang, trg_lang, batch_size=128, max_tokens=None, max_epochs=10, learning_rate=1e-3,
          weight_decay=0, clip_norm=1.0, patience=10, criterion="cross_entropy", optimizer="adam",
          checkpoints_path=None, logs_path=None):

    # Create paths
    for p in [checkpoints_path, logs_path]:
        Path(p).mkdir(parents=True, exist_ok=True)

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device')

    # Create Datasets: Train, Val
    train_ds = TranslationDataset(data_path, src_lang=src_lang, trg_lang=trg_lang, split="train")
    val_ds = TranslationDataset(data_path, src_lang=src_lang, trg_lang=trg_lang, split="val")

    # Create and train model
    model = models.Transformer(src_vocab_size=len(train_ds.src_vocab), trg_vocab_size=len(train_ds.trg_vocab)).to(device)
    model.fit(train_ds, val_ds, batch_size=batch_size, max_tokens=max_tokens, max_epochs=max_epochs,
              learning_rate=learning_rate, weight_decay=weight_decay, clip_norm=clip_norm, patience=patience,
              criterion=criterion, optimizer=optimizer, checkpoints_path=checkpoints_path, logs_path=logs_path)

    # Evaluate test
    test_ds = TranslationDataset(data_path, src_lang=src_lang, trg_lang=trg_lang, split="test")
    test_loss, test_acc = model.evaluate(test_ds, batch_size=batch_size, max_tokens=max_tokens,
                                         criterion=criterion, prefix="test")


def translate(data_path, checkpoint_path, output_path, src_lang, trg_lang, batch_size=128, max_tokens=None,
              beam_width=5, max_length=200):
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device')

    # Create Datasets: Test
    test_ds = TranslationDataset(data_path, src_lang=src_lang, trg_lang=trg_lang, split="test")

    # Create model
    model = models.Transformer(src_vocab_size=len(test_ds.src_vocab), trg_vocab_size=len(test_ds.trg_vocab)).to(device)

    # Load model
    model_state_dict = torch.load(checkpoint_path)
    model.load_state_dict(model_state_dict)

    # Beam search
    predictions, log_probabilities = search_algorithms.greedy_search(model, ds=test_ds,
                                                                     sos_id=test_ds.trg_vocab.sos_id,
                                                                     eos_id=test_ds.trg_vocab.eos_id,
                                                                     batch_size=batch_size, max_tokens=max_tokens,
                                                                     beam_width=beam_width, max_length=max_length)

    # Create output path
    Path(output_path).mkdir(parents=True, exist_ok=True)

    # Decode sentences
    hyp_tok = [test_ds.trg_vocab.decode(tokens, remove_special_tokens=True) for tokens in predictions]

    # Write file: hyp
    for lines, fname in [(hyp_tok, "hyp.tok")]:
        with open(os.path.join(output_path, fname), 'w') as f:
            f.writelines([' '.join(tokens) + '\n' for tokens in lines])

    # Write files: src, ref
    for lines, fname in [(test_ds.src_lines, "src.tok"), (test_ds.trg_lines, "ref.tok")]:
        with open(os.path.join(output_path, fname), 'w') as f:
            f.writelines([line + '\n' for line in lines])
