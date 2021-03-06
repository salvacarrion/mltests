import random
import time

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from torchtext import data
from torch.utils.tensorboard import SummaryWriter

from seq2seq.rnn_packed import helpers
from seq2seq import utils

###########################################################################
###########################################################################

# Names
EXPERIMENT_NAME = "runs/rnn_luong"
MODEL_NAME = "rnn_luong"

###########################################################################
###########################################################################

# Build model and initialize
DATASET_NAME = "miguel"  # multi30k, miguel
DATASET_PATH = f"../.data/{DATASET_NAME}"
TENSORBOARD = True
ALLOW_DATA_PARALLELISM = False
LEARNING_RATE = 1e-3
TEACHER_FORCING_RATIO = 0.5
MIN_FREQ = 3
MAX_SIZE = 10000 - 4  # 4 reserved words <sos>, <eos>, <pad>, <unk>
N_EPOCHS = 1000
MAX_SRC_LENGTH = 100 + 2  # Doesn't include <sos>, <eos>
MAX_TRG_LENGTH = 100 + 2  # Doesn't include <sos>, <eos>
BATCH_SIZE = 32
CHECKPOINT_PATH = f'checkpoints/checkpoint_{MODEL_NAME}.pt'
TR_RATIO = 0.1
DV_RATIO = 1.0
TB_BATCH_RATE = 100
SOS_WORD = '<sos>'
EOS_WORD = '<eos>'

###########################################################################
###########################################################################

print("###########################################################################")
print("###########################################################################")
print(f"- Mode: Training")
print(f"- Executing model: {EXPERIMENT_NAME}")
print(f"- Experiment name: {MODEL_NAME}")
print(f"- Checkpoint path: {CHECKPOINT_PATH}")
print("###########################################################################")
print("###########################################################################")

###########################################################################
###########################################################################

# Deterministic environment
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

###########################################################################
###########################################################################

# Set fields
SRC = data.Field(tokenize='spacy', tokenizer_language="en", init_token=SOS_WORD, eos_token=EOS_WORD, lower=True, include_lengths=True)
TRG = data.Field(tokenize='spacy', tokenizer_language="es", init_token=SOS_WORD, eos_token=EOS_WORD, lower=True, include_lengths=True)
fields = [('src', SRC), ('trg', TRG)]

# Load examples
train_data = utils.load_dataset(f"{DATASET_PATH}/tokenized/train.json", fields, TR_RATIO)
dev_data = utils.load_dataset(f"{DATASET_PATH}/tokenized/dev.json", fields, DV_RATIO)

print(f"Number of training examples: {len(train_data.examples)}")
print(f"Number of validation examples: {len(dev_data.examples)}")

start = time.time()

# Build vocab
SRC.build_vocab(train_data, max_size=MAX_SIZE)
TRG.build_vocab(train_data, max_size=MAX_SIZE)

end = time.time()
print(f"Time to build vocabularies: {end - start} sec")

print(f"Unique tokens in source (en) vocabulary: {len(SRC.vocab)}")
print(f"Unique tokens in target (es) vocabulary: {len(TRG.vocab)}")

###########################################################################
###########################################################################

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(utils.gpu_info())

# Set iterator (this is where words are replaced by indices, and <sos>/<eos> tokens are appended
train_iter, dev_iter = data.BucketIterator.splits((train_data, dev_data),
                                                  batch_size=BATCH_SIZE, device=device,
                                                  sort_within_batch=True, sort_key=helpers.sort_batch)

###########################################################################
###########################################################################

# Select model
if MODEL_NAME == "rnn_bahdanau":
    from seq2seq.models import s2s_4_bahdanau as builder

    model = builder.make_model(src_field=SRC, trg_field=TRG, device=device, data_parallelism=ALLOW_DATA_PARALLELISM)
    model.apply(builder.init_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

elif MODEL_NAME == "rnn_ba_lu_mixed":
    from seq2seq.models import s2s_4_ba_lu_mixed as builder

    model = builder.make_model(src_field=SRC, trg_field=TRG, device=device, data_parallelism=ALLOW_DATA_PARALLELISM)
    model.apply(builder.init_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

elif MODEL_NAME == "rnn_luong":
    from seq2seq.models import s2s_4_luong as builder

    model = builder.make_model(src_field=SRC, trg_field=TRG, device=device, data_parallelism=ALLOW_DATA_PARALLELISM)
    model.apply(builder.init_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

else:
    raise ValueError("Unknown model name")


###########################################################################
###########################################################################

# Set loss (ignore when the target token is <pad>)
TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)

###########################################################################
###########################################################################

# Tensorboard (it needs some epochs to start working ~10-20)
tr_writer = SummaryWriter(f"{EXPERIMENT_NAME}/train")
val_writer = SummaryWriter(f"{EXPERIMENT_NAME}/val")

# TB: Get graph
# dummy_batch = next(iter(train_iter))
# train_writer.add_graph(model, [dummy_batch.src, dummy_batch.trg[:, :-1]])

###########################################################################
###########################################################################

# Train and validate model
helpers.fit(model, train_iter, dev_iter=dev_iter,
            epochs=N_EPOCHS, optimizer=optimizer, criterion=criterion, checkpoint_path=CHECKPOINT_PATH,
            tr_writer=tr_writer, val_writer=val_writer, tb_batch_rate=TB_BATCH_RATE, tf_ratio=TEACHER_FORCING_RATIO)

###########################################################################
###########################################################################

print("Done!")
