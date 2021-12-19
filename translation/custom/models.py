import os

import torch
import torch.nn as nn
import torch.utils.data as tud
import tqdm

from translation.custom.dataset import TranslationDataset


class Seq2Seq(nn.Module):

    def print_architecture(self):
        for k in self.architecture.keys():
            print(f"{k.replace('_', ' ').capitalize()}: {self.architecture[k]}")
        print(f"Trainable parameters: {sum([p.numel() for p in self.parameters()]):,}")
        print()

    def fit(self, train_ds, val_ds, batch_size=128, max_tokens=None, max_epochs=5, learning_rate=1e-3, weight_decay=0,
            clip_norm=1.0, patience=10, criterion="cross_entropy", optimizer="adam", checkpoints_path=None,
            logs_path=None):
        device = next(self.parameters()).device  # Get device

        # Create dataloaders
        collate_fn = lambda x: train_ds.collate_fn(x, max_tokens=max_tokens)
        train_dataloader = tud.DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

        # Criterion and Optimizer
        assert train_ds.src_vocab.pad_id == train_ds.src_vocab.pad_id
        pad_idx = train_ds.src_vocab.pad_id
        criterion_fn = nn.CrossEntropyLoss(ignore_index=pad_idx)
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Train loop
        best_loss = float(torch.inf)
        for i in range(max_epochs):
            self.train()
            print(f"Epoch #{i+1}:")
            train_losses, train_errors, train_sizes = [], [], []
            for x, y in tqdm.tqdm(train_dataloader, total=len(train_dataloader)):
                # Move to device
                x, y = x.to(device), y.to(device)

                # Forward
                probs = self.forward(x, y).transpose(1, 2)[:, :, :-1]
                y = y[:, 1:]

                # Backward
                loss = criterion_fn(probs, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=clip_norm)  # Clip norm
                optimizer.step()
                optimizer.zero_grad()

                # compute accuracy
                predictions = probs.argmax(1)
                batch_errors = (predictions != y)

                # To keep track
                train_errors.append(batch_errors.sum().item())
                train_sizes.append(batch_errors.numel())

                # Keep results
                train_losses.append(loss.item())

            # Compute metrics
            train_loss = sum(train_losses) / len(train_losses)
            train_acc = 1 - sum(train_errors) / sum(train_sizes)
            print("\t- train_loss={:.3f}; train_acc={:.3f}".format(train_loss, train_acc))

            # Validation
            val_loss, val_acc = self.evaluate(val_ds, batch_size=batch_size, max_tokens=max_tokens,
                                              criterion=criterion, device=device, prefix="val")

            # Save model
            if checkpoints_path is not None:
                print("\t- Saving checkpoint...")
                torch.save(self.state_dict(), os.path.join(checkpoints_path, "checkpoint_last.pt"))

                # Save best
                if val_loss < best_loss:
                    best_loss = val_loss
                    torch.save(self.state_dict(), os.path.join(checkpoints_path, "checkpoint_best.pt"))

            # # Validation with beam search
            # predictions, log_probabilities = seq2seq.beam_search(model, X_new)
            # output = [target_index.tensor2text(p) for p in predictions]

    def evaluate(self, eval_ds, batch_size=128, max_tokens=None, criterion="cross_entropy", prefix="eval"):
        self.eval()
        device = next(self.parameters()).device  # Get device

        # Create dataloader
        collate_fn = lambda x: eval_ds.collate_fn(x, max_tokens=max_tokens)
        eval_dataloader = tud.DataLoader(eval_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        # Criterion and Optimizer
        assert eval_ds.src_vocab.pad_id == eval_ds.src_vocab.pad_id
        pad_idx = eval_ds.src_vocab.pad_id
        criterion_fn = nn.CrossEntropyLoss(ignore_index=pad_idx)

        with torch.no_grad():
            eval_losses, eval_errors, eval_sizes = [], [], []
            for x, y in tqdm.tqdm(eval_dataloader, total=len(eval_dataloader)):
                # Move to device
                x, y = x.to(device), y.to(device)

                # Forward
                probs = self.forward(x, y).transpose(1, 2)[:, :, :-1]
                y = y[:, 1:]

                # Get loss
                loss = criterion_fn(probs, y)

                # compute accuracy
                predictions = probs.argmax(1)
                batch_errors = (predictions != y)

                # To keep track
                eval_errors.append(batch_errors.sum().item())
                eval_sizes.append(batch_errors.numel())

                # Keep results
                eval_losses.append(loss.item())

            # Compute metrics
            eval_loss = sum(eval_losses) / len(eval_losses)
            eval_acc = 1 - sum(eval_errors) / sum(eval_sizes)
            print("\t- {}_loss={:.3f}; {}_acc={:.3f}".format(prefix, eval_loss, prefix, eval_acc))
        return eval_loss, eval_acc


class Transformer(Seq2Seq):
    def __init__(self,
                 src_vocab_size, trg_vocab_size,
                 encoder_embed_dim=256,
                 decoder_embed_dim=256,
                 encoder_layers=3,
                 decoder_layers=3,
                 encoder_attention_heads=8,
                 decoder_attention_heads=8,
                 encoder_ffn_embed_dim=512,
                 decoder_ffn_embed_dim=512,
                 dropout=0.1,
                 activation_fn="relu",
                 max_sequence_length=256):
        super().__init__()
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size
        self.max_sequence_length = max_sequence_length

        # Model
        self.src_embeddings = nn.Embedding(src_vocab_size, encoder_embed_dim)
        self.trg_embeddings = nn.Embedding(trg_vocab_size, encoder_embed_dim)
        self.pos_embeddings = nn.Embedding(max_sequence_length, encoder_embed_dim)
        self.transformer = nn.Transformer(d_model=encoder_embed_dim,
                                          nhead=encoder_attention_heads,
                                          num_encoder_layers=encoder_layers,
                                          num_decoder_layers=decoder_layers,
                                          dim_feedforward=encoder_ffn_embed_dim,
                                          dropout=dropout,
                                          activation=activation_fn)
        self.output_layer = nn.Linear(encoder_embed_dim, src_vocab_size)

        # Checks
        assert encoder_embed_dim == decoder_embed_dim
        assert encoder_attention_heads == decoder_attention_heads
        assert encoder_ffn_embed_dim == decoder_ffn_embed_dim

    def forward(self, X, Y):
        assert X.shape[1] <= self.max_sequence_length
        assert Y.shape[1] <= self.max_sequence_length

        # Encode src
        X = self.src_embeddings(X)
        X_positional = torch.arange(X.shape[1], device=X.device).repeat((X.shape[0], 1))
        X_positional = self.pos_embeddings(X_positional)
        X = (X + X_positional).transpose(0, 1)

        # Encode trg
        Y = self.trg_embeddings(Y)
        Y_positional = torch.arange(Y.shape[1], device=Y.device).repeat((Y.shape[0], 1))
        Y_positional = self.pos_embeddings(Y_positional)
        Y = (Y + Y_positional).transpose(0, 1)

        # Make trg mask
        mask = self.transformer.generate_square_subsequent_mask(Y.shape[0]).to(Y.device)

        # Forward model
        output = self.transformer.forward(src=X, tgt=Y, tgt_mask=mask)

        # Get output
        output = output.transpose(0, 1)
        output = self.output_layer(output)
        return output
