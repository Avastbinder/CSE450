import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import os
import pickle
from torch.utils.data import random_split


class CharLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1):
        super(CharLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )

        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        embedded = self.embedding(x)

        if hidden is None:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        else:
            h0, c0 = hidden

        out, (hn, cn) = self.lstm(embedded, (h0, c0))
        out = self.fc(out)
        return out, (hn, cn)

    def init_hidden(self, batch_size, device):
        return (
            torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device),
            torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        )

    def train_model(self, dataset, val_loader, num_epochs=10, learning_rate=0.002, device='cpu', print_every=100, initial_seq_length=20, max_seq_length=200, step_size=20, batch_size=32, patience=3, char_to_idx=None, idx_to_char=None):

        self.to(device)
        self.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        best_loss = float('inf')
        patience = 3
        counter = 0

        for epoch in range(1, num_epochs + 1):
            current_seq_length = min(initial_seq_length + (epoch - 1) * step_size, max_seq_length)
            print(f"\nEpoch {epoch} | Sequence Length: {current_seq_length}")

            dataset.seq_length = current_seq_length

            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            total_loss = 0
            hidden = None

            for batch_idx, (x, y) in tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch}"):
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                if hidden is not None:
                    hidden = tuple(h.detach() for h in hidden)
                    if hidden[0].size(1) != x.size(0):
                        hidden = self.init_hidden(x.size(0), device)

                output, hidden = self(x, hidden)

                loss = criterion(output.view(-1, self.vocab_size), y.view(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5)
                optimizer.step()

                total_loss += loss.item()

            total_samples = len(dataloader)

            if total_loss > 0:
                avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
                perplexity = np.exp(avg_loss)
                print(f"Training Loss: {avg_loss:.4f} | Perplexity: {perplexity:.2f}")

            val_loss = self.evaluate(val_loader, device)
            if val_loss < best_loss:
                best_loss = val_loss
                counter = 0
                self.save_model("checkpoints/best_model", char_to_idx, idx_to_char)
            else:
                counter += 1
                print(f"Early stopping counter: {counter}/{patience}")
                if counter >= patience:
                    print("Early stopping triggered.")
                    break

            generated = self.generate_text("Let me tell you why  ", char_to_idx, idx_to_char, length=300, temperature=0.8, device=device)
            print(f"\nSample Text:\n{generated}")

        print("\nTraining complete.")


    def save_model(self, path, char_to_idx, idx_to_char):
        os.makedirs(path, exist_ok=True)

        torch.save(self.state_dict(), os.path.join(path, "model_weights.pt"))

        metadata = {
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "char_to_idx": char_to_idx,
            "idx_to_char": idx_to_char
        }
        with open(os.path.join(path, "metadata.pkl"), "wb") as f:
            pickle.dump(metadata, f)

        print(f"Model saved to: {path}")

    @classmethod
    def load_model(cls, path, device="cpu"):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model path '{path}' does not exist.")

        with open(os.path.join(path, "metadata.pkl"), "rb") as f:
            metadata = pickle.load(f)

        model = cls(
            vocab_size=metadata["vocab_size"],
            embedding_dim=metadata["embedding_dim"],
            hidden_dim=metadata["hidden_dim"],
            num_layers=metadata["num_layers"]
        ).to(device)

        model.load_state_dict(torch.load(os.path.join(path, "model_weights.pt"), map_location=device))

        return model, metadata["char_to_idx"], metadata["idx_to_char"]
    @torch.no_grad()
    def evaluate(self, dataloader, device='cpu'):
        self.eval()
        self.to(device)
        criterion = nn.CrossEntropyLoss()
        total_loss = 0
        total_samples = 0

        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            output, _ = self(x)
            loss = criterion(output.view(-1, self.vocab_size), y.view(-1))
            total_loss += loss.item() * x.size(0)
            total_samples += x.size(0)

        avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
        print(f"Evaluation Loss: {avg_loss:.4f}")
        return avg_loss

    @torch.no_grad()
    def generate_text(self, start_seq, char_to_idx, idx_to_char, length=200, temperature=0.3, device='cpu'):
        self.eval()
        self.to(device)

        input_seq = [char_to_idx.get(ch, 0) for ch in start_seq]
        input_tensor = torch.tensor(input_seq, dtype=torch.long).unsqueeze(0).to(device)

        hidden = self.init_hidden(1, device)

        for i in range(len(input_seq) - 1):
            _, hidden = self(input_tensor[:, i:i+1], hidden)

        input_tensor = input_tensor[:, -1:]
        output_text = start_seq

        for _ in range(length):
            output, hidden = self(input_tensor, hidden)
            logits = output[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1).item()
            next_char = idx_to_char[next_idx]

            output_text += next_char
            input_tensor = torch.tensor([[next_idx]], dtype=torch.long).to(device)

        return output_text

class TextDataset(Dataset):
    def __init__(self, raw_text, seq_length):
        self.seq_length = seq_length
        self.text = self.clean_text(raw_text)

        self.chars = sorted(set(self.text))
        self.char_to_idx = {ch: idx for idx, ch in enumerate(self.chars)}
        self.idx_to_char = {idx: ch for idx, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)

        self.data = [self.char_to_idx[ch] for ch in self.text]

    def clean_text(self, text):
        text = text.lower()
        text = text.replace("\n", " ")
        text = ''.join(c for c in text if c.isprintable())
        return text

    def __len__(self):
        return len(self.data) - self.seq_length
    
    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx:idx+self.seq_length], dtype=torch.long)
        y = torch.tensor(self.data[idx+1:idx+self.seq_length+1], dtype=torch.long)
        return x, y


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    print("Loading Trump speech and tweet text data . . .")

    with open("trump_speeches_final.txt", "r", encoding="utf-8") as f1, \
        open("trump_tweets_final.txt", "r", encoding="utf-8") as f2:
        speech_text = f1.read()
        tweet_text = f2.read()

    text = speech_text + "\n" + tweet_text
    text = text[:1000]
    print("Cleaning and preparing dataset . . .")

    seq_length = 150
    dataset = TextDataset(text, seq_length)

    val_ratio = 0.25
    val_len = int(len(dataset) * val_ratio)
    train_len = len(dataset) - val_len
    train_dataset, val_dataset = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    print("Data cleaned and prepared")

    print("Training model . . .")
    model = CharLSTM(dataset.vocab_size, embedding_dim=128, hidden_dim=256).to(device)
    model.train_model(
    dataset=train_dataset,
    val_loader=val_loader,
    num_epochs=100,
    learning_rate=0.002,
    device=device,
    batch_size=32,
    print_every=10,
    char_to_idx=dataset.char_to_idx,
    idx_to_char=dataset.idx_to_char
    )

    print("Model Trained.")

    model.save_model("checkpoints/best_model", dataset.char_to_idx, dataset.idx_to_char)

    print("Evaluating model . . .")
    model.evaluate(val_loader, device=device)

    print("\nSample Generation:")
    sample = model.generate_text("It attacked", dataset.char_to_idx, dataset.idx_to_char, length=2500, temperature=0.85, device=device)
    print(sample)


if __name__ == "__main__":
    main()