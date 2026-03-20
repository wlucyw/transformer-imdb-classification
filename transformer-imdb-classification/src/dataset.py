import re
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset


def simple_tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    tokens = text.split()
    return tokens


def build_vocab(texts, max_vocab_size=20000, min_freq=2):
    counter = Counter()
    for text in texts:
        counter.update(simple_tokenize(text))

    vocab = {
        "<PAD>": 0,
        "<UNK>": 1
    }

    for word, freq in counter.most_common():
        if freq < min_freq:
            continue
        if len(vocab) >= max_vocab_size:
            break
        vocab[word] = len(vocab)

    return vocab


def encode_text(text, vocab, max_len=256):
    tokens = simple_tokenize(text)
    ids = [vocab.get(token, vocab["<UNK>"]) for token in tokens]

    if len(ids) > max_len:
        ids = ids[:max_len]
    else:
        ids = ids + [vocab["<PAD>"]] * (max_len - len(ids))

    return ids


class IMDBDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=256):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        input_ids = encode_text(self.texts[idx], self.vocab, self.max_len)
        label = self.labels[idx]
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)


def load_imdb_data(max_vocab_size=20000, max_len=256, batch_size=64):
    dataset = load_dataset("imdb")

    train_texts = dataset["train"]["text"]
    train_labels = dataset["train"]["label"]

    test_texts = dataset["test"]["text"]
    test_labels = dataset["test"]["label"]

    vocab = build_vocab(train_texts, max_vocab_size=max_vocab_size)

    train_dataset = IMDBDataset(train_texts, train_labels, vocab, max_len=max_len)
    test_dataset = IMDBDataset(test_texts, test_labels, vocab, max_len=max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, vocab