import torch

from dataset import build_vocab, simple_tokenize
from model_transformer import TransformerClassifier


def encode_text(text, vocab, max_len=256):
    tokens = simple_tokenize(text)
    ids = [vocab.get(token, vocab["<UNK>"]) for token in tokens]

    if len(ids) > max_len:
        ids = ids[:max_len]
    else:
        ids = ids + [vocab["<PAD>"]] * (max_len - len(ids))

    return torch.tensor([ids], dtype=torch.long)


def predict_text(model, text, vocab, device):
    model.eval()
    with torch.no_grad():
        x = encode_text(text, vocab).to(device)
        out = model(x)
        pred = out.argmax(dim=1).item()
    return "positive" if pred == 1 else "negative"