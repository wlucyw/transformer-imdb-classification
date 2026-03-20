import os
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

from dataset import load_imdb_data
from model_transformer import TransformerClassifier
from model_rnn import RNNClassifier


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    return total_loss / len(loader), acc, f1


def get_model(model_name, vocab_size):
    if model_name == "transformer":
        return TransformerClassifier(
            vocab_size=vocab_size,
            embed_dim=128,
            num_heads=4,
            ff_dim=256,
            num_layers=2,
            dropout=0.2,
            max_len=256
        )

    elif model_name == "lstm":
        return RNNClassifier(
            vocab_size=vocab_size,
            embed_dim=128,
            hidden_dim=128,
            model_type="lstm"
        )

    elif model_name == "rnn":
        return RNNClassifier(
            vocab_size=vocab_size,
            embed_dim=128,
            hidden_dim=128,
            model_type="rnn"
        )

    else:
        raise ValueError("invalid model_name")


def main():
    os.makedirs("outputs/models", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # 一次性加载数据（所有模型共享）
    train_loader, test_loader, vocab = load_imdb_data(
        max_vocab_size=20000,
        max_len=256,
        batch_size=64
    )

    model_list = ["rnn", "lstm", "transformer"]

    results = []

    for model_name in model_list:
        print("\n" + "=" * 50)
        print(f"Training model: {model_name}")
        print("=" * 50)

        model = get_model(model_name, len(vocab))
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        best_acc = 0.0
        num_epochs = 5

        for epoch in range(num_epochs):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc, val_f1 = evaluate(model, test_loader, criterion, device)

            print(f"[{model_name}] Epoch {epoch + 1}/{num_epochs}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Acc:    {val_acc:.4f}")
            print(f"Val F1:     {val_f1:.4f}")
            print("-" * 40)

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), f"outputs/models/best_{model_name}.pt")

        results.append((model_name, best_acc))

    # 最终结果汇总
    print("\n" + "=" * 50)
    print("FINAL RESULTS")
    print("=" * 50)

    for model_name, acc in results:
        print(f"{model_name}: {acc:.4f}")


if __name__ == "__main__":
    main()