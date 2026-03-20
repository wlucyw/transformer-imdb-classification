import torch
import torch.nn as nn


class RNNClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim=128,
        hidden_dim=128,
        num_layers=1,
        num_classes=2,
        model_type="lstm",
        bidirectional=True,
        dropout=0.2
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.model_type = model_type.lower()
        self.bidirectional = bidirectional

        if self.model_type == "rnn":
            self.encoder = nn.RNN(
                input_size=embed_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout if num_layers > 1 else 0
            )
        elif self.model_type == "lstm":
            self.encoder = nn.LSTM(
                input_size=embed_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout if num_layers > 1 else 0
            )
        else:
            raise ValueError("model_type must be 'rnn' or 'lstm'")

        out_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(out_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)

        if self.model_type == "lstm":
            _, (hidden, _) = self.encoder(x)
        else:
            _, hidden = self.encoder(x)

        if self.bidirectional:
            hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            hidden = hidden[-1]

        hidden = self.dropout(hidden)
        out = self.fc(hidden)

        return out