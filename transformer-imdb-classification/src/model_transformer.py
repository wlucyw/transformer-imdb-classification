import torch
import torch.nn as nn


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim=128,
        num_heads=4,
        ff_dim=256,
        num_layers=2,
        num_classes=2,
        dropout=0.2,
        max_len=256
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_len, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="relu"
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        batch_size, seq_len = x.size()

        padding_mask = (x == 0)

        positions = torch.arange(
            0,
            seq_len,
            device=x.device
        ).unsqueeze(0).expand(batch_size, seq_len)

        x_embed = self.embedding(x)
        pos_embed = self.position_embedding(positions)
        x = x_embed + pos_embed

        x = self.encoder(
            x,
            src_key_padding_mask=padding_mask
        )

        valid_mask = (~padding_mask).unsqueeze(-1).float()
        x = x * valid_mask
        x = x.sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1e-9)

        x = self.dropout(x)
        out = self.fc(x)

        return out