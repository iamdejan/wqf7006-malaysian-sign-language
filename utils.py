import torch
import torch.nn as nn

TRAIN_DATASET_PATH = r"./data/train_dataset"
MODEL_FOLDER_PATH = r"./model"
DROPOUT = 0.2


# =========================
# Model
# =========================
class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, h):
        a = torch.tanh(self.proj(h))
        score = self.v(a).squeeze(-1)
        w = torch.softmax(score, dim=1)
        out = torch.sum(h * w.unsqueeze(-1), dim=1)
        return out, w


class CustomLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=2, dropout=0.35):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.norm = nn.LayerNorm(hidden_size * 2)
        self.attn = AttentionPooling(hidden_size * 2)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        h, _ = self.lstm(x)
        h = self.norm(h)
        ctx, _ = self.attn(h)
        return self.mlp(ctx)
