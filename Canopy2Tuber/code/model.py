import torch
from torch import nn

class ResNetBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.GELU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class DualStreamMSTNet(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=4, num_layers=3, dropout=0.2):
        super().__init__()

        # --- Stream A: Transformer Branch ---
        self.embedding_trans = nn.Linear(1, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, input_dim, d_model))
        self.trans_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2, dropout=dropout,
                                       batch_first=True)
            for _ in range(num_layers)
        ])
        self.trans_pool = nn.Sequential(nn.Linear(d_model, 1), nn.Softmax(dim=1))

        # --- Stream B: CNN ResNet Branch ---
        self.cnn_encoder = nn.Sequential(
            ResNetBlock1D(1, 32, kernel_size=3),
            nn.MaxPool1d(2),
            ResNetBlock1D(32, 64, kernel_size=3),
            nn.MaxPool1d(2),
            ResNetBlock1D(64, 128, kernel_size=3),
            nn.AdaptiveAvgPool1d(1)
        )

        # --- Fusion Head ---
        combined_dim = d_model + 128
        self.fusion_head = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        B, N = x.shape
        # Path A
        x_trans = x.unsqueeze(-1)
        x_trans = self.embedding_trans(x_trans)
        x_trans = x_trans + self.pos_embed[:, :N, :]
        for layer in self.trans_layers:
            x_trans = layer(x_trans)
        attn_weights = self.trans_pool(x_trans)
        feat_trans = torch.sum(x_trans * attn_weights, dim=1)

        # Path B
        x_cnn = x.unsqueeze(1)
        feat_cnn = self.cnn_encoder(x_cnn).squeeze(-1)

        # Fusion
        feat_fused = torch.cat([feat_trans, feat_cnn], dim=1)
        out = self.fusion_head(feat_fused)
        return out