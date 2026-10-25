"""Minimal deployed Dual-Stream MST-Net definition for LAI inference.

This architecture is copied from the source deployment script
``stage2_transformer应用.py`` and must be used unchanged with the supplied
``MSTNet_LAI_best.pth`` checkpoint.
"""

import torch
from torch import nn


class ResNetBlock1D(nn.Module):
    """Residual 1D convolution block used by the spectral CNN branch."""

    def __init__(self, in_c: int, out_c: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_c, out_c, 3, padding=1),
            nn.BatchNorm1d(out_c),
            nn.GELU(),
            nn.Conv1d(out_c, out_c, 3, padding=1),
            nn.BatchNorm1d(out_c),
        )
        self.relu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.conv(x))


class DualStreamMSTNet(nn.Module):
    """Transformer plus residual-CNN spectral regressor for LAI."""

    def __init__(self, input_dim: int, d_model: int = 128) -> None:
        super().__init__()
        self.embedding = nn.Linear(1, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, input_dim, d_model))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=4, batch_first=True),
            num_layers=3,
        )
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 64, 3, padding=1),
            nn.GELU(),
            ResNetBlock1D(64, 64),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Linear(d_model + 64, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        transformer_features = self.embedding(x.unsqueeze(-1)) + self.pos_embed
        transformer_features = torch.mean(self.transformer(transformer_features), dim=1)
        cnn_features = self.cnn(x.unsqueeze(1)).squeeze(-1)
        return self.head(torch.cat([transformer_features, cnn_features], dim=1))
