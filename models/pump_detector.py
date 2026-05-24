import torch
import torch.nn as nn


class FeatureExtractor(nn.Module):
    """
    Shared 3-block 1D-CNN backbone used by both PumpDetector (v2) and PumpDetectorV3.

    Input:  [Batch, Timesteps, num_features]
    Output: [Batch, 128]  — temporal axis fully collapsed via AdaptiveAvgPool1d(1)
    """

    def __init__(self, num_features: int):
        super().__init__()
        self.net = nn.Sequential(
            # Block 1 — local spike / spread anomaly detection
            nn.Conv1d(num_features, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.1),

            # Block 2 — medium-range pump build-up structure
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.1),

            # Block 3 — high-level pump-vs-control aggregation
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),   # → [B, 128, 1]
        )
        self.flatten = nn.Flatten()    # → [B, 128]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)         # [B, T, F] → [B, F, T]
        return self.flatten(self.net(x))


class PumpDetector(nn.Module):
    """
    v2 single-stream detector (backwards-compatible).
    Input:  [Batch, 96, num_features]   (default 6: bid/ask price+size + buy_ratio + agg_imb)
    Output: [Batch]  scalar in [0, 1]
    """

    def __init__(self, num_features: int = 6):
        super().__init__()
        self.backbone   = FeatureExtractor(num_features)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1),   nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.backbone(x)).squeeze(-1)


class PumpDetectorV3(nn.Module):
    """
    v3 dual-stream detector — separate CNN towers for coin L2 and market context.

    Captures the 3×3 coin×market regime matrix:
      Coin: Normal / Uncertain / Pumped  ×  Market: Normal / Uncertain / Pumped

    The most suspicious case (Coin:Pumped + Market:Normal) produces the largest
    combined feature divergence between the two streams.

    Inputs:
      x_coin   [Batch, 96, num_coin_features]    — coin orderbook + trade flow
      x_market [Batch, 96, num_market_features]  — BTC/market context (same feature schema)
    Outputs:
      pump_prob  [Batch]  scalar in [0, 1]  — pump probability (classification head)
      peak_pos   [Batch]  scalar in [0, 1]  — normalised peak position peak_idx/96
                                              (regression head; trained on synthetic data)
    """

    def __init__(self, num_coin_features: int = 6, num_market_features: int = 6):
        super().__init__()
        self.coin_stream   = FeatureExtractor(num_coin_features)    # → [B, 128]
        self.market_stream = FeatureExtractor(num_market_features)  # → [B, 128]
        # Shared trunk: merges coin + market representations
        self.trunk = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
        )
        # Classification head — is this a pump?
        self.cls_head = nn.Sequential(nn.Linear(128, 1), nn.Sigmoid())
        # Regression head — where is the peak within the 96-step window?
        self.reg_head = nn.Sequential(nn.Linear(128, 1), nn.Sigmoid())

    def forward(
        self, x_coin: torch.Tensor, x_market: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        coin_feat   = self.coin_stream(x_coin)      # [B, 128]
        market_feat = self.market_stream(x_market)  # [B, 128]
        combined    = torch.cat([coin_feat, market_feat], dim=1)  # [B, 256]
        trunk       = self.trunk(combined)                        # [B, 128]
        pump_prob   = self.cls_head(trunk).squeeze(-1)            # [B]
        peak_pos    = self.reg_head(trunk).squeeze(-1)            # [B] in [0, 1]
        return pump_prob, peak_pos
