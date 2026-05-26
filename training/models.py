import torch
import torch.nn as nn
from torchvision import models


class MobileNetV3LSTM(nn.Module):
    """
    MobileNetV3-Large backbone + 2-layer BiLSTM classifier (Experiment 3).

    Architecture:
      - CNN backbone: MobileNetV3-Large feature extractor (output 960-dim per frame)
      - AdaptiveAvgPool2d → (B*T, 960)
      - BatchNorm1d for feature normalisation
      - BiLSTM: hidden=256, layers=2, dropout=0.3 → (B, T, 512) → mean-pooled
      - Dropout(0.6) → Linear(512,256) → ReLU → Dropout(0.5) → Linear(256,100)
    All parameters are trainable (no frozen layers).
    """

    def __init__(self, num_classes=100, hidden_size=256, lstm_layers=2):
        super().__init__()

        backbone = models.mobilenet_v3_large(
            weights=models.MobileNet_V3_Large_Weights.DEFAULT
        )
        self.cnn = backbone.features          # output: (B*T, 960, 7, 7)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_dim = 960
        self.feature_norm = nn.BatchNorm1d(self.feature_dim)

        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )

        self.dropout = nn.Dropout(0.6)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        batch_size, seq_len, C, H, W = x.size()
        x = x.reshape(batch_size * seq_len, C, H, W)

        features = self.cnn(x)                          # (B*T, 960, 7, 7)
        features = self.pool(features).squeeze(-1).squeeze(-1)  # (B*T, 960)
        features = self.feature_norm(features)
        features = features.view(batch_size, seq_len, self.feature_dim)

        lstm_out, _ = self.lstm(features)
        out = torch.mean(lstm_out, dim=1)
        out = self.dropout(out)
        return self.fc(out)


class ResNet50LSTM(nn.Module):
    """
    ResNet50 backbone + 2-layer BiLSTM classifier (Experiment 3 baseline).

    Architecture:
      - CNN backbone: ResNet50 minus final FC layer (output 2048-dim per frame)
      - BatchNorm1d for feature normalisation
      - BiLSTM: hidden=256, layers=2, dropout=0.3 → (B, T, 512) → mean-pooled
      - Dropout(0.6) → Linear(512,256) → ReLU → Dropout(0.5) → Linear(256,100)
    All parameters are trainable (no frozen layers).
    """

    def __init__(self, num_classes=100, hidden_size=256, lstm_layers=2):
        super().__init__()

        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])  # strips FC, keeps AvgPool
        for param in self.cnn.parameters():
            param.requires_grad = True

        self.feature_dim = 2048
        self.feature_norm = nn.BatchNorm1d(self.feature_dim)

        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )

        self.dropout = nn.Dropout(0.6)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        batch_size, seq_len, C, H, W = x.size()
        x = x.reshape(batch_size * seq_len, C, H, W)

        features = self.cnn(x).squeeze(-1).squeeze(-1)  # (B*T, 2048)
        features = self.feature_norm(features)
        features = features.view(batch_size, seq_len, self.feature_dim)

        lstm_out, _ = self.lstm(features)
        out = torch.mean(lstm_out, dim=1)
        out = self.dropout(out)
        return self.fc(out)


class ResNet101LSTM(nn.Module):
    """
    ResNet101 backbone + 2-layer BiLSTM classifier (Experiment 3).

    Architecture:
      - CNN backbone: ResNet101 minus final FC layer (output 2048-dim per frame)
      - BatchNorm1d for feature normalisation
      - BiLSTM: hidden=256, layers=2, dropout=0.3 → (B, T, 512) → mean-pooled
      - Dropout(0.6) → Linear(512,256) → ReLU → Dropout(0.5) → Linear(256,100)
    All parameters are trainable (no frozen layers).
    """

    def __init__(self, num_classes=100, hidden_size=256, lstm_layers=2):
        super().__init__()

        resnet = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])
        for param in self.cnn.parameters():
            param.requires_grad = True

        self.feature_dim = 2048
        self.feature_norm = nn.BatchNorm1d(self.feature_dim)

        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )

        self.dropout = nn.Dropout(0.6)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        batch_size, seq_len, C, H, W = x.size()
        x = x.reshape(batch_size * seq_len, C, H, W)

        features = self.cnn(x).squeeze(-1).squeeze(-1)  # (B*T, 2048)
        features = self.feature_norm(features)
        features = features.view(batch_size, seq_len, self.feature_dim)

        lstm_out, _ = self.lstm(features)
        out = torch.mean(lstm_out, dim=1)
        out = self.dropout(out)
        return self.fc(out)
