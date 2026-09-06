"""
Arquitecturas de red neuronal. Todas residuales: predicen Delta y devuelven
Y[t] + Delta.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv


def _batch_edge_index(edge_index, batch_size, n_nodos):
    return torch.cat([edge_index + i * n_nodos for i in range(batch_size)], dim=1)


class MLPSimple(nn.Module):
    def __init__(self, n_features, hidden=64, n_layers=2, dropout=0.1):
        super().__init__()
        layers, prev = [], n_features
        for _ in range(n_layers):
            layers += [nn.Linear(prev, hidden), nn.ReLU(), nn.Dropout(dropout)]
            prev = hidden
        last = nn.Linear(prev, 1)
        nn.init.zeros_(last.weight); nn.init.zeros_(last.bias)
        layers.append(last)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        y_last = x[:, :, -3]
        delta = self.net(x).squeeze(-1)
        return y_last + delta


class MLPCompleto(nn.Module):
    def __init__(self, window, n_features, hidden=64, n_layers=2, dropout=0.1):
        super().__init__()
        layers, prev = [], window * n_features
        for _ in range(n_layers):
            layers += [nn.Linear(prev, hidden), nn.ReLU(), nn.Dropout(dropout)]
            prev = hidden
        last = nn.Linear(prev, 1)
        nn.init.zeros_(last.weight); nn.init.zeros_(last.bias)
        layers.append(last)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        B, W, N, F = x.shape
        y_last = x[:, -1, :, -3]
        x_flat = x.permute(0, 2, 1, 3).reshape(B, N, W * F)
        delta = self.net(x_flat).squeeze(-1)
        return y_last + delta


class GCNSimpleV2(nn.Module):
    def __init__(self, n_features, hidden=64, dropout=0.05):
        super().__init__()
        self.encoder = nn.Linear(n_features, hidden)
        self.gcn1 = GCNConv(hidden, hidden)
        self.gcn2 = GCNConv(hidden, hidden)
        self.dropout = nn.Dropout(dropout)
        self.decoder = nn.Sequential(
            nn.Linear(hidden + 3, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        nn.init.xavier_normal_(self.decoder[-1].weight, gain=0.01)
        nn.init.zeros_(self.decoder[-1].bias)

    def forward(self, x, edge_index):
        B, N, F = x.shape
        y_last, y_lag24, y_lag7d = x[:, :, -3], x[:, :, -2], x[:, :, -1]

        h_init = torch.relu(self.encoder(x.reshape(B * N, F)))
        h = self.dropout(h_init)
        ei = _batch_edge_index(edge_index, B, N)
        h = torch.relu(self.gcn1(h, ei))
        h = self.dropout(h)
        h = torch.relu(self.gcn2(h, ei))

        h = (h + h_init).view(B, N, -1)
        y_lags = torch.stack([y_last, y_lag24, y_lag7d], dim=-1)
        delta = self.decoder(torch.cat([h, y_lags], dim=-1)).squeeze(-1)
        return y_last + delta


class GCNCompletoV2(nn.Module):
    def __init__(self, window, n_features, hidden=64, dropout=0.05):
        super().__init__()
        self.encoder = nn.Linear(window * n_features, hidden)
        self.gcn1 = GCNConv(hidden, hidden)
        self.gcn2 = GCNConv(hidden, hidden)
        self.dropout = nn.Dropout(dropout)
        self.decoder = nn.Sequential(
            nn.Linear(hidden + 3, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        nn.init.xavier_normal_(self.decoder[-1].weight, gain=0.01)
        nn.init.zeros_(self.decoder[-1].bias)

    def forward(self, x, edge_index):
        B, W, N, F = x.shape
        y_last, y_lag24, y_lag7d = x[:, -1, :, -3], x[:, -1, :, -2], x[:, -1, :, -1]

        x_flat = x.permute(0, 2, 1, 3).reshape(B * N, W * F)
        h_init = torch.relu(self.encoder(x_flat))
        h = self.dropout(h_init)
        ei = _batch_edge_index(edge_index, B, N)
        h = torch.relu(self.gcn1(h, ei))
        h = self.dropout(h)
        h = torch.relu(self.gcn2(h, ei))

        h = (h + h_init).view(B, N, -1)
        y_lags = torch.stack([y_last, y_lag24, y_lag7d], dim=-1)
        delta = self.decoder(torch.cat([h, y_lags], dim=-1)).squeeze(-1)
        return y_last + delta


class GCNCompletoBig(nn.Module):
    def __init__(self, window, n_features, hidden=128, dropout=0.15):
        super().__init__()
        self.encoder = nn.Linear(window * n_features, hidden)
        self.gcn1 = GCNConv(hidden, hidden)
        self.gcn2 = GCNConv(hidden, hidden)
        self.gcn3 = GCNConv(hidden, hidden)
        self.dropout = nn.Dropout(dropout)
        self.decoder = nn.Sequential(
            nn.Linear(hidden + 3, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1))
        nn.init.xavier_normal_(self.decoder[-1].weight, gain=0.01)
        nn.init.zeros_(self.decoder[-1].bias)

    def forward(self, x, edge_index):
        B, W, N, F = x.shape
        y_last, y_lag24, y_lag7d = x[:, -1, :, -3], x[:, -1, :, -2], x[:, -1, :, -1]

        x_flat = x.permute(0, 2, 1, 3).reshape(B * N, W * F)
        h_init = torch.relu(self.encoder(x_flat))
        h = self.dropout(h_init)
        ei = _batch_edge_index(edge_index, B, N)
        h = torch.relu(self.gcn1(h, ei)); h = self.dropout(h)
        h = torch.relu(self.gcn2(h, ei)); h = self.dropout(h)
        h = torch.relu(self.gcn3(h, ei))

        h = (h + h_init).view(B, N, -1)
        y_lags = torch.stack([y_last, y_lag24, y_lag7d], dim=-1)
        delta = self.decoder(torch.cat([h, y_lags], dim=-1)).squeeze(-1)
        return y_last + delta


class GATCompleto(nn.Module):
    def __init__(self, window, n_features, hidden=64, heads=4, dropout=0.1):
        super().__init__()
        self.encoder = nn.Linear(window * n_features, hidden)
        self.gat1 = GATConv(hidden, hidden, heads=heads, dropout=dropout, concat=True)
        self.gat2 = GATConv(hidden * heads, hidden, heads=1, dropout=dropout, concat=False)
        self.dropout = nn.Dropout(dropout)
        self.decoder = nn.Sequential(
            nn.Linear(hidden + 3, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        nn.init.xavier_normal_(self.decoder[-1].weight, gain=0.01)
        nn.init.zeros_(self.decoder[-1].bias)

    def forward(self, x, edge_index):
        B, W, N, F = x.shape
        y_last, y_lag24, y_lag7d = x[:, -1, :, -3], x[:, -1, :, -2], x[:, -1, :, -1]

        x_flat = x.permute(0, 2, 1, 3).reshape(B * N, W * F)
        h_init = torch.relu(self.encoder(x_flat))
        h = self.dropout(h_init)
        ei = _batch_edge_index(edge_index, B, N)
        h = torch.relu(self.gat1(h, ei))
        h = self.dropout(h)
        h = torch.relu(self.gat2(h, ei))

        h = (h + h_init).view(B, N, -1)
        y_lags = torch.stack([y_last, y_lag24, y_lag7d], dim=-1)
        delta = self.decoder(torch.cat([h, y_lags], dim=-1)).squeeze(-1)
        return y_last + delta


class GATCompletoPlus(nn.Module):
    """MODELO FINAL (28.35 veh/h). Índices de Y[t]/lag24/lag7d POSICIONALES
    (22,23,24), no negativos, porque hay más features detrás."""
    def __init__(self, window, n_features, hidden=64, heads=4, dropout=0.1):
        super().__init__()
        self.encoder = nn.Linear(window * n_features, hidden)
        self.gat1 = GATConv(hidden, hidden, heads=heads, dropout=dropout, concat=True)
        self.gat2 = GATConv(hidden * heads, hidden, heads=1, dropout=dropout, concat=False)
        self.dropout = nn.Dropout(dropout)
        self.decoder = nn.Sequential(
            nn.Linear(hidden + 3, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        nn.init.xavier_normal_(self.decoder[-1].weight, gain=0.01)
        nn.init.zeros_(self.decoder[-1].bias)

    def forward(self, x, edge_index):
        B, W, N, F = x.shape
        y_last, y_lag24, y_lag7d = x[:, -1, :, 22], x[:, -1, :, 23], x[:, -1, :, 24]

        x_flat = x.permute(0, 2, 1, 3).reshape(B * N, W * F)
        h_init = torch.relu(self.encoder(x_flat))
        h = self.dropout(h_init)
        ei = _batch_edge_index(edge_index, B, N)
        h = torch.relu(self.gat1(h, ei))
        h = self.dropout(h)
        h = torch.relu(self.gat2(h, ei))

        h = (h + h_init).view(B, N, -1)
        y_lags = torch.stack([y_last, y_lag24, y_lag7d], dim=-1)
        delta = self.decoder(torch.cat([h, y_lags], dim=-1)).squeeze(-1)
        return y_last + delta


class GATCompletoUltra(nn.Module):
    def __init__(self, window, n_features, hidden=64, heads=4, dropout=0.1):
        super().__init__()
        self.encoder = nn.Linear(window * n_features, hidden)
        self.gat1 = GATConv(hidden, hidden, heads=heads, dropout=dropout,
                             concat=True, negative_slope=0.01)
        self.gat2 = GATConv(hidden * heads, hidden, heads=heads, dropout=dropout,
                             concat=False, negative_slope=0.01)
        self.skip_proj = nn.Linear(hidden * heads, hidden)
        self.dropout = nn.Dropout(dropout)
        self.decoder = nn.Sequential(
            nn.Linear(hidden + 3, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        nn.init.xavier_normal_(self.decoder[-1].weight, gain=0.01)
        nn.init.zeros_(self.decoder[-1].bias)

    def forward(self, x, edge_index):
        B, W, N, F = x.shape
        y_last, y_lag24, y_lag7d = x[:, -1, :, 22], x[:, -1, :, 23], x[:, -1, :, 24]

        x_flat = x.permute(0, 2, 1, 3).reshape(B * N, W * F)
        h_init = torch.relu(self.encoder(x_flat))
        h = self.dropout(h_init)
        ei = _batch_edge_index(edge_index, B, N)

        h1 = torch.relu(self.gat1(h, ei))
        h1_d = self.dropout(h1)
        h2 = torch.relu(self.gat2(h1_d, ei))

        h = (h2 + self.skip_proj(h1) + h_init).view(B, N, -1)
        y_lags = torch.stack([y_last, y_lag24, y_lag7d], dim=-1)
        delta = self.decoder(torch.cat([h, y_lags], dim=-1)).squeeze(-1)
        return y_last + delta