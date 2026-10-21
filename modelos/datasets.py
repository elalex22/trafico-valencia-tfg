"""
Datasets de PyTorch.

- DatasetSimple:       (N, 25)      -- sin ventana temporal
- DatasetCompleto:     (W, N, 25)   -- con ventana de W timesteps
- DatasetCompletoPlus: (W, N, 33)   -- añade lags cortos y diferencias

Orden de features en los últimos canales:
  Simple / Completo (25):  [..22 contextuales..] Y[t] Y[t-96] Y[t-672]
  CompletoPlus (33):       [..22 contextuales..] Y[t](22) Y[t-96](23) Y[t-672](24)
                            Y[t-1](25) Y[t-2](26) Y[t-4](27) Y[t-8](28)
                            dY[t-1](29) dY[t-4](30) dY[t-96](31) dY[t-672](32)
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class DatasetSimple(Dataset):
    def __init__(self, data_split, window=12, horizon=1):
        self.X = data_split['X']
        self.Y = data_split['Y']
        self.M = data_split['M']
        self.Y_lag24 = data_split['Y_lag24']
        self.Y_lag7d = data_split['Y_lag7d']
        self.M_lag24 = data_split['M_lag24']
        self.M_lag7d = data_split['M_lag7d']
        self.window = window
        self.horizon = horizon
        self.n_samples = len(self.X) - window - horizon + 1

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        t_now = idx + self.window - 1
        t_target = t_now + self.horizon

        x_feat = np.asarray(self.X[t_now])
        y_now = np.asarray(self.Y[t_now]) * np.asarray(self.M[t_now]).astype(np.float32)
        y_lag24 = np.asarray(self.Y_lag24[t_now]) * np.asarray(self.M_lag24[t_now]).astype(np.float32)
        y_lag7d = np.asarray(self.Y_lag7d[t_now]) * np.asarray(self.M_lag7d[t_now]).astype(np.float32)

        x = np.concatenate([
            x_feat, y_now[:, None], y_lag24[:, None], y_lag7d[:, None],
        ], axis=-1).copy()

        y = np.asarray(self.Y[t_target]).copy()
        m = np.asarray(self.M[t_target]).astype(np.float32).copy()

        return (torch.from_numpy(x).float(),
                torch.from_numpy(y).float(),
                torch.from_numpy(m).float())


class DatasetCompleto(Dataset):
    def __init__(self, data_split, window=12, horizon=1):
        self.X = data_split['X']
        self.Y = data_split['Y']
        self.M = data_split['M']
        self.Y_lag24 = data_split['Y_lag24']
        self.Y_lag7d = data_split['Y_lag7d']
        self.M_lag24 = data_split['M_lag24']
        self.M_lag7d = data_split['M_lag7d']
        self.window = window
        self.horizon = horizon
        self.n_samples = len(self.X) - window - horizon + 1

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x_feat = np.asarray(self.X[idx:idx + self.window])

        y_hist = np.asarray(self.Y[idx:idx + self.window])
        m_hist = np.asarray(self.M[idx:idx + self.window]).astype(np.float32)
        y_hist = y_hist * m_hist

        y_lag24_hist = np.asarray(self.Y_lag24[idx:idx + self.window])
        m_lag24_hist = np.asarray(self.M_lag24[idx:idx + self.window]).astype(np.float32)
        y_lag24_hist = y_lag24_hist * m_lag24_hist

        y_lag7d_hist = np.asarray(self.Y_lag7d[idx:idx + self.window])
        m_lag7d_hist = np.asarray(self.M_lag7d[idx:idx + self.window]).astype(np.float32)
        y_lag7d_hist = y_lag7d_hist * m_lag7d_hist

        x = np.concatenate([
            x_feat, y_hist[:, :, None], y_lag24_hist[:, :, None], y_lag7d_hist[:, :, None],
        ], axis=-1).copy()

        t_target = idx + self.window + self.horizon - 1
        y = np.asarray(self.Y[t_target]).copy()
        m = np.asarray(self.M[t_target]).astype(np.float32).copy()

        return (torch.from_numpy(x).float(),
                torch.from_numpy(y).float(),
                torch.from_numpy(m).float())


class DatasetCompletoPlus(Dataset):
    """Igual que DatasetCompleto + lags cortos (t-1,t-2,t-4,t-8) y diferencias.
    Requiere 8 timesteps adicionales de contexto antes del inicio de la
    ventana (start_offset) -> tiene 8 muestras menos que DatasetCompleto
    sobre el mismo split."""
    def __init__(self, data_split, window=12, horizon=1):
        self.X = data_split['X']
        self.Y = data_split['Y']
        self.M = data_split['M']
        self.Y_lag24 = data_split['Y_lag24']
        self.Y_lag7d = data_split['Y_lag7d']
        self.M_lag24 = data_split['M_lag24']
        self.M_lag7d = data_split['M_lag7d']
        self.window = window
        self.horizon = horizon
        self.start_offset = 8
        self.n_samples = len(self.X) - window - horizon + 1 - self.start_offset

    def __len__(self):
        return self.n_samples

    def _y_masked(self, a, b):
        return np.asarray(self.Y[a:b]) * np.asarray(self.M[a:b]).astype(np.float32)

    def __getitem__(self, idx):
        idx = idx + self.start_offset

        x_feat = np.asarray(self.X[idx:idx + self.window])

        y_hist = self._y_masked(idx, idx + self.window)
        y_lag24_hist = (np.asarray(self.Y_lag24[idx:idx + self.window]) *
                         np.asarray(self.M_lag24[idx:idx + self.window]).astype(np.float32))
        y_lag7d_hist = (np.asarray(self.Y_lag7d[idx:idx + self.window]) *
                         np.asarray(self.M_lag7d[idx:idx + self.window]).astype(np.float32))

        y_lag1_hist = self._y_masked(idx - 1, idx - 1 + self.window)
        y_lag2_hist = self._y_masked(idx - 2, idx - 2 + self.window)
        y_lag4_hist = self._y_masked(idx - 4, idx - 4 + self.window)
        y_lag8_hist = self._y_masked(idx - 8, idx - 8 + self.window)

        d_lag1_hist = y_hist - y_lag1_hist
        d_lag4_hist = y_hist - y_lag4_hist
        d_lag24_hist = y_hist - y_lag24_hist
        d_lag7d_hist = y_hist - y_lag7d_hist

        x = np.concatenate([
            x_feat,
            y_hist[:, :, None], y_lag24_hist[:, :, None], y_lag7d_hist[:, :, None],
            y_lag1_hist[:, :, None], y_lag2_hist[:, :, None],
            y_lag4_hist[:, :, None], y_lag8_hist[:, :, None],
            d_lag1_hist[:, :, None], d_lag4_hist[:, :, None],
            d_lag24_hist[:, :, None], d_lag7d_hist[:, :, None],
        ], axis=-1).copy()

        t_target = idx + self.window + self.horizon - 1
        y = np.asarray(self.Y[t_target]).copy()
        m = np.asarray(self.M[t_target]).astype(np.float32).copy()

        return (torch.from_numpy(x).float(),
                torch.from_numpy(y).float(),
                torch.from_numpy(m).float())