"""Helpers compartidos por el pipeline de construcción de tensores."""

import pandas as pd
import numpy as np


def normalize_id(x):
    """Normaliza identificadores heterogéneos (int, float con .0, str) a str limpio."""
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s.endswith('.0'):
        s = s[:-2]
    try:
        return str(int(float(s)))
    except (ValueError, TypeError):
        return s


def load_and_filter(parquet_path, start, end, ts_col='timestamp'):
    """Carga un parquet, filtra por rango temporal y redondea a 15 min."""
    df = pd.read_parquet(parquet_path)
    df[ts_col] = pd.to_datetime(df[ts_col])
    mask = (df[ts_col] >= start) & (df[ts_col] <= end)
    df = df[mask].copy()
    df[ts_col] = df[ts_col].dt.floor('15min')
    return df


def aggregate_15min(df, ts_col, id_col, value_col, agg='mean'):
    return df.groupby([ts_col, id_col], as_index=False)[value_col].agg(agg)


def aplicar_fix_ceros_extendidos(Y_all, M_all, racha_umbral=16):
    """
    Detecta rachas de ih_trafico=0 durante >= racha_umbral timesteps consecutivos
    (con mask=1, i.e. "dato válido pero cero") y las marca como mask=0.

    Un sensor con >=4h seguidas de "cero vehículos" es, con altísima probabilidad,
    un sensor roto, no una calle vacía. El umbral de 16 timesteps se eligió para
    NO penalizar las horas muertas de madrugada (2-3h de cero son plausibles ahí).
    """
    T, N = Y_all.shape
    M_fixed = M_all.copy()
    n_marcados_por_sensor = np.zeros(N, dtype=np.int32)

    for n in range(N):
        y_n = Y_all[:, n]
        m_n = M_all[:, n]
        cero_activo = (y_n == 0) & (m_n == 1)

        diff = np.diff(np.concatenate([[0], cero_activo.astype(np.int8), [0]]))
        inicios = np.where(diff == 1)[0]
        finales = np.where(diff == -1)[0]

        for ini, fin in zip(inicios, finales):
            if fin - ini >= racha_umbral:
                M_fixed[ini:fin, n] = 0
                n_marcados_por_sensor[n] += fin - ini

    return M_fixed, n_marcados_por_sensor