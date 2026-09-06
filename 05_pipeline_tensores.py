"""
05_pipeline_tensores.py — Tensorización final: X/Y/M (+ lags), limpieza del
target (outliers, log1p) y estandarización de features de entrada.

Requiere: 04_fusion_master.py ya ejecutado.

Genera en tensors_final/: X_static.npy, X_{split}_norm.npy, Y_{split}_log.npy,
M_{split}.npy, Y_{split}_lag24_log.npy, M_{split}_lag24.npy,
Y_{split}_lag7d_log.npy, M_{split}_lag7d.npy, timestamps_{split}.npy,
config.json, stats.json
"""

import gc
import json
import numpy as np
import pandas as pd

from config import (MASTER_DIR, GRAPH_DIR, TENSOR_DIR, FEATURES_DIN,
                     FEATURES_STATIC, FEATURES_A_ESTANDARIZAR, MASK_POR_FEATURE,
                     UMBRAL_MAX_Y)

ESPERADO_T = {'train': 42460, 'val': 9312, 'test': 11040}


def cargar_split(split_name):
    df = pd.read_parquet(MASTER_DIR / f'master_{split_name}.parquet')
    df = df.sort_values(['timestamp', 'idx']).reset_index(drop=True)

    N = int(df['idx'].nunique())
    T = int(df['timestamp'].nunique())
    assert len(df) == T * N, f'{split_name}: grid incompleto {len(df)} != {T*N}'

    esperado = ESPERADO_T.get(split_name)
    check = '✅' if T == esperado else f'⚠️ esperaba {esperado:,}'
    print(f'  {split_name}: T={T:,} N={N} {check} | '
          f'cobertura mask_valid={df["mask_valid"].mean()*100:.2f}%')
    return df, T, N


def construir_tensores(df, T, N):
    for col in FEATURES_DIN:
        if df[col].isna().any():
            df[col] = df[col].fillna(0)

    X = df[FEATURES_DIN].values.astype(np.float32).reshape(T, N, len(FEATURES_DIN))
    Y = df['ih_trafico'].fillna(0).values.astype(np.float32).reshape(T, N)
    M = df['mask_valid'].values.astype(np.int8).reshape(T, N)
    Y_lag24 = df['Y_lag24'].fillna(0).values.astype(np.float32).reshape(T, N)
    M_lag24 = df['M_lag24'].values.astype(np.int8).reshape(T, N)
    Y_lag7d = df['Y_lag7d'].fillna(0).values.astype(np.float32).reshape(T, N)
    M_lag7d = df['M_lag7d'].values.astype(np.int8).reshape(T, N)
    ts = df['timestamp'].drop_duplicates().sort_values().values
    return X, Y, M, Y_lag24, M_lag24, Y_lag7d, M_lag7d, ts


def limpiar_y_log(Y, M, umbral_max=UMBRAL_MAX_Y):
    """-1 (fuera de servicio) y outliers (>umbral) -> mask=0. Misma regla
    para Y, Y_lag24, Y_lag7d (misma variable, distintos instantes)."""
    problemas = (Y == -1) | (Y < -1) | (Y > umbral_max)
    M_clean = M.copy()
    M_clean[problemas] = 0

    Y_clean = Y.copy()
    Y_clean[M_clean == 0] = 0
    Y_clean = np.clip(Y_clean, 0, umbral_max)
    Y_log = np.log1p(Y_clean).astype(np.float32)
    return Y_log, M_clean.astype(np.int8), int(problemas.sum())


def calcular_stats_normalizacion(X_train, M_train):
    stats = {}
    for fname in FEATURES_A_ESTANDARIZAR:
        idx = FEATURES_DIN.index(fname)
        values = X_train[:, :, idx]
        mask_name = MASK_POR_FEATURE.get(fname)
        if mask_name and mask_name in FEATURES_DIN:
            mask_valid = X_train[:, :, FEATURES_DIN.index(mask_name)] > 0.5
        else:
            mask_valid = M_train > 0

        vals = values[mask_valid]
        mean_v, std_v = (0.0, 1.0) if len(vals) == 0 else (float(vals.mean()), float(vals.std()))
        if std_v < 1e-6:
            std_v = 1.0

        stats[fname] = {'idx': idx, 'mean': mean_v, 'std': std_v, 'n_valid': int(mask_valid.sum())}
        print(f'    [{idx:2d}] {fname:15s} mean={mean_v:>10.3f} std={std_v:>10.3f} '
              f'n_valid={mask_valid.sum():>12,}')
    return stats


def normalizar_X(X, stats):
    X_norm = X.copy()
    for fname, s in stats.items():
        idx, mean_v, std_v = s['idx'], s['mean'], s['std']
        mask_name = MASK_POR_FEATURE.get(fname)
        if mask_name and mask_name in FEATURES_DIN:
            mask_valid = X[:, :, FEATURES_DIN.index(mask_name)] > 0.5
        else:
            mask_valid = np.ones(X.shape[:2], dtype=bool)
        X_norm[:, :, idx] = np.where(mask_valid, (X[:, :, idx] - mean_v) / std_v, 0.0).astype(np.float32)
    return X_norm


def main():
    print('=' * 70)
    print('05_pipeline_tensores — Tensorización + limpieza Y + normalización X')
    print('=' * 70)

    node_order = pd.read_csv(GRAPH_DIR / 'node_order_1035.csv').sort_values('idx')
    X_static = node_order[FEATURES_STATIC].values.astype(np.float32)
    np.save(TENSOR_DIR / 'X_static.npy', X_static)
    print(f'\nX_static: {X_static.shape}')

    datos = {}
    print('\n[1] Cargando y tensorizando splits...')
    for split_name in ['train', 'val', 'test']:
        df, T, N = cargar_split(split_name)
        X, Y, M, Y_lag24, M_lag24, Y_lag7d, M_lag7d, ts = construir_tensores(df, T, N)
        datos[split_name] = dict(X=X, Y=Y, M=M, Y_lag24=Y_lag24, M_lag24=M_lag24,
                                  Y_lag7d=Y_lag7d, M_lag7d=M_lag7d)
        np.save(TENSOR_DIR / f'timestamps_{split_name}.npy', ts)
        del df
        gc.collect()

    print('\n[2] Limpiando target (outliers, -1) y transformando log1p...')
    for split_name, d in datos.items():
        Y_log, M_clean, n_prob = limpiar_y_log(d['Y'], d['M'])
        Ylag24_log, Mlag24_clean, _ = limpiar_y_log(d['Y_lag24'], d['M_lag24'])
        Ylag7d_log, Mlag7d_clean, _ = limpiar_y_log(d['Y_lag7d'], d['M_lag7d'])
        print(f'  {split_name}: {n_prob:,} celdas excluidas por -1/outlier '
              f'(cobertura final: {M_clean.mean()*100:.2f}%)')
        d.update(Y_log=Y_log, M=M_clean, Y_lag24_log=Ylag24_log, M_lag24=Mlag24_clean,
                  Y_lag7d_log=Ylag7d_log, M_lag7d=Mlag7d_clean)

    print('\n[3] Estadísticas de normalización (desde train)...')
    stats_x = calcular_stats_normalizacion(datos['train']['X'], datos['train']['M'])

    print('\n[4] Normalizando y guardando tensores finales...')
    for split_name, d in datos.items():
        X_norm = normalizar_X(d['X'], stats_x)
        np.save(TENSOR_DIR / f'X_{split_name}_norm.npy', X_norm)
        np.save(TENSOR_DIR / f'Y_{split_name}_log.npy', d['Y_log'])
        np.save(TENSOR_DIR / f'M_{split_name}.npy', d['M'])
        np.save(TENSOR_DIR / f'Y_{split_name}_lag24_log.npy', d['Y_lag24_log'])
        np.save(TENSOR_DIR / f'M_{split_name}_lag24.npy', d['M_lag24'])
        np.save(TENSOR_DIR / f'Y_{split_name}_lag7d_log.npy', d['Y_lag7d_log'])
        np.save(TENSOR_DIR / f'M_{split_name}_lag7d.npy', d['M_lag7d'])
        print(f'  {split_name}: X{X_norm.shape} Y{d["Y_log"].shape} guardados')

    Y_valid = datos['train']['Y_log'][datos['train']['M'] == 1]
    config = {
        'n_nodos': X_static.shape[0], 'features_dinamicas': FEATURES_DIN,
        'features_estaticas': FEATURES_STATIC, 'target': 'ih_trafico',
        'mask': 'mask_valid', 'transform': 'log1p', 'inverse': 'expm1',
        'lags': {'lag24': 96, 'lag7d': 672},
    }
    stats = {
        'x_features': stats_x,
        'y': {'mean_log': float(Y_valid.mean()), 'std_log': float(Y_valid.std()),
              'min_log': float(Y_valid.min()), 'max_log': float(Y_valid.max())},
        'cleaning': {'valid_range': [0, UMBRAL_MAX_Y],
                     'excluded': ['ih_trafico == -1', f'ih_trafico > {UMBRAL_MAX_Y}', 'ih_trafico < -1']},
    }
    with open(TENSOR_DIR / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    with open(TENSOR_DIR / 'stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    print('\n  Guardados: config.json, stats.json')

    print('\n' + '=' * 70)
    print('VERIFICACIÓN FINAL — archivos en tensors_final/')
    print('=' * 70)
    for f in sorted(TENSOR_DIR.iterdir()):
        if f.is_file():
            print(f'  {f.name:35s} ({f.stat().st_size/(1024*1024):>7.1f} MB)')


def cargar_split_para_modelo(split_name):
    """Dict listo para DatasetSimple/Completo/CompletoPlus (modelos/datasets.py)."""
    return dict(
        X=np.load(TENSOR_DIR / f'X_{split_name}_norm.npy'),
        Y=np.load(TENSOR_DIR / f'Y_{split_name}_log.npy'),
        M=np.load(TENSOR_DIR / f'M_{split_name}.npy'),
        Y_lag24=np.load(TENSOR_DIR / f'Y_{split_name}_lag24_log.npy'),
        M_lag24=np.load(TENSOR_DIR / f'M_{split_name}_lag24.npy'),
        Y_lag7d=np.load(TENSOR_DIR / f'Y_{split_name}_lag7d_log.npy'),
        M_lag7d=np.load(TENSOR_DIR / f'M_{split_name}_lag7d.npy'),
    )


if __name__ == '__main__':
    main()
