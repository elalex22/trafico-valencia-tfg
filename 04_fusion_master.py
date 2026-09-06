"""
04_fusion_master.py — Fusión de las 6 fuentes sobre el timeline, por trozos
con buffer de 7 días (para poder calcular Y_lag7d desde el primer timestep
real de cada trozo).

Requiere: 01_extraccion.py, 02_construccion_grafo.py, 03_asignaciones_espaciales.py

Genera: master_final/master_{split}.parquet (train, val, test)
"""

import gc
import time
import numpy as np
import pandas as pd
import holidays

from config import (PROC, MASTER_DIR, TROZOS_POR_SPLIT,
                     MAX_GAP_TIMESTEPS, RACHA_CEROS, LAG_24H, LAG_7D)
from datos.utils_datos import (normalize_id, load_and_filter, aggregate_15min,
                                aplicar_fix_ceros_extendidos)


def cargar_asignaciones():
    asig = pd.read_csv(MASTER_DIR / 'asignaciones_espaciales.csv')
    asig['id_tramo'] = asig['id_tramo'].apply(normalize_id)
    return asig.sort_values('idx').reset_index(drop=True)


def construir_festivos(year_min, year_max):
    fest = set()
    for year in range(year_min, year_max + 1):
        for d in holidays.Spain(years=year, subdiv='VC'):
            fest.add(pd.Timestamp(d).date())
    return fest


def construir_base(buf_start, end, node_ids, asig):
    ts = pd.date_range(start=buf_start, end=end, freq='15min')
    base = pd.DataFrame({'timestamp': np.repeat(ts, len(node_ids))})
    base['id_sensor'] = np.tile(node_ids, len(ts))
    base = base.merge(
        asig[['id_sensor', 'idx', 'lat', 'lon', 'id_tramo',
              'id_bici_cercano', 'id_valen_cercano', 'id_ruido_cercano']],
        on='id_sensor', how='left')
    return base


# ── Fusión por fuente (idéntico patrón al confirmado en el bloque de fusión) ──

def fusionar_ih_trafico(base, buf_start, end):
    df = load_and_filter(PROC / 'espires_trafico.parquet', buf_start, end)
    df = df[df['id_sensor'].isin(base['id_sensor'].unique())]
    df = aggregate_15min(df, 'timestamp', 'id_sensor', 'ih_trafico', 'mean')
    base = base.merge(df, on=['timestamp', 'id_sensor'], how='left')
    base['mask_ih_trafico'] = base['ih_trafico'].notna().astype(np.int8)
    return base


def fusionar_estado(base, buf_start, end):
    df_est = load_and_filter(PROC / 'estat_trafic.parquet', buf_start, end)
    df_est['id_tramo'] = df_est['id_tramo'].apply(normalize_id)
    tramos_usados = set(base['id_tramo'].dropna().unique())
    df_est = df_est[df_est['id_tramo'].isin(tramos_usados)]
    df_est = aggregate_15min(df_est, 'timestamp', 'id_tramo', 'estado', 'mean')
    esp_tramo = base[['id_sensor', 'id_tramo']].drop_duplicates().dropna(subset=['id_tramo'])
    df_est = df_est.merge(esp_tramo, on='id_tramo', how='inner')[['timestamp', 'id_sensor', 'estado']]
    base = base.merge(df_est, on=['timestamp', 'id_sensor'], how='left')
    base['mask_estado'] = base['estado'].notna().astype(np.int8)
    return base


def fusionar_valenbici(base, buf_start, end):
    df_v = load_and_filter(PROC / 'valenbici.parquet', buf_start, end)
    df_v = df_v.groupby(['timestamp', 'id_estacion'], as_index=False).agg(
        available=('available', 'mean'), free=('free', 'mean'))
    esp_valen = base[['id_sensor', 'id_valen_cercano']].drop_duplicates().rename(
        columns={'id_valen_cercano': 'id_estacion'})
    df_v = df_v.merge(esp_valen, on='id_estacion', how='inner')[['timestamp', 'id_sensor', 'available', 'free']]
    base = base.merge(df_v, on=['timestamp', 'id_sensor'], how='left')
    base['mask_valenbici'] = base['available'].notna().astype(np.int8)
    return base


def fusionar_bici(base, buf_start, end):
    """CORREGIDO tras revisión a fondo: el apartado 4.3 de la memoria dice
    'Espiras tráfico / bici: valores -1 (fallo de sensor) → clip(lower=0) +
    máscara binaria' para AMBAS fuentes. El target (tráfico) ya lo tenía
    resuelto (racha de ceros en 04 + limpiar_y_log en 05), pero ih_bici
    (una feature, no el target) solo tenía mask_ih_bici=notna() — que no
    detecta -1 (no es NaN). Se añade aquí el mismo criterio: -1 -> mask=0
    + valor a 0. Como 05_pipeline_tensores.py ya usa MASK_POR_FEATURE para
    excluir de la media/std cualquier celda con máscara=0, basta con
    corregir la máscara aquí — no hace falta tocar 05."""
    df_b = load_and_filter(PROC / 'espires_bici.parquet', buf_start, end)
    df_b = aggregate_15min(df_b, 'timestamp', 'id_sensor', 'ih_bici', 'mean')
    df_b = df_b.rename(columns={'id_sensor': 'id_bici_cercano'})
    esp_bici = base[['id_sensor', 'id_bici_cercano']].drop_duplicates()
    df_b = df_b.merge(esp_bici, on='id_bici_cercano', how='inner')[['timestamp', 'id_sensor', 'ih_bici']]
    base = base.merge(df_b, on=['timestamp', 'id_sensor'], how='left')

    # -1 (fallo de sensor) -> inválido, igual que el target. NaN (sin dato)
    # también sigue siendo inválido, como antes.
    base['mask_ih_bici'] = (base['ih_bici'].notna() & (base['ih_bici'] >= 0)).astype(np.int8)
    base['ih_bici'] = base['ih_bici'].clip(lower=0)
    return base


def fusionar_ruido(base, buf_start, end):
    df_r = pd.read_parquet(PROC / 'ruido.parquet',
                            columns=['timestamp', 'id_sensor_ruido', 'laeq', 'laeq_n'])
    df_r['timestamp'] = pd.to_datetime(df_r['timestamp'])
    df_r = df_r[(df_r['timestamp'] >= buf_start) & (df_r['timestamp'] <= end)].copy()
    df_r['timestamp'] = df_r['timestamp'].dt.floor('15min')
    df_r = df_r.groupby(['timestamp', 'id_sensor_ruido'], as_index=False).agg(
        laeq=('laeq', 'mean'), laeq_n=('laeq_n', 'mean'))
    df_r = df_r.rename(columns={'id_sensor_ruido': 'id_ruido_cercano'})
    esp_ruido = base[['id_sensor', 'id_ruido_cercano']].drop_duplicates()
    df_r = df_r.merge(esp_ruido, on='id_ruido_cercano', how='inner')[['timestamp', 'id_sensor', 'laeq', 'laeq_n']]
    base = base.merge(df_r, on=['timestamp', 'id_sensor'], how='left')
    base['mask_ruido'] = base['laeq'].notna().astype(np.int8)
    return base


def fusionar_meteo(base, buf_start, end):
    df_m = pd.read_parquet(PROC / 'meteo_diario.parquet')
    df_m['fecha'] = pd.to_datetime(df_m['fecha']).dt.date
    df_m = df_m[['fecha', 'tmed', 'prec']]
    base['fecha'] = base['timestamp'].dt.date
    base = base.merge(df_m, on='fecha', how='left')
    base['mask_meteo'] = base['tmed'].notna().astype(np.int8)
    return base.drop(columns=['fecha'])


def anadir_features_temporales(df, festivos_set):
    df['hora'] = df['timestamp'].dt.hour
    df['minuto'] = df['timestamp'].dt.minute
    dia_semana = df['timestamp'].dt.dayofweek.astype(np.int8)
    df['es_finde'] = (dia_semana >= 5).astype(np.int8)

    df['hora_sin'] = np.sin(2 * np.pi * df['hora'] / 24).astype('float32')
    df['hora_cos'] = np.cos(2 * np.pi * df['hora'] / 24).astype('float32')
    df['minuto_sin'] = np.sin(2 * np.pi * df['minuto'] / 60).astype('float32')
    df['minuto_cos'] = np.cos(2 * np.pi * df['minuto'] / 60).astype('float32')
    df['dia_semana_sin'] = np.sin(2 * np.pi * dia_semana / 7).astype('float32')
    df['dia_semana_cos'] = np.cos(2 * np.pi * dia_semana / 7).astype('float32')
    df = df.drop(columns=['hora', 'minuto'])

    fechas = df['timestamp'].dt.date
    df['es_festivo'] = fechas.isin(festivos_set).astype(np.int8)
    df['es_fallas'] = ((df['timestamp'].dt.month == 3) &
                        (df['timestamp'].dt.day.between(1, 19))).astype(np.int8)
    return df


def aplicar_racha_y_mask(df, node_ids):
    """Regla de 6_limpieza_final.py: rachas de ih_trafico=0 (dato observado,
    no NaN) de >= RACHA_CEROS timesteps seguidos -> sensor roto, mask=0.
    Se calcula ANTES de interpolar: opera sobre ceros reales, no sobre huecos."""
    T = df['timestamp'].nunique()
    N = len(node_ids)
    assert len(df) == T * N, f'Grid incompleto: {len(df)} != {T*N}'

    Y = df['ih_trafico'].fillna(0).values.reshape(T, N).astype(np.float32)
    M = df['mask_ih_trafico'].values.reshape(T, N).astype(np.int8)
    M_fixed, n_marcados = aplicar_fix_ceros_extendidos(Y, M, racha_umbral=RACHA_CEROS)
    df['mask_racha'] = M_fixed.reshape(-1).astype(np.int8)
    return df, int(n_marcados.sum())


def interpolar_y_mask_valid(df):
    """Interpolación lineal de huecos <= MAX_GAP_TIMESTEPS (2h). Los valores
    interpolados cuentan como válidos en mask_valid (no hay forma de
    distinguirlos después salvo por mask_ih_trafico, que sí conserva el
    original)."""
    df = df.sort_values(['id_sensor', 'timestamp'])
    df['ih_trafico'] = df.groupby('id_sensor')['ih_trafico'].transform(
        lambda x: x.interpolate(method='linear', limit=MAX_GAP_TIMESTEPS,
                                 limit_direction='forward'))
    df['mask_valid'] = (df['ih_trafico'].notna() & (df['mask_racha'] == 1)).astype(np.int8)
    return df.sort_values(['timestamp', 'id_sensor']).reset_index(drop=True)


def calcular_lags(df):
    """Lags en veh/h crudos (la conversión a log1p se hace en 05, junto con
    la limpieza de outliers, para aplicar la misma regla a Y/Y_lag24/Y_lag7d)."""
    df = df.sort_values(['id_sensor', 'timestamp'])
    g = df.groupby('id_sensor')
    df['Y_lag24'] = g['ih_trafico'].shift(LAG_24H)
    df['M_lag24'] = g['mask_valid'].shift(LAG_24H)
    df['Y_lag7d'] = g['ih_trafico'].shift(LAG_7D)
    df['M_lag7d'] = g['mask_valid'].shift(LAG_7D)

    df['Y_lag24'] = df['Y_lag24'].fillna(0).astype('float32')
    df['M_lag24'] = df['M_lag24'].fillna(0).astype(np.int8)
    df['Y_lag7d'] = df['Y_lag7d'].fillna(0).astype('float32')
    df['M_lag7d'] = df['M_lag7d'].fillna(0).astype(np.int8)
    return df.sort_values(['timestamp', 'id_sensor']).reset_index(drop=True)


def procesar_trozo(trozo, asig, festivos_set, node_ids):
    start = pd.Timestamp(trozo['start'])
    end = pd.Timestamp(trozo['end'])
    buf_start = pd.Timestamp(trozo['buf_start'])
    print(f'  Trozo: buffer {buf_start} → real {start} → {end}')
    t0 = time.time()

    base = construir_base(buf_start, end, node_ids, asig)
    base = fusionar_ih_trafico(base, buf_start, end)
    base = fusionar_estado(base, buf_start, end)
    base = fusionar_valenbici(base, buf_start, end)
    base = fusionar_bici(base, buf_start, end)
    base = fusionar_ruido(base, buf_start, end)
    base = fusionar_meteo(base, buf_start, end)
    base = anadir_features_temporales(base, festivos_set)
    base, n_racha = aplicar_racha_y_mask(base, node_ids)
    base = interpolar_y_mask_valid(base)
    base = calcular_lags(base)

    df_final = base[(base['timestamp'] >= start) & (base['timestamp'] <= end)].reset_index(drop=True)
    print(f'    {len(df_final)//len(node_ids):,} timesteps reales | '
          f'{n_racha:,} celdas marcadas por racha de ceros | {time.time()-t0:.0f}s')
    return df_final


def main():
    asig = cargar_asignaciones()
    node_ids = asig['id_sensor'].tolist()
    print(f'Nodos: {len(node_ids)}')

    todas_fechas = [pd.Timestamp(t[k]) for trozos in TROZOS_POR_SPLIT.values()
                     for t in trozos for k in ('buf_start', 'end')]
    festivos_set = construir_festivos(min(todas_fechas).year, max(todas_fechas).year)

    ESPERADO_T = {'train': 42460, 'val': 9312, 'test': 11040}

    for split_name, trozos in TROZOS_POR_SPLIT.items():
        print('\n' + '=' * 70)
        print(f'SPLIT: {split_name} ({len(trozos)} trozo(s))')
        print('=' * 70)

        partes = [procesar_trozo(t, asig, festivos_set, node_ids) for t in trozos]
        df_split = pd.concat(partes, ignore_index=True).sort_values(['timestamp', 'idx']).reset_index(drop=True)

        T_real = df_split['timestamp'].nunique()
        cobertura = df_split['mask_valid'].mean() * 100
        esperado = ESPERADO_T[split_name]
        check = '✅' if T_real == esperado else f'⚠️ ESPERABA {esperado:,} — revisar trozos/festivos'
        print(f'\n  {split_name}: T={T_real:,} {check} | cobertura mask_valid={cobertura:.2f}%')

        df_split.to_parquet(MASTER_DIR / f'master_{split_name}.parquet', index=False, compression='snappy')
        print(f'  Guardado: master_{split_name}.parquet ({len(df_split):,} filas)')
        del partes, df_split
        gc.collect()


if __name__ == '__main__':
    main()
