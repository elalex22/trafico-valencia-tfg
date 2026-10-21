"""Tests de calidad de los datos: los hallazgos de la memoria y de la
reconstrucción.

- Régimen corrupto de las espiras (octubre 2024 – mayo 2025), medido con el
  porcentaje de valores positivos por mes, y fuera de los splits.
- Congelación de la fuente del 1 al 12 de junio de 2025 en test.
- Baseline de persistencia en test: 38,30 veh/h.

Los que necesitan datos generados por 01 y 05 (fuera del repositorio) se
saltan si faltan, para que la CI funcione solo con lo versionado.
"""
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from config import HORIZON, PROC, RESULTADOS_DIR, TENSOR_DIR, TROZOS_POR_SPLIT
from modelos.entrenamiento import persistencia_mae

ESPIRAS = PROC / 'espires_trafico.parquet'
TENSORES_TEST = [TENSOR_DIR / f for f in ('Y_test_log.npy', 'M_test.npy', 'timestamps_test.npy')]

MESES_CORRUPTOS = pd.period_range('2024-10', '2025-05', freq='M')
UMBRAL_POSITIVOS = 0.45     # la memoria (p. 29): "inferiores al 45 %"
UMBRAL_IGUALES = 0.99       # fracción de sensores con el mismo valor que el paso anterior
UMBRAL_CONGELACION_H = 24   # las rachas normales no pasan de unas horas (la mayor, 17,75 h)
BASELINE_TEST = 38.30

falta_espiras = pytest.mark.skipif(not ESPIRAS.exists(),
                                   reason='Falta processed/espires_trafico.parquet (01_extraccion.py)')
faltan_tensores = pytest.mark.skipif(not all(p.exists() for p in TENSORES_TEST),
                                     reason='Faltan los tensores de test (05_pipeline_tensores.py)')


# ── Régimen corrupto (octubre 2024 – mayo 2025) ──────────────────────────

@pytest.fixture(scope='module')
def positivos_por_mes() -> pd.Series:
    """Fracción de lecturas de espiras con ih > 0, por mes (lectura por lotes)."""
    base = np.datetime64('2020-01', 'M')
    pos, tot = np.zeros(240), np.zeros(240)
    for lote in pq.ParquetFile(ESPIRAS).iter_batches(batch_size=10_000_000,
                                                     columns=['timestamp', 'ih_trafico']):
        mes = (lote.column(0).to_numpy().astype('datetime64[M]') - base).astype(int)
        ih = lote.column(1).to_numpy(zero_copy_only=False)
        tot += np.bincount(mes, minlength=240)
        pos += np.bincount(mes, weights=ih > 0, minlength=240)
    hay = np.flatnonzero(tot)
    meses = pd.PeriodIndex((base + hay).astype(str), freq='M')
    return pd.Series(pos[hay] / tot[hay], index=meses)


@falta_espiras
def test_regimen_corrupto_detectado(positivos_por_mes: pd.Series) -> None:
    corruptos = positivos_por_mes.index[positivos_por_mes < UMBRAL_POSITIVOS]
    assert list(corruptos) == list(MESES_CORRUPTOS)
    assert (positivos_por_mes[pd.PeriodIndex(['2024-11', '2024-12'], freq='M')] == 0).all()
    limpios = positivos_por_mes.drop(MESES_CORRUPTOS)
    assert limpios.between(0.60, 0.80).all(), limpios[~limpios.between(0.60, 0.80)]


def test_meses_corruptos_fuera_de_los_splits() -> None:
    """Ningún trozo de train, val o test (sin contar el buffer de lags)
    toca un mes corrupto. Solo usa config.py: corre también en la CI."""
    for split, trozos in TROZOS_POR_SPLIT.items():
        for t in trozos:
            meses = pd.period_range(t['start'], t['end'], freq='M')
            assert not set(meses) & set(MESES_CORRUPTOS), (split, t)


@faltan_tensores
def test_timestamps_de_test_fuera_del_regimen_corrupto() -> None:
    ts = pd.DatetimeIndex(np.load(TENSOR_DIR / 'timestamps_test.npy'))
    assert not ts.to_period('M').isin(MESES_CORRUPTOS).any()


# ── Congelación del 1 al 12 de junio de 2025 ─────────────────────────────

def rachas_congeladas(Y: np.ndarray, M: np.ndarray, ts: pd.DatetimeIndex) -> pd.DataFrame:
    """Rachas de pasos en los que al menos UMBRAL_IGUALES de los sensores
    con dato repiten el valor del paso anterior."""
    ambos = (M[1:] > 0) & (M[:-1] > 0)
    iguales = (ambos & (Y[1:] == Y[:-1])).sum(1) / np.maximum(ambos.sum(1), 1)
    congelado = (iguales >= UMBRAL_IGUALES) & (ambos.sum(1) >= 100)
    d = np.diff(np.r_[0, congelado.astype(int), 0])
    ini, fin = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return pd.DataFrame({'inicio': ts[ini], 'fin': ts[fin], 'horas': (fin - ini) * 0.25})


@faltan_tensores
def test_congelacion_de_junio_detectada() -> None:
    Y = np.load(TENSOR_DIR / 'Y_test_log.npy', mmap_mode='r')
    M = np.load(TENSOR_DIR / 'M_test.npy', mmap_mode='r')
    ts = pd.DatetimeIndex(np.load(TENSOR_DIR / 'timestamps_test.npy'))
    largas = rachas_congeladas(Y, M, ts).query('horas > @UMBRAL_CONGELACION_H')
    assert len(largas) == 1, largas
    racha = largas.iloc[0]
    assert racha.inicio.normalize() == pd.Timestamp('2025-06-01')
    assert racha.fin.normalize() == pd.Timestamp('2025-06-12')
    assert racha.horas / 24 == pytest.approx(11.3, abs=0.1)


# ── Baseline de persistencia ─────────────────────────────────────────────

def test_baseline_json_test() -> None:
    """resultados/baseline.json (versionado) da 38,30 veh/h en test."""
    with open(RESULTADOS_DIR / 'baseline.json', encoding='utf-8') as f:
        assert round(json.load(f)['test']['mae_real'], 2) == BASELINE_TEST


@faltan_tensores
def test_baseline_persistencia_test() -> None:
    """La persistencia recalculada sobre los tensores de test da 38,30 veh/h."""
    Y = np.load(TENSOR_DIR / 'Y_test_log.npy')
    M = np.load(TENSOR_DIR / 'M_test.npy')
    mae, _ = persistencia_mae(Y, M, horizon=HORIZON)
    assert round(mae, 2) == BASELINE_TEST
