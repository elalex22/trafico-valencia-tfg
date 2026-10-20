"""
inferencia_demo.py — Inferencia de gat_plus (GAT+Plus) para la demo, sobre
la muestra de pipeline_v2_recon/demo/ (la genera app/crear_muestra_demo.py).

Cada predicción usa una sola ventana: la de DatasetCompletoPlus para ese
instante (12 pasos de 15 min, 33 features, más 8 pasos de contexto para
los lags cortos). No necesita los tensores completos ni importa config.py,
que crea carpetas de datos al importarse.

Un "instante" es la hora que se predice (t+1): el modelo ve los datos
hasta 15 minutos antes.

Uso típico:
    modelo = cargar_modelo()
    semana = cargar_semana('semana_normal')
    resultado = predecir(modelo, semana, '2025-09-15 08:00')
    resultado.tabla       # DataFrame con una fila por sensor
    resultado.verificable # False si la fuente no tiene datos en ese instante
"""

import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch

RAIZ_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ_REPO))

from modelos.arquitecturas import GATCompletoPlus  # noqa: E402
from modelos.datasets import DatasetCompletoPlus  # noqa: E402

DEMO_DIR = RAIZ_REPO / 'pipeline_v2_recon' / 'demo'
MODELO_PT = RAIZ_REPO / 'pipeline_v2_recon' / 'models' / 'gat_plus.pt'
SEMANAS = ('semana_normal', 'semana_san_juan')
WINDOW = 12   # config.WINDOW
HORIZON = 1   # config.HORIZON
N_FEATURES = 33  # DatasetCompletoPlus
AVISO_SIN_DATOS = 'Sin datos de la fuente oficial: predicción no verificable'


@dataclass
class Semana:
    """Una semana de la muestra lista para predecir.

    Atributos:
        nombre: 'semana_normal' o 'semana_san_juan'.
        dataset: DatasetCompletoPlus sobre los 692 pasos del .npz.
        instantes: los 672 instantes que se pueden predecir; el de
            posición j corresponde a dataset[j].
        nodos: id_sensor, lat y lon de los 1.035 nodos, en el orden de
            los tensores.
        edge_index: grafo físico (2, E) con el que se entrenó gat_plus.
    """
    nombre: str
    dataset: DatasetCompletoPlus
    instantes: pd.DatetimeIndex
    nodos: pd.DataFrame
    edge_index: torch.Tensor


@dataclass
class Prediccion:
    """Resultado de predecir() para un instante.

    Atributos:
        instante: hora predicha.
        tabla: una fila por sensor con id_sensor, lat, lon,
            prediccion_veh_h, real_veh_h, error (|predicción − real|) y
            persistencia_veh_h (Y[t], el último valor antes del instante,
            como en 07_baseline_persistencia.py); real_veh_h y error son
            NaN donde la máscara no es válida.
        verificable: False si la máscara está vacía en ese instante (no
            hay ningún valor real con el que comparar).
        aviso: AVISO_SIN_DATOS si no es verificable; None si lo es.
    """
    instante: pd.Timestamp
    tabla: pd.DataFrame
    verificable: bool
    aviso: str | None


def cargar_modelo() -> GATCompletoPlus:
    """Carga gat_plus con sus pesos guardados, en CPU y en modo eval."""
    modelo = GATCompletoPlus(window=WINDOW, n_features=N_FEATURES)
    modelo.load_state_dict(torch.load(MODELO_PT, map_location='cpu', weights_only=True))
    modelo.eval()
    return modelo


@lru_cache(maxsize=1)
def _cargar_comunes() -> tuple[pd.DataFrame, torch.Tensor]:
    """Lee los nodos y el grafo, comunes a las dos semanas."""
    nodos = pd.read_csv(DEMO_DIR / 'node_order.csv')
    nodos = nodos.sort_values('idx')[['id_sensor', 'lat', 'lon']].reset_index(drop=True)
    edge_index = torch.from_numpy(np.load(DEMO_DIR / 'edge_index_final.npy')).long()
    return nodos, edge_index


def cargar_semana(nombre: str) -> Semana:
    """Carga 'semana_normal' (15–21 de septiembre de 2025) o
    'semana_san_juan' (23–29 de junio de 2025) desde la muestra."""
    if nombre not in SEMANAS:
        raise ValueError(f'Semana desconocida: {nombre!r}. Opciones: {SEMANAS}')
    with np.load(DEMO_DIR / f'{nombre}.npz') as npz:
        datos = {k: npz[k] for k in npz.files}

    dataset = DatasetCompletoPlus(datos, window=WINDOW, horizon=HORIZON)
    contexto = dataset.start_offset + WINDOW + HORIZON - 1
    instantes = pd.DatetimeIndex(datos['timestamps'][contexto:])
    assert len(instantes) == len(dataset), (len(instantes), len(dataset))

    nodos, edge_index = _cargar_comunes()
    return Semana(nombre, dataset, instantes, nodos, edge_index)


def instantes_disponibles(semana: Semana) -> pd.DatetimeIndex:
    """Devuelve los instantes que se pueden predecir (672, cada 15 min,
    de lunes 00:00 a domingo 23:45)."""
    return semana.instantes


def _posicion(semana: Semana, paso: int | str | pd.Timestamp) -> int:
    """Convierte `paso` (posición 0–671 o fecha y hora) en la posición
    dentro de la semana."""
    if isinstance(paso, (int, np.integer)):
        if not 0 <= paso < len(semana.instantes):
            raise IndexError(f'paso {paso} fuera de 0–{len(semana.instantes) - 1}')
        return int(paso)
    instante = pd.Timestamp(paso)
    if instante not in semana.instantes:
        raise KeyError(f'{instante} no está en {semana.nombre} '
                       f'({semana.instantes[0]} – {semana.instantes[-1]}, cada 15 min)')
    return int(semana.instantes.get_loc(instante))


def predecir(modelo: GATCompletoPlus, semana: Semana,
             paso: int | str | pd.Timestamp) -> Prediccion:
    """Predice la intensidad (veh/h) de los 1.035 sensores en un instante.

    `paso` es la posición del instante en instantes_disponibles() o la
    fecha y hora (p. ej. '2025-09-15 08:00'). Las predicciones y los
    valores reales se pasan de log1p a veh/h con expm1, como en
    modelos/inferencia.py.
    """
    j = _posicion(semana, paso)
    x, y, m = semana.dataset[j]
    t_ahora = j + semana.dataset.start_offset + WINDOW - 1
    persistencia = np.expm1(np.clip(semana.dataset.Y[t_ahora], 0, None))
    with torch.no_grad():
        pred_log = modelo(x.unsqueeze(0), semana.edge_index)[0]

    prediccion = torch.expm1(pred_log.clamp(min=0)).numpy()
    valido = m.numpy() > 0
    real = np.where(valido, torch.expm1(y).numpy(), np.nan)

    tabla = semana.nodos.copy()
    tabla['prediccion_veh_h'] = prediccion
    tabla['real_veh_h'] = real
    tabla['error'] = np.abs(prediccion - real)
    tabla['persistencia_veh_h'] = persistencia

    verificable = bool(valido.any())
    return Prediccion(
        instante=semana.instantes[j],
        tabla=tabla,
        verificable=verificable,
        aviso=None if verificable else AVISO_SIN_DATOS,
    )
