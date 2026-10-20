"""
crear_muestra_demo.py — Extrae del split de test la muestra que usa la
demo y la guarda en pipeline_v2_recon/demo/.

Escenarios: una semana normal (2–8 de junio de 2025) y otra con festivo
(23–29 de junio de 2025, San Juan el martes 24). Cada uno se guarda como
un .npz comprimido en float32 con 692 pasos: los 672 de la semana más 20
de contexto previo (start_offset 8 de DatasetCompletoPlus + ventana 12),
para que gat_plus pueda predecir desde el lunes a las 00:00. Las claves
del .npz son las de cargar_split_para_modelo(), así que el dict se puede
pasar tal cual a DatasetCompletoPlus.

Ejecutar desde la raíz del repositorio (necesita los tensores de 05).
"""

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import GRAPH_DIR, HORIZON, MODELS_DIR, TENSOR_DIR, WINDOW  # noqa: E402
from datos.io_tensores import cargar_split_para_modelo  # noqa: E402
from modelos.datasets import DatasetCompletoPlus  # noqa: E402

DEMO_DIR = MODELS_DIR.parent / 'demo'
PASOS_SEMANA = 7 * 96
START_OFFSET = 8  # DatasetCompletoPlus.start_offset
CONTEXTO = START_OFFSET + WINDOW + HORIZON - 1
ESCENARIOS = {
    'semana_normal': '2025-06-02 00:00:00',
    'semana_san_juan': '2025-06-23 00:00:00',
}


def recortar_semana(data: dict[str, np.ndarray], timestamps: np.ndarray,
                    inicio: str) -> tuple[dict[str, np.ndarray], int]:
    """Devuelve los arrays de la semana que empieza en `inicio` con
    CONTEXTO pasos previos, y el índice del primer paso en el split."""
    ts = pd.to_datetime(timestamps)
    i0 = int(np.where(ts == pd.Timestamp(inicio))[0][0])
    a, b = i0 - CONTEXTO, i0 + PASOS_SEMANA
    assert a >= 0 and b <= len(ts), f'{inicio}: la semana no cabe en test'
    assert (ts[b - 1] - ts[a]) == pd.Timedelta(minutes=15) * (b - a - 1), \
        f'{inicio}: hay huecos en el timeline'

    muestra = {k: np.ascontiguousarray(v[a:b]) for k, v in data.items()}
    muestra['timestamps'] = timestamps[a:b].astype('datetime64[ns]')
    return muestra, a


def comprobar_muestra(muestra: dict[str, np.ndarray], data: dict[str, np.ndarray],
                      a: int) -> None:
    """Comprueba que DatasetCompletoPlus produce sobre la muestra las
    mismas entradas (33 features) y objetivos que sobre el test completo."""
    ds_muestra = DatasetCompletoPlus(muestra, window=WINDOW, horizon=HORIZON)
    ds_test = DatasetCompletoPlus(data, window=WINDOW, horizon=HORIZON)
    assert len(ds_muestra) == PASOS_SEMANA, len(ds_muestra)
    for j in (0, PASOS_SEMANA // 2, PASOS_SEMANA - 1):
        xm, ym, mm = ds_muestra[j]
        xt, yt, mt = ds_test[a + j]
        assert xm.shape[-1] == 33
        assert all((p == q).all() for p, q in ((xm, xt), (ym, yt), (mm, mt)))


def main() -> None:
    """Genera los .npz de los escenarios y copia los ficheros comunes."""
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    data = cargar_split_para_modelo('test')
    timestamps = np.load(TENSOR_DIR / 'timestamps_test.npy')

    for nombre, inicio in ESCENARIOS.items():
        muestra, a = recortar_semana(data, timestamps, inicio)
        comprobar_muestra(muestra, data, a)
        ruta = DEMO_DIR / f'{nombre}.npz'
        np.savez_compressed(ruta, **muestra)
        ts = pd.to_datetime(muestra['timestamps'])
        print(f'  {ruta.name}: {len(ts)} pasos ({ts[0]} → {ts[-1]}) | '
              f'{ruta.stat().st_size / 1e6:.2f} MB | cobertura M={muestra["M"].mean() * 100:.1f}%')

    comunes = {
        GRAPH_DIR / 'node_order_1035.csv': 'node_order.csv',
        GRAPH_DIR / 'edge_index_final.npy': 'edge_index_final.npy',
        TENSOR_DIR / 'stats.json': 'stats.json',
        TENSOR_DIR / 'config.json': 'config.json',
    }
    for origen, destino in comunes.items():
        shutil.copyfile(origen, DEMO_DIR / destino)
        print(f'  {destino}: {(DEMO_DIR / destino).stat().st_size / 1e3:.1f} KB')

    total = sum(f.stat().st_size for f in DEMO_DIR.iterdir() if f.is_file())
    print(f'\n  Total en {DEMO_DIR}: {total / 1e6:.2f} MB')


if __name__ == '__main__':
    main()
