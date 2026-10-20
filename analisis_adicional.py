"""
analisis_adicional.py — MAE de gat_plus y de la persistencia en test sobre
tres subconjuntos de instantes:

  a) test completo;
  b) solo los instantes :15 (la intensidad es horaria y cambia a y cuarto;
     en :00, :30 y :45 el valor real es el anterior);
  c) solo los :15, sin los días congelados del 1 al 12 de junio de 2025.

Las dos métricas se calculan sobre los mismos objetivos (los de
DatasetCompletoPlus, desde el paso 20 del test) y como evaluate() de
modelos/entrenamiento.py: MAE en veh/h sobre las celdas con máscara
válida. La persistencia es la de 07_baseline_persistencia.py
(Ŷ[t+1] = Y[t]).

Requiere: 05_pipeline_tensores.py ya ejecutado. Sin reentrenar.
Genera: demo/analisis_adicional.json
"""

import json
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import GRAPH_DIR, HORIZON, MODELS_DIR, TENSOR_DIR, WINDOW
from datos.io_tensores import cargar_split_para_modelo
from modelos.arquitecturas import GATCompletoPlus
from modelos.datasets import DatasetCompletoPlus

DEMO_DIR = MODELS_DIR.parent / 'demo'
CONGELACION = ('2025-06-01', '2025-06-12')  # días completos excluidos en (c)


def errores_por_paso(data: dict[str, np.ndarray]) -> pd.DataFrame:
    """Suma de errores absolutos (veh/h) de gat_plus y de la persistencia y
    número de celdas válidas, por instante objetivo."""
    ds = DatasetCompletoPlus(data, window=WINDOW, horizon=HORIZON)
    ei = torch.from_numpy(np.load(GRAPH_DIR / 'edge_index_final.npy')).long()
    modelo = GATCompletoPlus(window=WINDOW, n_features=33)
    modelo.load_state_dict(torch.load(MODELS_DIR / 'gat_plus.pt', map_location='cpu',
                                      weights_only=True))
    modelo.eval()

    ae_gat = []
    with torch.no_grad():
        for x, y, m in DataLoader(ds, batch_size=16, shuffle=False):
            pred = torch.expm1(modelo(x, ei).clamp(min=0))
            ae_gat.append(((pred - torch.expm1(y)).abs() * m).sum(1).numpy())
    ae_gat = np.concatenate(ae_gat)

    primero = ds.start_offset + WINDOW + HORIZON - 1
    objetivo = np.arange(primero, primero + len(ds))
    Y, M = data['Y'], data['M']
    real = np.expm1(np.clip(Y[objetivo], 0, None))
    anterior = np.expm1(np.clip(Y[objetivo - HORIZON], 0, None))
    ae_pers = (np.abs(anterior - real) * M[objetivo]).sum(1)

    ts = pd.DatetimeIndex(np.load(TENSOR_DIR / 'timestamps_test.npy'))
    return pd.DataFrame({'instante': ts[objetivo], 'ae_gat': ae_gat,
                         'ae_pers': ae_pers, 'n': M[objetivo].sum(1)})


def resumen(df: pd.DataFrame) -> dict[str, Any]:
    """MAE de los dos modelos y mejora de gat_plus en un subconjunto."""
    n = df['n'].sum()
    gat, pers = df['ae_gat'].sum() / n, df['ae_pers'].sum() / n
    return {'mae_gat_plus': round(float(gat), 2), 'mae_persistencia': round(float(pers), 2),
            'mejora_pct': round(float((pers - gat) / pers * 100), 2),
            'instantes': int(len(df)), 'celdas_validas': int(n)}


def main() -> None:
    """Calcula los tres casos y guarda demo/analisis_adicional.json."""
    df = errores_por_paso(cargar_split_para_modelo('test'))
    cuarto = df['instante'].dt.minute == 15
    dia = df['instante'].dt.normalize()
    congelado = dia.between(pd.Timestamp(CONGELACION[0]), pd.Timestamp(CONGELACION[1]))

    casos = {
        'a_test_completo': resumen(df),
        'b_solo_15': resumen(df[cuarto]),
        'c_solo_15_sin_congelacion': resumen(df[cuarto & ~congelado]),
    }
    salida = {
        'unidades': 'veh/h; MAE sobre las celdas con máscara válida',
        'casos': casos,
        'nota': ('Las dos métricas se calculan sobre los mismos objetivos: los de '
                 'DatasetCompletoPlus (test desde 2025-06-01 05:00, 20 pasos de contexto). '
                 'Por eso la persistencia del caso (a) difiere ligeramente de la de '
                 'baseline.json (38,30), que empieza en el segundo paso del test. '
                 'Instantes :15: los únicos en los que llega el valor horario nuevo. '
                 'Congelación: se excluyen los días completos del 1 al 12 de junio de 2025.'),
    }
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    with open(DEMO_DIR / 'analisis_adicional.json', 'w', encoding='utf-8') as f:
        json.dump(salida, f, indent=2, ensure_ascii=False)

    print(f'  {"Caso":<28} {"GAT+Plus":>9} {"Persist.":>9} {"Mejora":>8} {"Instantes":>10}')
    for nombre, c in casos.items():
        print(f'  {nombre:<28} {c["mae_gat_plus"]:>9.2f} {c["mae_persistencia"]:>9.2f} '
              f'{c["mejora_pct"]:>7.1f}% {c["instantes"]:>10}')


if __name__ == '__main__':
    main()
