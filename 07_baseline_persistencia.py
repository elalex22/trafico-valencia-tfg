"""
07_baseline_persistencia.py — MAE/RMSE de la persistencia (Y[t+1]=Y[t]),
escala real (veh/h), en los 3 splits.

Requiere: 05_pipeline_tensores.py ya ejecutado.
Genera: resultados/baseline.json
"""
import json
import numpy as np

from config import TENSOR_DIR, RESULTADOS_DIR, HORIZON
from modelos.entrenamiento import persistencia_mae


def main():
    print('=' * 70)
    print('07 — BASELINE DE PERSISTENCIA')
    print('=' * 70)

    resultados = {}
    for split in ['train', 'val', 'test']:
        Y = np.load(TENSOR_DIR / f'Y_{split}_log.npy')
        M = np.load(TENSOR_DIR / f'M_{split}.npy')
        mae, rmse = persistencia_mae(Y, M, horizon=HORIZON)
        resultados[split] = {'mae_real': mae, 'rmse_real': rmse}
        print(f'  {split:6s}: MAE={mae:.2f} veh/h  RMSE={rmse:.2f} veh/h')

    with open(RESULTADOS_DIR / 'baseline.json', 'w') as f:
        json.dump(resultados, f, indent=2)

    print(f'\n  Guardado: resultados/baseline.json')
    print(f'\n  Nota: el MAE de persistencia en test debería rondar los 37-38 veh/h.')
    print(f'  Una desviación grande respecto a ese rango indicaría revisar 04/05.')


if __name__ == '__main__':
    main()
