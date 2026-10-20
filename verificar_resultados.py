"""
verificar_resultados.py — Evalúa en test los 9 modelos ya entrenados
(pipeline_v2_recon/models/*.pt), sin reentrenar, con los datos y grafos
regenerados, y compara su MAE con el de resultados/entrenamiento.json.

Para no duplicar nada, reutiliza de 08_entrenar_modelos.py la tabla
CONFIGURACIONES (clase, dataset y grafo de cada modelo) y de
modelos/entrenamiento.py la función evaluate(), así que las métricas y
las unidades (veh/h) son exactamente las de 08.

Requiere: 05_pipeline_tensores.py y 06_grafo_enriquecido.py ya ejecutados.
Genera: demo/verificacion.json
"""

import importlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import GRAPH_DIR, HORIZON, MODELS_DIR, RESULTADOS_DIR, WINDOW
from datos.io_tensores import cargar_split_para_modelo
from modelos.entrenamiento import evaluate, persistencia_mae

# El nombre del módulo empieza por un dígito: no se puede usar `import`.
_entrenar = importlib.import_module('08_entrenar_modelos')
CONFIGURACIONES: list[dict[str, Any]] = _entrenar.CONFIGURACIONES

DEMO_DIR = MODELS_DIR.parent / 'demo'
BASELINE_TFG_MAE = 38.30  # MAE de persistencia en test (07_baseline_persistencia.py)


def cargar_edge_indices(device: torch.device) -> dict[str, torch.Tensor]:
    """Carga el grafo físico y el enriquecido, igual que 08."""
    return {
        'fisico': torch.from_numpy(
            np.load(GRAPH_DIR / 'edge_index_final.npy')).long().to(device),
        'enriquecido': torch.from_numpy(
            np.load(GRAPH_DIR / 'edge_index_enriched.npy')).long().to(device),
    }


def evaluar_modelo(cfg: dict[str, Any], data_test: dict[str, np.ndarray],
                   edge_indices: dict[str, torch.Tensor],
                   device: torch.device) -> dict[str, float]:
    """Construye el modelo con su clase y n_features, carga su checkpoint
    y devuelve MAE/RMSE en test (veh/h), como al final de entrenar()."""
    ds_test = cfg['dataset_cls'](data_test, window=WINDOW, horizon=HORIZON)
    n_feats = ds_test[0][0].shape[-1]

    if cfg['con_ventana']:
        model = cfg['clase'](window=WINDOW, n_features=n_feats).to(device)
    else:
        model = cfg['clase'](n_features=n_feats).to(device)

    ckpt = MODELS_DIR / f'{cfg["nombre"]}.pt'
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))

    edge_index = edge_indices[cfg['grafo']] if cfg['grafo'] else None
    is_gcn = cfg['grafo'] is not None
    loader = DataLoader(ds_test, batch_size=cfg['batch_size'], shuffle=False, num_workers=0)
    return evaluate(model, loader, device, edge_index, is_gcn)


def construir_tabla(resultados: dict[str, dict[str, float]],
                    referencia: dict[str, dict[str, float]],
                    baseline_mae: float) -> list[dict[str, Any]]:
    """Una fila por modelo: MAE guardado, MAE actual, diferencia, RMSE
    actual y mejora del MAE actual frente a la persistencia (%)."""
    filas = []
    for nombre, r in resultados.items():
        mae_guardado = referencia[nombre]['mae_real']
        filas.append({
            'modelo': nombre,
            'mae_entrenamiento_json': mae_guardado,
            'mae_actual': r['mae_real'],
            'diferencia_mae': r['mae_real'] - mae_guardado,
            'rmse_actual': r['rmse_real'],
            'mejora_vs_persistencia_pct': (baseline_mae - r['mae_real']) / baseline_mae * 100,
        })
    return filas


def imprimir_tabla(filas: list[dict[str, Any]], baseline_mae: float) -> None:
    """Imprime la tabla de verificación por consola."""
    print(f'\n  Persistencia (test, datos actuales): {baseline_mae:.2f} veh/h\n')
    print(f'  {"Modelo":<18} {"MAE json":>9} {"MAE actual":>11} {"Δ MAE":>8} '
          f'{"RMSE act.":>10} {"Mejora":>8}')
    print('  ' + '-' * 68)
    for f in filas:
        print(f'  {f["modelo"]:<18} {f["mae_entrenamiento_json"]:>9.2f} '
              f'{f["mae_actual"]:>11.2f} {f["diferencia_mae"]:>+8.2f} '
              f'{f["rmse_actual"]:>10.2f} {f["mejora_vs_persistencia_pct"]:>7.1f}%')


def main() -> None:
    """Evalúa los 9 modelos y guarda demo/verificacion.json."""
    device = torch.device('cpu')
    print('=' * 70)
    print('VERIFICACIÓN — modelos guardados sobre los datos regenerados (test)')
    print('=' * 70)

    with open(RESULTADOS_DIR / 'entrenamiento.json') as f:
        referencia = json.load(f)

    edge_indices = cargar_edge_indices(device)
    for k, v in edge_indices.items():
        print(f'  edge_index {k}: {tuple(v.shape)}')

    data_test = cargar_split_para_modelo('test')
    baseline_mae, _ = persistencia_mae(data_test['Y'], data_test['M'], horizon=HORIZON)
    if abs(baseline_mae - BASELINE_TFG_MAE) > 0.01:
        print(f'  ⚠️ La persistencia da {baseline_mae:.2f} y no {BASELINE_TFG_MAE:.2f}: '
              f'los datos de test no son los del TFG.')

    resultados = {}
    with torch.no_grad():
        for cfg in CONFIGURACIONES:
            print(f'\n  [{cfg["nombre"]}] grafo={cfg["grafo"]} ...', flush=True)
            resultados[cfg['nombre']] = evaluar_modelo(cfg, data_test, edge_indices, device)
            print(f'    MAE={resultados[cfg["nombre"]]["mae_real"]:.4f}  '
                  f'RMSE={resultados[cfg["nombre"]]["rmse_real"]:.4f}', flush=True)

    filas = construir_tabla(resultados, referencia, baseline_mae)
    imprimir_tabla(filas, baseline_mae)

    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    salida: Path = DEMO_DIR / 'verificacion.json'
    with open(salida, 'w', encoding='utf-8') as f:
        json.dump({'persistencia_mae_test': baseline_mae, 'modelos': filas},
                  f, indent=2, ensure_ascii=False)
    print(f'\n  Guardado: {salida}')


if __name__ == '__main__':
    main()
