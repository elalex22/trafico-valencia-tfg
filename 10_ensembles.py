"""
10_ensembles.py — 4 estrategias de ensemble sobre test:
  oracle por día · promedio 3 GAT · oracle por hora · oracle hora×día.

── CAMBIO respecto a la versión anterior, tras revisión a fondo ──
Se QUITA el override `config.MODELS_DIR = ... pipeline_v2 / 'models'`.
Ya no hace falta: 08_entrenar_modelos.py deja los checkpoints nuevos en
pipeline_v2_recon/models (config.MODELS_DIR por defecto) — mezclar
checkpoints viejos con tensores nuevos fue el problema de fondo
detectado en la revisión (ver Y_lag24.npy vs Y_lag24_log.npy).

Requiere: 08_entrenar_modelos.py ya ejecutado (necesita gat_plus.pt,
gat_completo.pt, gat_enriched.pt en pipeline_v2_recon/models).
Genera: resultados/ensembles.json
"""
import json
import numpy as np
import pandas as pd
import torch

from config import TENSOR_DIR, RESULTADOS_DIR, MODELS_DIR
from modelos.registro import MODEL_REGISTRY, GATS_PARA_PROMEDIO
from modelos.inferencia import predecir_modelo, alinear


def mae_enmascarado(pred, y, m):
    m = m.astype(bool)
    return float(np.abs(pred[m] - y[m]).mean()) if m.sum() else float('nan')


def oracle_por_grupo(alineados, y_ref, m_ref, grupo_mask_dict):
    pred_oracle = np.zeros_like(y_ref)
    mejor_por_grupo = {}
    for clave, mask in grupo_mask_dict.items():
        if mask.sum() == 0:
            continue
        maes = {n: mae_enmascarado(p[mask], y_ref[mask], m_ref[mask]) for n, (p, _, _) in alineados.items()}
        mejor = min(maes, key=maes.get)
        mejor_por_grupo[clave] = mejor
        pred_oracle[mask] = alineados[mejor][0][mask]
    return pred_oracle, mejor_por_grupo


def main():
    print('=' * 70)
    print('10 — ESTUDIO DE ENSEMBLES — split test')
    print('=' * 70)
    print(f'  Checkpoints desde: {MODELS_DIR}')
    print(f'  Tensores desde:    {TENSOR_DIR}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('\n[1] Inferencia de los modelos...')
    crudos = {n: predecir_modelo(n, 'test', device) for n in MODEL_REGISTRY}
    alineados, offset_comun = alinear(crudos)
    print(f'  Alineado en target_offset={offset_comun}')

    n = next(iter(alineados.values()))[0].shape[0]
    ts_test = np.load(TENSOR_DIR / 'timestamps_test.npy', allow_pickle=True)
    ts_target = pd.to_datetime(ts_test[offset_comun: offset_comun + n])
    horas, dias = ts_target.hour.values, ts_target.dayofweek.values

    y_ref, m_ref = alineados['gat_plus'][1], alineados['gat_plus'][2]
    resultados = {}

    mae_plus = mae_enmascarado(alineados['gat_plus'][0], y_ref, m_ref)
    resultados['gat_plus_individual'] = mae_plus
    print(f'\n[2] GAT+Plus individual: {mae_plus:.2f} veh/h (memoria: 28.35)')

    print('\n[3] Oracle por día...')
    grupo_dia = {d: (dias == d) for d in range(7)}
    pred_od, mejor_dia = oracle_por_grupo(alineados, y_ref, m_ref, grupo_dia)
    resultados['oracle_dia'] = mae_enmascarado(pred_od, y_ref, m_ref)
    print(f'  {mejor_dia}')
    print(f'  MAE: {resultados["oracle_dia"]:.2f} veh/h (memoria: 28.27)')

    print('\n[4] Promedio de 3 GAT...')
    pred_prom = np.mean([alineados[n][0] for n in GATS_PARA_PROMEDIO], axis=0)
    resultados['promedio_3gat'] = mae_enmascarado(pred_prom, y_ref, m_ref)
    print(f'  MAE: {resultados["promedio_3gat"]:.2f} veh/h (memoria: 27.94)')

    print('\n[5] Oracle por hora...')
    grupo_hora = {h: (horas == h) for h in range(24)}
    pred_oh, mejor_hora = oracle_por_grupo(alineados, y_ref, m_ref, grupo_hora)
    resultados['oracle_hora'] = mae_enmascarado(pred_oh, y_ref, m_ref)
    print(f'  MAE: {resultados["oracle_hora"]:.2f} veh/h (memoria: 27.85)')

    print('\n[6] Oracle por hora × día (168 celdas)...')
    grupo_hd = {(h, d): (horas == h) & (dias == d) for h in range(24) for d in range(7)}
    pred_ohd, mejor_hd = oracle_por_grupo(alineados, y_ref, m_ref, grupo_hd)
    resultados['oracle_hora_dia'] = mae_enmascarado(pred_ohd, y_ref, m_ref)
    print(f'  MAE: {resultados["oracle_hora_dia"]:.2f} veh/h (memoria: 27.63)')

    conteo = {}
    for modelo in mejor_hd.values():
        conteo[modelo] = conteo.get(modelo, 0) + 1
    print(f'\n  Reparto de victorias (168 celdas): {conteo}')
    print('  ⚠️ Esto NO es directamente comparable al reparto "por hora agregada"')
    print('     de la memoria — son dos análisis de granularidad distinta.')

    with open(RESULTADOS_DIR / 'ensembles.json', 'w') as f:
        json.dump({**resultados, 'reparto_victorias_hd': conteo}, f, indent=2, default=str)

    print('\n' + '=' * 70)
    for k, v in resultados.items():
        print(f'  {k:20s} {v:.2f} veh/h')


if __name__ == '__main__':
    main()
