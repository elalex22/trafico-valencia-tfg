"""
09_analisis_temporal.py — GAT+Plus vs persistencia: por franja horaria,
por hora y por día de la semana. Split: test.

Requiere: 08_entrenar_modelos.py ya ejecutado (necesita gat_plus.pt en
models/).
Genera: resultados/analisis_temporal.json
"""
import json
import numpy as np
import pandas as pd
import torch

from config import TENSOR_DIR, RESULTADOS_DIR
from modelos.inferencia import predecir_modelo

# ── Franjas horarias del día ──
FRANJAS = {
    'madrugada': range(0, 6),
    'punta_manana': range(6, 10),
    'valle_mediodia': range(10, 14),
    'tarde': range(14, 17),
    'punta_tarde': range(17, 20),
    'noche': range(20, 24),
}
NOMBRE_DIAS = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']


def mae_enmascarado(pred, y, m):
    m = m.astype(bool)
    return float(np.abs(pred[m] - y[m]).mean()) if m.sum() else float('nan')


def desglosar(pred, y, m, y_now, grupo):
    out = {}
    for etiqueta, mask in grupo.items():
        pred_g, y_g, m_g, ynow_g = pred[mask], y[mask], m[mask], y_now[mask]
        mae_mod = mae_enmascarado(pred_g, y_g, m_g)
        mae_per = mae_enmascarado(ynow_g, y_g, m_g)
        mejora = (mae_per - mae_mod) / mae_per * 100 if mae_per else float('nan')
        out[etiqueta] = {'mae_modelo': mae_mod, 'mae_persist': mae_per, 'mejora_pct': mejora}
        print(f'  {str(etiqueta):16s} modelo={mae_mod:.2f}  persist={mae_per:.2f}  mejora={mejora:+.2f}%')
    return out


def main():
    print('=' * 70)
    print('09 — ANÁLISIS TEMPORAL (GAT+Plus vs persistencia) — split test')
    print('=' * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('\n[1] Inferencia GAT+Plus sobre test...')
    pred, y, m, target_offset = predecir_modelo('gat_plus', 'test', device)
    print(f'  Predicciones: {pred.shape} | target_offset={target_offset}')

    Y_real_test = np.expm1(np.clip(np.load(TENSOR_DIR / 'Y_test_log.npy'), 0, None))
    y_now = Y_real_test[target_offset - 1: target_offset - 1 + pred.shape[0]]

    ts_test = np.load(TENSOR_DIR / 'timestamps_test.npy', allow_pickle=True)
    ts_target = pd.to_datetime(ts_test[target_offset: target_offset + pred.shape[0]])
    horas, dias = ts_target.hour.values, ts_target.dayofweek.values

    resultados = {}

    print('\n[2] Por franja horaria...')
    grupo_franja = {n: np.isin(horas, list(r)) for n, r in FRANJAS.items()}
    resultados['franja'] = desglosar(pred, y, m, y_now, grupo_franja)

    print('\n[3] Por hora del día...')
    grupo_hora = {h: (horas == h) for h in range(24)}
    resultados['hora'] = desglosar(pred, y, m, y_now, grupo_hora)

    print('\n[4] Por día de la semana...')
    grupo_dia = {NOMBRE_DIAS[d]: (dias == d) for d in range(7)}
    resultados['dia_semana'] = desglosar(pred, y, m, y_now, grupo_dia)

    with open(RESULTADOS_DIR / 'analisis_temporal.json', 'w') as f:
        json.dump(resultados, f, indent=2, default=str)

    print('\n  Guardado: resultados/analisis_temporal.json')
    print('  ⚠️ El MAE medio de "dia_semana" debería salir ≈ 28.35 veh/h (global GAT+Plus).')


if __name__ == '__main__':
    main()
