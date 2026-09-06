"""
08_entrenar_modelos.py — Entrena las 9 arquitecturas evaluadas en el
proyecto (Persistencia no necesita entrenamiento: tiene 0 parámetros y
ya se calcula en 07_baseline_persistencia.py).

Los hiperparámetros de entrenamiento (optimizador, scheduler, gradient
clipping) son comunes a todos los modelos y viven en
modelos/entrenamiento.py. batch_size y nº de epochs sí varían por
arquitectura:

    modelo             batch_size  epochs
    mlp_simple                 32      10
    mlp_completo                32      10
    gcn_simple_v2               32       8
    gcn_completo_v2             16       8
    gcn_completo_big            16      10
    gat_completo                16      10
    gat_enriched                16      12
    gat_plus                    16      12
    gat_ultra                   16      12

Nota sobre reproducibilidad: no se fija ninguna semilla aleatoria
(`torch.manual_seed`, etc.), así que dos ejecuciones no dan resultados
bit-idénticos — es esperable una variación de hasta un par de puntos
porcentuales entre runs. Una desviación mucho mayor (>5-10%) sí sería
indicio de un problema en el pipeline de datos, no de aleatoriedad del
entrenamiento.

Requiere: 05_pipeline_tensores.py y 06_grafo_enriquecido.py ya ejecutados.
Genera: models/*.pt (9 checkpoints) + resultados/entrenamiento.json
"""

import json
import numpy as np
import torch

from config import GRAPH_DIR, MODELS_DIR, RESULTADOS_DIR, WINDOW, HORIZON
from datos.io_tensores import cargar_split_para_modelo
from modelos.arquitecturas import (
    MLPSimple, MLPCompleto, GCNSimpleV2, GCNCompletoV2, GCNCompletoBig,
    GATCompleto, GATCompletoPlus, GATCompletoUltra,
)
from modelos.datasets import DatasetSimple, DatasetCompleto, DatasetCompletoPlus
from modelos.entrenamiento import entrenar, persistencia_mae


# ── Configuración por modelo — extraída literalmente de 07_modelos_final.ipynb ──
CONFIGURACIONES = [
    dict(nombre='mlp_simple', clase=MLPSimple, dataset_cls=DatasetSimple,
         con_ventana=False, grafo=None, batch_size=32, epochs=10),
    dict(nombre='mlp_completo', clase=MLPCompleto, dataset_cls=DatasetCompleto,
         con_ventana=True, grafo=None, batch_size=32, epochs=10),
    dict(nombre='gcn_simple_v2', clase=GCNSimpleV2, dataset_cls=DatasetSimple,
         con_ventana=False, grafo='fisico', batch_size=32, epochs=8),
    dict(nombre='gcn_completo_v2', clase=GCNCompletoV2, dataset_cls=DatasetCompleto,
         con_ventana=True, grafo='fisico', batch_size=16, epochs=8),
    dict(nombre='gcn_completo_big', clase=GCNCompletoBig, dataset_cls=DatasetCompleto,
         con_ventana=True, grafo='fisico', batch_size=16, epochs=10),
    dict(nombre='gat_completo', clase=GATCompleto, dataset_cls=DatasetCompleto,
         con_ventana=True, grafo='fisico', batch_size=16, epochs=10),
    dict(nombre='gat_enriched', clase=GATCompleto, dataset_cls=DatasetCompleto,
         con_ventana=True, grafo='enriquecido', batch_size=16, epochs=12),
    dict(nombre='gat_plus', clase=GATCompletoPlus, dataset_cls=DatasetCompletoPlus,
         con_ventana=True, grafo='fisico', batch_size=16, epochs=12),
    dict(nombre='gat_ultra', clase=GATCompletoUltra, dataset_cls=DatasetCompletoPlus,
         con_ventana=True, grafo='fisico', batch_size=16, epochs=12),
]

# Valores de referencia obtenidos previamente para cada arquitectura,
# usados solo para comparar al final (no intervienen en el entrenamiento).
# gat_enriched no tiene entrada propia porque solo se usa en el ensemble
# de 10_ensembles.py, no se reporta como modelo individual.
RESULTADOS_REFERENCIA = {
    'mlp_simple': 35.55, 'mlp_completo': 34.20, 'gcn_simple_v2': 33.12,
    'gcn_completo_v2': 30.68, 'gcn_completo_big': 30.73,
    'gat_completo': 28.50, 'gat_plus': 28.35, 'gat_ultra': 29.60,
}


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('=' * 70)
    print(f'08 — ENTRENAMIENTO DE MODELOS  |  device={device}')
    print(f'  Checkpoints -> {MODELS_DIR}')
    print('=' * 70)

    edge_index_fisico = torch.from_numpy(
        np.load(GRAPH_DIR / 'edge_index_final.npy')).long().to(device)
    edge_index_enriquecido = torch.from_numpy(
        np.load(GRAPH_DIR / 'edge_index_enriched.npy')).long().to(device)
    print(f'  edge_index físico:     {tuple(edge_index_fisico.shape)}')
    print(f'  edge_index enriquecido: {tuple(edge_index_enriquecido.shape)}')

    print('\nCargando splits...')
    data = {s: cargar_split_para_modelo(s) for s in ['train', 'val', 'test']}
    for s, d in data.items():
        print(f'  {s}: X={d["X"].shape}')

    persistencia_test_mae, _ = persistencia_mae(data['test']['Y'], data['test']['M'], horizon=HORIZON)
    print(f'\nPersistencia (test): {persistencia_test_mae:.2f} veh/h')

    resultados = {}
    for cfg in CONFIGURACIONES:
        nombre = cfg['nombre']
        DatasetCls = cfg['dataset_cls']

        ds_train = DatasetCls(data['train'], window=WINDOW, horizon=HORIZON)
        ds_val = DatasetCls(data['val'], window=WINDOW, horizon=HORIZON)
        ds_test = DatasetCls(data['test'], window=WINDOW, horizon=HORIZON)

        x_sample, _, _ = ds_train[0]
        n_feats = x_sample.shape[-1]

        if cfg['con_ventana']:
            model = cfg['clase'](window=WINDOW, n_features=n_feats).to(device)
        else:
            model = cfg['clase'](n_features=n_feats).to(device)

        edge_index = None
        if cfg['grafo'] == 'fisico':
            edge_index = edge_index_fisico
        elif cfg['grafo'] == 'enriquecido':
            edge_index = edge_index_enriquecido
        is_gcn = cfg['grafo'] is not None

        print(f'\n  [{nombre}] features={n_feats}  grafo={cfg["grafo"]}  '
              f'batch={cfg["batch_size"]}  epochs={cfg["epochs"]}')

        resultados[nombre] = entrenar(
            nombre, model, ds_train, ds_val, ds_test, device,
            persistencia_test_mae, edge_index=edge_index, is_gcn=is_gcn,
            batch_size=cfg['batch_size'], epochs=cfg['epochs'], lr=1e-3, patience=3,
        )

    print('\n\n' + '=' * 70)
    print('RESUMEN FINAL')
    print('=' * 70)
    print(f'\n  {"Modelo":<20} {"MAE":>8} {"RMSE":>8} {"Referencia":>10} {"Δ":>8}')
    print('  ' + '-' * 58)
    for nombre, r in resultados.items():
        ref = RESULTADOS_REFERENCIA.get(nombre)
        if ref is not None:
            diff = r['mae_real'] - ref
            marca = f'{ref:>10.2f} {diff:>+7.2f}'
            if abs(diff) / ref > 0.10:
                marca += '  ⚠️ desviación >10%'
        else:
            marca = f'{"—":>10} {"—":>8}'
        print(f'  {nombre:<20} {r["mae_real"]:>8.2f} {r["rmse_real"]:>8.2f}  {marca}')

    RESULTADOS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTADOS_DIR / 'entrenamiento.json', 'w') as f:
        json.dump(resultados, f, indent=2)
    print(f'\n  Guardado: resultados/entrenamiento.json')
    print(f'  Checkpoints en: {MODELS_DIR}')


if __name__ == '__main__':
    main()
