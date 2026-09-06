"""Inferencia de un modelo ya entrenado sobre un split.

Usa `import config` + `config.ATRIBUTO` (no `from config import ...`) a
propósito: así, si otro script hace `config.MODELS_DIR = ...` ANTES de
llamar a predecir_modelo(), el cambio se respeta.
"""
import numpy as np
import torch
from torch.utils.data import DataLoader

import config
from modelos.registro import MODEL_REGISTRY
from datos.io_tensores import cargar_split_para_modelo


def _edge_index(grafo, device):
    fname = 'edge_index_final.npy' if grafo == 'fisico' else 'edge_index_enriched.npy'
    ei = np.load(config.GRAPH_DIR / fname)
    return torch.from_numpy(ei).long().to(device)

def predecir_modelo(nombre, split, device, batch_size=32, registry=MODEL_REGISTRY):
    info = registry[nombre]
    data_split = cargar_split_para_modelo(split)

    ds = info['dataset'](data_split, window=config.WINDOW, horizon=config.HORIZON)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    is_gcn = info.get('is_gcn', True)
    edge_index = _edge_index(info['grafo'], device) if is_gcn else None

    if info.get('con_ventana', True):
        model = info['clase'](window=config.WINDOW, n_features=info['n_features']).to(device)
    else:
        model = info['clase'](n_features=info['n_features']).to(device)

    ckpt = config.MODELS_DIR / f'{nombre}.pt'
    print(f'    (cargando {ckpt})')
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()

    preds, ys, ms = [], [], []
    with torch.no_grad():
        for x, y, m in loader:
            x, y, m = x.to(device), y.to(device), m.to(device)
            pred = model(x, edge_index) if is_gcn else model(x)
            preds.append(torch.expm1(pred.clamp(min=0)).cpu().numpy())
            ys.append(torch.expm1(y).cpu().numpy())
            ms.append(m.cpu().numpy())

    pred_real = np.concatenate(preds, axis=0)
    y_real = np.concatenate(ys, axis=0)
    mask = np.concatenate(ms, axis=0)

    start_offset = getattr(ds, 'start_offset', 0)
    target_offset = config.WINDOW + config.HORIZON - 1 + start_offset
    return pred_real, y_real, mask, target_offset


def alinear(resultados):
    offset_comun = max(r[3] for r in resultados.values())
    alineados = {}
    for nombre, (pred, y, m, off) in resultados.items():
        recorte = offset_comun - off
        alineados[nombre] = (pred[recorte:], y[recorte:], m[recorte:])
    largos = {n: p.shape[0] for n, (p, _, _) in alineados.items()}
    assert len(set(largos.values())) == 1, f'Longitudes no alineadas: {largos}'
    return alineados, offset_comun