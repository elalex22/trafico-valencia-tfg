"""
06_grafo_enriquecido.py — Adyacencia enriquecida con correlación histórica.

Añade aristas entre nodos con alta correlación de tráfico histórico (no
necesariamente conectados físicamente), complementando el grafo físico.

Requiere: 02_construccion_grafo.py (edge_index_final.npy) y
          05_pipeline_tensores.py (Y_train_log.npy, M_train.npy) ya ejecutados.

Genera:
  - grafo/edge_index_enriched.npy    (físico ∪ correlación)
  - grafo/edge_index_correlacion.npy (solo aristas nuevas por correlación)
  - grafo/corr_matrix.npy            (matriz de correlación completa N×N)
"""

import gc
import numpy as np

from config import (TENSOR_DIR, GRAPH_DIR, UMBRAL_CORR, K_MAX_CORR,
                     MIN_ACTIVIDAD_NODO, TIMESTEP_COBERTURA_MIN)


def calcular_actividad_nodos(M_train):
    """% de timesteps con dato válido, por nodo. Filtra nodos poco activos
    para que no generen correlaciones espurias."""
    activ = M_train.mean(axis=0)
    print(f'  Actividad media por nodo: {activ.mean()*100:.1f}%')
    print(f'    P10={np.percentile(activ,10):.2f}  P50={np.percentile(activ,50):.2f}  '
          f'P90={np.percentile(activ,90):.2f}')

    nodos_utilizables = np.where(activ >= MIN_ACTIVIDAD_NODO)[0]
    print(f'  Nodos con actividad ≥ {MIN_ACTIVIDAD_NODO*100:.0f}%: '
          f'{len(nodos_utilizables)}/{len(activ)}')
    return nodos_utilizables


def calcular_correlacion(Y_train, M_train, nodos_utilizables):
    """Pearson pareado sobre timesteps con cobertura suficiente, imputando
    huecos con la media del nodo (evita que los ceros distorsionen la corr)."""
    Y_util = Y_train[:, nodos_utilizables]
    M_util = M_train[:, nodos_utilizables]

    timesteps_buenos = M_util.mean(axis=1) > TIMESTEP_COBERTURA_MIN
    print(f'  Timesteps con >{TIMESTEP_COBERTURA_MIN*100:.0f}% de nodos válidos: '
          f'{timesteps_buenos.sum():,}/{len(M_util):,}')

    Y_sub, M_sub = Y_util[timesteps_buenos], M_util[timesteps_buenos]
    means = (Y_sub * M_sub).sum(axis=0) / np.maximum(M_sub.sum(axis=0), 1)
    Y_filled = np.where(M_sub == 1, Y_sub, means[None, :])

    print('  Calculando np.corrcoef...')
    corr_util = np.corrcoef(Y_filled, rowvar=False).astype(np.float32)
    np.fill_diagonal(corr_util, 0)

    print(f'  corr_util: min={corr_util.min():.3f}  max={corr_util.max():.3f}  '
          f'|mean|={np.abs(corr_util).mean():.3f}')
    if corr_util.min() < -1.01 or corr_util.max() > 1.01:
        print('  ⚠ ERROR: correlación fuera de rango [-1, 1]')
    else:
        print('  ✓ Correlación en rango válido')

    N = Y_train.shape[1]
    corr_full = np.zeros((N, N), dtype=np.float32)
    corr_full[np.ix_(nodos_utilizables, nodos_utilizables)] = corr_util
    np.fill_diagonal(corr_full, 0)

    del Y_util, M_util, Y_sub, M_sub, Y_filled, corr_util
    gc.collect()
    return corr_full


def seleccionar_aristas_por_correlacion(corr_full):
    """Top-K vecinos por nodo entre los que superan el umbral de correlación.
    Se añaden ambas direcciones explícitamente porque el top-K no es
    simétrico (i puede tener a j en su top-K sin que ocurra lo inverso)."""
    N = corr_full.shape[0]
    abs_corr = np.abs(corr_full)

    mask_upper = np.triu(np.ones_like(abs_corr, dtype=bool), k=1)
    valores = abs_corr[mask_upper]
    print(f'  Distribución |corr| (triángulo sup.): P50={np.percentile(valores,50):.3f}  '
          f'P75={np.percentile(valores,75):.3f}  P90={np.percentile(valores,90):.3f}  '
          f'P95={np.percentile(valores,95):.3f}  P99={np.percentile(valores,99):.3f}')

    edges_correlacion = set()
    for i in range(N):
        scores = abs_corr[i].copy()
        scores[i] = 0
        candidatos = np.where(scores >= UMBRAL_CORR)[0]
        if len(candidatos) == 0:
            continue
        top_scores = scores[candidatos]
        k_efectivo = min(K_MAX_CORR, len(candidatos))
        top_idx = candidatos[np.argsort(top_scores)[-k_efectivo:]]
        for j in top_idx:
            edges_correlacion.add((int(i), int(j)))
            edges_correlacion.add((int(j), int(i)))

    grados = np.zeros(N, dtype=np.int32)
    for i, _ in edges_correlacion:
        grados[i] += 1
    print(f'  Umbral={UMBRAL_CORR}  K_max={K_MAX_CORR}  '
          f'Aristas por correlación: {len(edges_correlacion):,}')
    print(f'  Grado min/mediana/max: {grados.min()}/{int(np.median(grados))}/{grados.max()}')
    return edges_correlacion


def combinar_con_fisica(edges_correlacion):
    edge_index_fisica = np.load(GRAPH_DIR / 'edge_index_final.npy')
    edges_fisicas = set(map(tuple, edge_index_fisica.T.tolist()))

    edges_union = edges_fisicas | edges_correlacion
    edges_solo_corr = edges_correlacion - edges_fisicas

    print(f'  Físicas: {len(edges_fisicas):,}  Correlación: {len(edges_correlacion):,}  '
          f'Nuevas (solo corr): {len(edges_solo_corr):,}  Unión total: {len(edges_union):,}')

    edge_index_enriched = np.array(list(edges_union)).T.astype(np.int64)
    edge_index_correlacion = (np.array(list(edges_solo_corr)).T.astype(np.int64)
                               if edges_solo_corr else np.zeros((2, 0), dtype=np.int64))
    return edge_index_enriched, edge_index_correlacion


def main():
    print('=' * 70)
    print('ADYACENCIA ENRIQUECIDA — Correlación histórica')
    print('=' * 70)

    Y_train = np.load(TENSOR_DIR / 'Y_train_log.npy').astype(np.float32)
    M_train = np.load(TENSOR_DIR / 'M_train.npy').astype(np.float32)
    print(f'\n[1] Shape: T={Y_train.shape[0]:,}, N={Y_train.shape[1]}')

    print('\n[2] Actividad por nodo...')
    nodos_utilizables = calcular_actividad_nodos(M_train)

    print('\n[3] Correlación pareada...')
    corr_full = calcular_correlacion(Y_train, M_train, nodos_utilizables)

    print('\n[4-5] Selección de aristas...')
    edges_correlacion = seleccionar_aristas_por_correlacion(corr_full)

    print('\n[6] Combinando con adyacencia física...')
    edge_index_enriched, edge_index_correlacion = combinar_con_fisica(edges_correlacion)

    grados_final = np.zeros(Y_train.shape[1], dtype=np.int32)
    for i in edge_index_enriched[0]:
        grados_final[i] += 1
    print(f'\n[7] Grado final: min={grados_final.min()} mediana={int(np.median(grados_final))} '
          f'media={grados_final.mean():.1f} max={grados_final.max()} '
          f'P95={int(np.percentile(grados_final, 95))}')

    print('\n[8] Guardando...')
    np.save(GRAPH_DIR / 'edge_index_enriched.npy', edge_index_enriched)
    np.save(GRAPH_DIR / 'edge_index_correlacion.npy', edge_index_correlacion)
    np.save(GRAPH_DIR / 'corr_matrix.npy', corr_full)
    print(f'  edge_index_enriched.npy     {edge_index_enriched.shape}')
    print(f'  edge_index_correlacion.npy  {edge_index_correlacion.shape}')
    print(f'  corr_matrix.npy             {corr_full.shape}')


if __name__ == '__main__':
    main()
