"""Registro de modelos, para 09, 10 y la figura de ablation study (6.2)."""
from modelos.arquitecturas import (GATCompletoPlus, GATCompleto, GATCompletoUltra,
                                     GCNCompletoV2, GCNSimpleV2, MLPSimple, MLPCompleto)
from modelos.datasets import DatasetSimple, DatasetCompleto, DatasetCompletoPlus

MODEL_REGISTRY = {
    'gat_plus':        {'clase': GATCompletoPlus,  'dataset': DatasetCompletoPlus, 'grafo': 'fisico',     'n_features': 33, 'is_gcn': True, 'con_ventana': True},
    'gat_completo':    {'clase': GATCompleto,      'dataset': DatasetCompleto,     'grafo': 'fisico',     'n_features': 25, 'is_gcn': True, 'con_ventana': True},
    'gat_enriched':    {'clase': GATCompleto,      'dataset': DatasetCompleto,     'grafo': 'enriquecido','n_features': 25, 'is_gcn': True, 'con_ventana': True},
    'gat_ultra':       {'clase': GATCompletoUltra, 'dataset': DatasetCompletoPlus, 'grafo': 'fisico',     'n_features': 33, 'is_gcn': True, 'con_ventana': True},
    'gcn_completo_v2': {'clase': GCNCompletoV2,    'dataset': DatasetCompleto,     'grafo': 'fisico',     'n_features': 25, 'is_gcn': True, 'con_ventana': True},
}

GATS_PARA_PROMEDIO = ['gat_plus', 'gat_completo', 'gat_enriched']

ABLATION_REGISTRY = {
    **MODEL_REGISTRY,
    'mlp_simple':    {'clase': MLPSimple,    'dataset': DatasetSimple,   'grafo': None,     'n_features': 25, 'is_gcn': False, 'con_ventana': False},
    'mlp_completo':  {'clase': MLPCompleto,  'dataset': DatasetCompleto, 'grafo': None,     'n_features': 25, 'is_gcn': False, 'con_ventana': True},
    'gcn_simple_v2': {'clase': GCNSimpleV2,  'dataset': DatasetSimple,   'grafo': 'fisico', 'n_features': 25, 'is_gcn': True,  'con_ventana': False},
}

ABLATION_ORDEN = ['mlp_simple', 'mlp_completo', 'gcn_simple_v2', 'gcn_completo_v2', 'gat_completo', 'gat_plus']