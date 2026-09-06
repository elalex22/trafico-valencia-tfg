"""
Configuración central del pipeline reconstruido.

⚠️ SEGURIDAD: DATA apunta a 'pipeline_v2_recon', NO a 'pipeline_v2' (tu
pipeline real). Nada de lo que escriban estos scripts puede tocar los datos
que ya respaldan las cifras de tu memoria. Las carpetas de fuentes brutas y
pipeline_v2/ se leen, nunca se escriben.
"""

from pathlib import Path

# Ejecuta los scripts desde la carpeta que contiene tus fuentes brutas y pipeline_v2/
RAIZ = Path('.')

# ── SANDBOX: todo lo que se ESCRIBE va aquí ──
DATA = RAIZ / 'pipeline_v2_recon' / 'data'
PROC = DATA / 'processed'
GRAPH_DIR = DATA / 'grafo'
EDA = DATA / 'eda_grafo'
MASTER_DIR = DATA / 'master_final'
TENSOR_DIR = DATA / 'tensors_final'
MODELS_DIR = RAIZ / 'pipeline_v2_recon' / 'models'
RESULTADOS_DIR = RAIZ / 'pipeline_v2_recon' / 'resultados'

for d in (PROC, GRAPH_DIR, EDA, MASTER_DIR, TENSOR_DIR, MODELS_DIR, RESULTADOS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ── Constantes del pipeline de datos ──
LAG_24H = 96
LAG_7D = 672
UMBRAL_MAX_Y = 5000
MAX_GAP_TIMESTEPS = 8
RACHA_CEROS = 16

# ── Constantes de modelado ──
WINDOW = 12
HORIZON = 1

# ── Splits definitivos (memoria, tabla 4.4.3) ──
TROZOS_POR_SPLIT = {
    'train': [
        {'start': '2023-03-16 17:00:00', 'end': '2023-10-31 23:45:00',
         'buf_start': '2023-03-09 17:00:00'},
        {'start': '2023-11-01 00:00:00', 'end': '2024-05-31 23:45:00',
         'buf_start': '2023-10-25 00:00:00'},
    ],
    'val': [
        {'start': '2024-06-01 00:00:00', 'end': '2024-07-22 23:45:00',
         'buf_start': '2024-05-25 00:00:00'},
        {'start': '2024-08-17 00:00:00', 'end': '2024-09-30 23:45:00',
         'buf_start': '2024-08-10 00:00:00'},
    ],
    'test': [
        {'start': '2025-06-01 00:00:00', 'end': '2025-09-23 23:45:00',
         'buf_start': '2025-05-25 00:00:00'},
    ],
}

# ── Features dinámicas (22) ──
FEATURES_DIN = [
    'estado', 'mask_estado', 'tmed', 'prec', 'mask_meteo',
    'hora_sin', 'hora_cos', 'minuto_sin', 'minuto_cos',
    'dia_semana_sin', 'dia_semana_cos',
    'es_finde', 'es_festivo', 'es_fallas',
    'ih_bici', 'mask_ih_bici', 'available', 'free', 'mask_valenbici',
    'laeq', 'laeq_n', 'mask_ruido',
]
FEATURES_STATIC = ['lat', 'lon']

FEATURES_A_ESTANDARIZAR = ['estado', 'tmed', 'prec', 'ih_bici',
                            'available', 'free', 'laeq', 'laeq_n']
MASK_POR_FEATURE = {
    'estado': 'mask_estado', 'tmed': 'mask_meteo', 'prec': 'mask_meteo',
    'ih_bici': 'mask_ih_bici', 'available': 'mask_valenbici',
    'free': 'mask_valenbici', 'laeq': 'mask_ruido', 'laeq_n': 'mask_ruido',
}

# ── Adyacencia enriquecida ──
UMBRAL_CORR = 0.7
K_MAX_CORR = 8
MIN_ACTIVIDAD_NODO = 0.5
TIMESTEP_COBERTURA_MIN = 0.9
FIGURAS_DIR = RAIZ / 'figuras_tfg'
FIGURAS_DIR.mkdir(parents=True, exist_ok=True)