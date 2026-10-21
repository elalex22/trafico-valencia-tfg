"""
escribir_metricas.py — Genera pipeline_v2_recon/demo/metricas.json, las
cifras que muestran la demo y el README.

Entradas (todas versionadas en el repositorio):
- pipeline_v2_recon/demo/verificacion.json (verificar_resultados.py): MAE,
  RMSE y mejora de los 9 modelos guardados sobre los datos regenerados, y
  el MAE de la persistencia en test.
- pipeline_v2_recon/resultados/baseline.json (07): RMSE de la persistencia.
- pipeline_v2_recon/resultados/ensembles.json (10): MAE de los ensembles.

Las mejoras se guardan con 4 decimales y las cifras a un decimal se
redondean una sola vez desde el valor sin redondear: redondear dos veces
daba un 25,1 % para GAT+Plus (25,0491 -> 25,05 -> 25,1) en lugar de 25,0 %.

Ejecutar desde cualquier directorio:
    .venv\\Scripts\\python.exe app\\escribir_metricas.py
"""

import json
from pathlib import Path
from typing import Any

RAIZ_REPO = Path(__file__).resolve().parent.parent
DEMO_DIR = RAIZ_REPO / 'pipeline_v2_recon' / 'demo'
RESULTADOS_DIR = RAIZ_REPO / 'pipeline_v2_recon' / 'resultados'
SALIDA = DEMO_DIR / 'metricas.json'

MODELO_DEMO = 'gat_plus'
MEMORIA_TFG = {
    'persistencia_mae': 37.55, 'persistencia_rmse': 128.14,
    'gat_plus_mae': 28.35, 'gat_plus_rmse': 95.39, 'mejora_gat_plus_pct': 24.5,
    'fuente': 'Memoria del TFG, Tabla 6.1 (p. 50) y Tabla 6.2 (p. 53)',
}
ENSEMBLES = {  # descripción y si elige el modelo mirando las etiquetas de test
    'promedio_3gat': ('Media de gat_plus, gat_completo y gat_enriched', False),
    'oracle_dia': ('Mejor modelo por día de la semana', True),
    'oracle_hora': ('Mejor modelo por hora', True),
    'oracle_hora_dia': ('Mejor modelo por hora × día (168 celdas)', True),
}
NOTA_DATOS = (
    'El error de 8 meses del TFG son datos corruptos (casi todo ceros) de la '
    'fuente oficial de espiras entre octubre de 2024 y mayo de 2025, que el '
    'portal reconocía como "temporalmente con errores de procesamiento"; '
    'esos meses quedan excluidos de los tres splits (memoria, p. 31 y 37). '
    'Además, del 1 al 12 de junio de 2025 la fuente estuvo congelada: 11,3 días '
    'con los valores repetidos, alrededor del 10 % del test. Las métricas lo '
    'incluyen. Durante la primera semana de test (1–7 de junio de 2025), '
    'la variable lag7d procede del 25–31 de mayo de 2025, que cae en el '
    'régimen corrupto.')
NOTA_ENSEMBLES = (
    'Los ensembles proceden de resultados/ensembles.json y no se han '
    'recalculado con el grafo regenerado. Los "oracle" eligen el mejor '
    'modelo de cada grupo mirando las etiquetas de test, así que son una '
    'cota optimista, no un modelo desplegable.')


def coma(x: float, decimales: int) -> str:
    """Número con coma decimal y `decimales` cifras: 28.7098 -> '28,71'."""
    return f'{x:.{decimales}f}'.replace('.', ',')


def leer_json(ruta: Path) -> Any:
    """Lee un JSON en UTF-8."""
    with open(ruta, encoding='utf-8') as f:
        return json.load(f)


def construir_metricas(verificacion: dict[str, Any], ensembles: dict[str, float],
                       baseline: dict[str, Any]) -> dict[str, Any]:
    """Arma el contenido de metricas.json a partir de las tres entradas."""
    base = verificacion['persistencia_mae_test']

    def mejora(mae: float) -> float:
        return round((base - mae) / base * 100, 4)

    modelos = {f['modelo']: {'mae': round(f['mae_actual'], 2), 'rmse': round(f['rmse_actual'], 2),
                             'mejora_vs_persistencia_pct': round(f['mejora_vs_persistencia_pct'], 4)}
               for f in verificacion['modelos']}
    demo = next(f for f in verificacion['modelos'] if f['modelo'] == MODELO_DEMO)

    return {
        'unidades': 'veh/h (MAE y RMSE en test, 2025-06-01 a 2025-09-23)',
        'baseline_persistencia': {'mae': round(base, 2),
                                  'rmse': round(baseline['test']['rmse_real'], 2)},
        'modelo_demo': MODELO_DEMO,
        'modelos': modelos,
        'ensembles': {k: {'mae': round(ensembles[k], 2), 'mejora_vs_persistencia_pct': mejora(ensembles[k]),
                          'descripcion': d, 'eleccion_con_etiquetas_de_test': o}
                      for k, (d, o) in ENSEMBLES.items()},
        'memoria_tfg': MEMORIA_TFG,
        'nota': ('pipeline_v2_recon es una reconstrucción del pipeline del TFG: los modelos '
                 'guardados (sin reentrenar) evaluados con los datos regenerados por los '
                 f'scripts 01–07. Su baseline de persistencia ({coma(base, 2)} veh/h) coincide con el '
                 'baseline.json del repositorio, no con la memoria, que reporta '
                 f'{coma(MEMORIA_TFG["persistencia_mae"], 2)} veh/h (Tabla 6.1, p. 50). En la memoria, '
                 f'GAT+Plus da {coma(MEMORIA_TFG["gat_plus_mae"], 2)} veh/h y mejora un '
                 f'{coma(MEMORIA_TFG["mejora_gat_plus_pct"], 1)} % a la persistencia (Tabla 6.2, p. 53); '
                 f'en la reconstrucción, {coma(demo["mae_actual"], 2)} veh/h y un '
                 f'{coma(demo["mejora_vs_persistencia_pct"], 1)} %. '
                 'Las coberturas de los splits también difieren de la Tabla 4.2 de la memoria. '
                 'El grafo de OpenStreetMap se descargó de nuevo en octubre de 2026 y tiene 2.855 '
                 'aristas en lugar de las 2.863 del TFG; los 1.035 nodos coinciden.'),
        'nota_datos': NOTA_DATOS,
        'nota_ensembles': NOTA_ENSEMBLES,
    }


def main() -> None:
    """Lee las entradas y escribe metricas.json."""
    metricas = construir_metricas(leer_json(DEMO_DIR / 'verificacion.json'),
                                  leer_json(RESULTADOS_DIR / 'ensembles.json'),
                                  leer_json(RESULTADOS_DIR / 'baseline.json'))
    with open(SALIDA, 'w', encoding='utf-8') as f:
        json.dump(metricas, f, indent=2, ensure_ascii=False)
    print(f'  {SALIDA.relative_to(RAIZ_REPO)}: GAT+Plus {coma(metricas["modelos"][MODELO_DEMO]["mae"], 2)} '
          f'veh/h, mejora {coma(metricas["modelos"][MODELO_DEMO]["mejora_vs_persistencia_pct"], 1)} %')


if __name__ == '__main__':
    main()
