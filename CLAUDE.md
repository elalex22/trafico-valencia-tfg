# Proyecto: predicción del tráfico urbano de Valencia con GNN

## Contexto
- TFG del Grado en Ciencia de Datos (UPV), calificación 10.
- Pipeline en scripts numerados que se ejecutan en orden:
  01_extraccion → 02_construccion_grafo → 03_asignaciones_espaciales →
  04_fusion_master → 05_pipeline_tensores → 06_grafo_enriquecido →
  07_baseline_persistencia → 08_entrenar_modelos → 09_analisis_temporal →
  10_ensembles.
- Configuración central en config.py.
- En el TFG se detectó un error de 8 meses en los datos oficiales.
- Modelo de la demo: gat_plus.pt (GAT+Plus), el mismo que usan
  09_analisis_temporal.py y 10_ensembles.py.

## Cifras
- La demo y el README usan las métricas verificadas con los datos
  regenerados: pipeline_v2_recon/demo/metricas.json.
- La memoria del TFG reporta una mejora del 24,5 % frente a la
  persistencia; la reconstrucción del repositorio da la cifra de
  metricas.json. La demo lo explica con una nota.
- El grafo de OSM se descargó de nuevo en octubre de 2026: 2.855 aristas
  en lugar de las 2.863 del TFG. Los 1.035 nodos coinciden y el baseline
  de persistencia coincide exactamente (MAE en test de 38,30 veh/h).

## Reglas
- No reentrenar: se usan los pesos guardados en pipeline_v2_recon/models/.
- No modificar: los scripts 01–10, config.py, modelos/inferencia.py (lo
  usan 09 y 10), las carpetas de datos brutos ni pipeline_v2/.
- Las carpetas de datos brutos de la raíz (estat_trafic_VLC-main,
  espires_VLC2-main, espires_bici_VLC-main, valenbici-master,
  datos_tiempo y fusion_ruido_sensores) son uniones (junctions) a la
  carpeta original: solo se leen; nunca se modifican, mueven ni borran.
- Código nuevo de la demo en app/; la inferencia, en app/inferencia_demo.py,
  que recibe una sola ventana. Tests en tests/.
- Muestra de la demo en pipeline_v2_recon/demo/: una ventana de unos
  20 pasos × 1035 nodos, más edge_index y node_order.
- Dependencias de la demo en requirements-app.txt; en Docker, torch en
  versión solo CPU.
- Python con tipado en las funciones nuevas y docstrings en español.
- Ejecutar pytest tras cada cambio.
- Ningún dato completo ni ninguna clave en el repositorio.
- Antes de cualquier cambio grande, mostrar el plan y esperar aprobación.

## Entorno
- Windows con PowerShell.
- Python del .venv de la raíz.
- Los scripts 01–10 se ejecutan con Python 3.10 y requirements.txt (el
  entorno del TFG); el .venv es para la demo.
- Ejecutar siempre desde la raíz del repositorio (RAIZ = '.' en config.py).
- Avast intercepta HTTPS en este equipo: si falla alguna descarga por
  certificados, avisar en lugar de desactivar la verificación.

## Comandos
- Tests: pytest -q
- Demo: streamlit run app/app.py
- Docker (build): docker build -t trafico-valencia .
- Docker (run): docker run -p 8501:8501 trafico-valencia
