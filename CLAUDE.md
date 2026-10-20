# Proyecto: predicción del tráfico urbano de Valencia con GNN
## Contexto
- TFG del Grado en Ciencia de Datos (UPV), calificación 10.
- Pipeline en scripts numerados que se ejecutan en orden:
01_extraccion → 02_construccion_grafo → 03_asignaciones_espaciales →
04_fusion_master → 05_pipeline_tensores → 06_grafo_enriquecido →
07_baseline_persistencia → 08_entrenar_modelos → 09_analisis_temporal →
10_ensembles.
- Configuración central en config.py.
- El mejor modelo (GAT) mejora un 24,5 % al baseline de persistencia.
- En el TFG se detectó un error de 8 meses en los datos oficiales.
## Reglas
- NO reorganizar ni renombrar los scripts 01–10.
- NO cambiar la lógica de los modelos ni el cálculo de métricas.
- Código nuevo de la demo en app/, tests en tests/, módulo de
inferencia en inferencia.py.
- Python con tipado en funciones nuevas y docstrings en español.
- Ejecutar pytest tras cada cambio.
- Ningún dato completo nuevo en el repo; ninguna clave en el código.
- Antes de cambios grandes, mostrar el plan y esperar aprobación.
## Comandos
- Tests: pytest -q
- Demo: streamlit run app/app.py
- Docker: docker build -t trafico-valencia . ; docker run -p 8501:8501
trafico-valencia
