# Proyecto: predicción del tráfico urbano de Valencia con GNN

## Contexto
- TFG del Grado en Ciencia de Datos (UPV), calificación 10.
- Pipeline en scripts numerados que se ejecutan en orden:
  01_extraccion → 02_construccion_grafo → 03_asignaciones_espaciales →
  04_fusion_master → 05_pipeline_tensores → 06_grafo_enriquecido →
  07_baseline_persistencia → 08_entrenar_modelos → 09_analisis_temporal →
  10_ensembles.
- Configuración central en config.py.
- En el TFG se detectó un error de 8 meses en los datos oficiales: datos
  corruptos (casi todo ceros) de octubre de 2024 a mayo de 2025,
  reconocidos por el portal y excluidos de los splits (memoria, p. 31 y 37).
- Del 1 al 12 de junio de 2025 la fuente estuvo congelada (11,3 días con
  valores repetidos, alrededor del 10 % del test); las métricas lo
  incluyen.
- Modelo de la demo: gat_plus.pt (GAT+Plus), el mismo que usan
  09_analisis_temporal.py y 10_ensembles.py.

## Cifras
- La demo y el README usan las métricas verificadas con los datos
  regenerados: pipeline_v2_recon/demo/metricas.json.
- pipeline_v2_recon es una reconstrucción del pipeline del TFG. Su
  baseline de persistencia (38,30 veh/h) coincide con el baseline.json del
  repositorio, no con la memoria, que reporta 37,55 (Tabla 6.1, p. 50).
- En la memoria, GAT+Plus da 28,35 y la mejora es del 24,5 % (Tabla 6.2,
  p. 53); en la reconstrucción, 28,71 y un 25,0 %. Las coberturas de los
  splits también difieren de la Tabla 4.2. La demo lo explica con una nota.
- El grafo de OSM se descargó de nuevo en octubre de 2026: 2.855 aristas
  en lugar de las 2.863 del TFG; los 1.035 nodos coinciden.

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
- Muestra de la demo en pipeline_v2_recon/demo/ (la genera
  app/crear_muestra_demo.py): dos semanas de test, semana_normal.npz
  (15–21 de septiembre de 2025) y semana_san_juan.npz (23–29 de junio de
  2025; el domingo 29 no hay datos de tráfico desde las 9:00),
  cada una un .npz comprimido en float32 con 692 pasos × 1035 nodos (672
  de la semana + 20 de contexto para gat_plus), más node_order.csv,
  edge_index_final.npy, stats.json y config.json.
- La app (app/app.py) solo usa los instantes :15 (la intensidad es horaria
  y cambia a y cuarto) y lee predicciones_<semana>.npz, que genera
  app/precalcular_predicciones.py; si cambia la muestra o el modelo, hay
  que volver a ejecutarlo. Mapa con teselas de OpenStreetMap (CartoDB pide
  clave). Sensores sin datos en gris. Las líneas salen de geometria.geojson
  (app/exportar_geometria.py): tramos oficiales (geo_shape de estat_trafic)
  por defecto y, con un interruptor, la arista de OSM de cada sensor
  recortada a ±75 m. Fuente de los datos: Ajuntament de València – Dades
  Obertes (CC BY 4.0); meteorología, AEMET; viario y mapa base, OSM.
- Dependencias de la demo en requirements-app.txt (con starlette fijado:
  la 1.9.0 rompe el gzip de streamlit 1.59.1); torch solo CPU en
  requirements-torch-cpu.txt. Los usan Docker, la CI y el despliegue.
- Despliegue de la demo en Streamlit Community Cloud (no Hugging Face):
  rama main, archivo principal app/app.py, Python 3.11 en Advanced
  settings; instala app/requirements.txt, que solo remite a los dos
  archivos anteriores. Memoria medida: ~210 MB, ~600 MB tras el cálculo
  en vivo. URL pública pendiente (marcador DEMO_URL en el README).
- Python con tipado en las funciones nuevas y docstrings en español.
- Ejecutar pytest tras cada cambio.
- Ningún dato completo ni ninguna clave en el repositorio.
- Antes de cualquier cambio grande, mostrar el plan y esperar aprobación.

## Entorno
- Windows con PowerShell.
- Todo (scripts 01–10, demo y tests) se ejecuta con .venv\Scripts\python.exe,
  un .venv de Python 3.10 (el del TFG) con requirements.txt,
  requirements-app.txt y requirements-dev.txt.
- Ejecutar siempre desde la raíz del repositorio (RAIZ = '.' en config.py).
- Avast intercepta HTTPS en este equipo: si falla alguna descarga por
  certificados, avisar en lugar de desactivar la verificación.

## Comandos
- Lint: ruff check . (configurado en pyproject.toml, solo para el código
  nuevo: app/, tests/, verificar_resultados.py y analisis_adicional.py)
- Tests: pytest -q
- CI: .github/workflows/ci.yml (ruff, pytest y build + healthcheck de
  Docker); torch CPU sale de requirements-torch-cpu.txt, igual que en Docker
- Demo: streamlit run app/app.py
- Docker (build): docker build -t trafico-valencia .
- Docker (run): docker run -p 8501:8501 trafico-valencia
