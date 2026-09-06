# Pipeline de predicción de tráfico — Valencia Smart City

**Autor:** Alejandro Navarro del Castillo
**Trabajo de Fin de Grado — Grado en Ciencia de Datos, Universitat Politècnica de València**

Pipeline de 10 scripts que va desde la extracción de datos brutos hasta el
entrenamiento y evaluación de 9 arquitecturas de red neuronal para predecir
intensidad de tráfico a 15 minutos vista. Corresponde a la Tabla 11.1 (Anexo
1) de la memoria del TFG.

## 1. Estructura de carpetas necesaria ANTES de ejecutar nada

```
Smart_Cities/                          <- raíz del proyecto (RAIZ en config.py)
├── config.py                          <- rutas y constantes (ver sección 3)
├── 01_extraccion.py ... 10_ensembles.py
├── modelos/                           <- paquete de arquitecturas y entrenamiento
│   ├── __init__.py
│   ├── arquitecturas.py               <- las 9 clases de red (MLPSimple, GATCompletoPlus, ...)
│   ├── datasets.py                    <- DatasetSimple / DatasetCompleto / DatasetCompletoPlus
│   ├── entrenamiento.py               <- entrenar(), persistencia_mae()
│   ├── inferencia.py                  <- predecir_modelo(), alinear()
│   └── registro.py                    <- MODEL_REGISTRY, GATS_PARA_PROMEDIO
├── datos/                             <- utilidades compartidas
│   ├── __init__.py
│   ├── utils_datos.py                 <- normalize_id(), load_and_filter(), aggregate_15min(), aplicar_fix_ceros_extendidos()
│   └── io_tensores.py                 <- cargar_split_para_modelo()
│
│  --- Datos brutos (input, SOLO LECTURA — coloca aquí tus descargas) ---
├── estat_trafic_VLC-main/             <- ZIPs de estado de tramos (Ayuntamiento Valencia)
├── espires_VLC2-main/                 <- ZIPs de espiras de tráfico (idpm + ih)
├── espires_bici_VLC-main/             <- ZIPs de espiras de bici
├── valenbici-master/                  <- ZIPs de estaciones Valenbisi (doble formato viejo/nuevo)
├── datos_tiempo/                      <- JSONs diarios de AEMET
└── fusion_ruido_sensores/             <- CSVs "carrer-*.csv" de sensores de ruido (Ruzafa)
```

Todo lo que generan los scripts (Parquet, tensores, checkpoints, resultados)
se escribe bajo una carpeta de salida separada (ver `config.py`) — nunca
dentro de las carpetas de datos brutos.

## 2. Entorno de Python

Ver `requirements.txt`. Antes de instalar nada, si quieres versiones
**exactas** (no solo rangos compatibles), ejecuta:

```
python capturar_entorno.py
```

y pégame `entorno_real.txt` — no tengo acceso a tu entorno, así que el
`requirements.txt` que te doy son rangos deducidos del código, no cifras
verificadas en tu máquina.

`torch` y `torch-geometric` instálalos siguiendo sus selectores oficiales
(dependen de tu versión de CUDA), no solo con `pip install` a ciegas.

## 3. `config.py` — variables que deben existir

Este README no sustituye a tu `config.py` real (que ya tienes y no hay que
tocar) — es la lista de lo que los 10 scripts esperan encontrar ahí, por si
alguna vez necesitas reconstruirlo o revisarlo:

| Variable | Para qué se usa |
|---|---|
| `RAIZ` | Carpeta raíz (relativa a desde dónde se ejecuta python) |
| `PROC` | `.../data/processed` — salida de `01` |
| `GRAPH_DIR` | `.../data/grafo` — salida de `02`, `06` |
| `EDA` | `.../data/eda_grafo` — coords intermedias de `02`, `03` |
| `MASTER_DIR` | `.../data/master_final` — salida de `04`, entrada de asignaciones de `03` |
| `TENSOR_DIR` | `.../data/tensors_final` — salida de `05`, entrada de `06`-`10` |
| `MODELS_DIR` | `.../models` — checkpoints, escritos por `08`, leídos por `09`/`10` |
| `RESULTADOS_DIR` | `.../resultados` — JSON de métricas de `07`, `08`, `09`, `10` |
| `TROZOS_POR_SPLIT` | Fechas de corte de train/val/test (Tabla 4.2) |
| `MAX_GAP_TIMESTEPS`, `RACHA_CEROS`, `LAG_24H`, `LAG_7D` | Parámetros de limpieza/lags de `04` |
| `UMBRAL_MAX_Y` | Tope de outliers del target en `05` |
| `FEATURES_DIN`, `FEATURES_STATIC`, `FEATURES_A_ESTANDARIZAR`, `MASK_POR_FEATURE` | Definición de las 22 features dinámicas + 2 estáticas (Tabla 4.3) |
| `UMBRAL_CORR`, `K_MAX_CORR`, `MIN_ACTIVIDAD_NODO`, `TIMESTEP_COBERTURA_MIN` | Parámetros del grafo enriquecido en `06` |
| `WINDOW`, `HORIZON` | Ventana temporal (12 pasos) y horizonte de predicción |

**Importante — seguridad de datos**: en el `config.py` de este proyecto,
`RAIZ` apunta a la raíz del repo pero `DATA` (de donde salen `PROC`,
`GRAPH_DIR`, etc.) apunta a una subcarpeta de reconstrucción separada de
`pipeline_v2/` (donde viven los checkpoints reales ya entrenados). Los
scripts **nunca escriben** en las carpetas de datos brutos ni en
`pipeline_v2/` — solo leen de ahí.

## 4. Los 10 scripts (Tabla 11.1)

| # | Script | Qué hace | Requiere | Genera |
|---|---|---|---|---|
| 01 | `01_extraccion.py` | Extrae las 6 fuentes brutas (ZIPs/JSON/CSV) a Parquet, con limpieza de meteo (rango, float32) y coordenadas de ruido | Carpetas de datos brutos | `processed/*.parquet` |
| 02 | `02_construccion_grafo.py` | Descarga el grafo viario (OSMnx), map-matching de espiras, construye aristas (regla secuencial + misma calle), asigna espira→tramo | `01` | `grafo/node_order_1035.csv`, `edge_index_final.npy`, `espira_to_tramo.csv` |
| 03 | `03_asignaciones_espaciales.py` | Asigna bici/Valenbisi/ruido más cercano a cada espira (cKDTree con proyección local, sin sesgo geográfico) | `02` | `master_final/asignaciones_espaciales.csv` |
| 04 | `04_fusion_master.py` | Fusiona las 6 fuentes sobre el timeline (por trozos con buffer de 7 días), festivos, rachas de sensor roto, interpolación, lags | `01`, `02`, `03` | `master_final/master_{split}.parquet` |
| 05 | `05_pipeline_tensores.py` | Tensoriza a (T,N,F), limpia el target (-1/outliers→log1p), normaliza features | `04` | `tensors_final/*.npy`, `config.json`, `stats.json` |
| 06 | `06_grafo_enriquecido.py` | Añade aristas por correlación histórica de tráfico al grafo físico | `02`, `05` | `grafo/edge_index_enriched.npy` |
| 07 | `07_baseline_persistencia.py` | MAE/RMSE del modelo trivial Ŷ[t+1]=Y[t] | `05` | `resultados/baseline.json` |
| 08 | `08_entrenar_modelos.py` | Entrena las 9 arquitecturas (MLP/GCN/GAT, hiperparámetros exactos por modelo) | `05`, `06` | `models/*.pt`, `resultados/entrenamiento.json` |
| 09 | `09_analisis_temporal.py` | GAT+Plus vs persistencia, desglosado por franja/hora/día | `08` | `resultados/analisis_temporal.json` |
| 10 | `10_ensembles.py` | 4 estrategias de ensemble (oracle día/hora, promedio 3 GAT, oracle hora×día) | `08` | `resultados/ensembles.json` |

## 5. Orden de ejecución

```bash
cd Smart_Cities          # imprescindible: RAIZ='.' es relativo al cwd
python 01_extraccion.py
python 02_construccion_grafo.py
python 03_asignaciones_espaciales.py
python 04_fusion_master.py
python 05_pipeline_tensores.py
python 06_grafo_enriquecido.py
python 07_baseline_persistencia.py
python 08_entrenar_modelos.py
python 09_analisis_temporal.py
python 10_ensembles.py
```

Cada script comprueba y avisa (⚠️) si algún resultado suyo no coincide con
los valores esperados (nº de nodos, tamaño de splits, parámetros por
arquitectura, MAE de persistencia...) — revisa esos avisos antes de pasar
al siguiente script si aparece alguno.

## 6. Notas conocidas

- **`08` no fija semilla aleatoria** — espera variaciones de ~1-3% entre
  ejecuciones, no resultados idénticos.
- **`06` es sensible a `UMBRAL_CORR`/`K_MAX_CORR`**: al estar el umbral
  (0.7) en zona densa de la distribución de correlaciones, pequeñísimas
  diferencias numéricas pueden mover el nº de aristas por correlación un
  par de puntos porcentuales — no indica un error si el resto de cifras
  cuadra.
- **Ruido**: la corrección de coordenadas corruptas en `01` está activa
  por si aparece algún registro con el formato corrupto descrito (más de
  un punto decimal), aunque en la práctica no siempre se encuentran casos.
