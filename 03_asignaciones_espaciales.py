"""
03_asignaciones_espaciales.py — Coordenadas de sensores exógenos (bici,
Valenbisi) + asignación al nodo de tráfico más cercano (cKDTree).

Requiere haber ejecutado antes 02_construccion_grafo.py
(necesita node_order_1035.csv y espira_to_tramo.csv).

Notas de implementación:

- Se usa `cKDTree` para el vecino más cercano.
- `normalize_id()` se aplica solo a `id_tramo` (igual que hace
  `04_fusion_master.py`); `id_sensor`/`id_estacion` se mantienen como
  enteros, coherente con el tipo que producen `01_extraccion.py` y
  `02_construccion_grafo.py`.
- Las distancias se calculan proyectando lat/lon a un plano local en
  metros (escalando la longitud por cos(lat_ref)) antes de construir el
  árbol, en vez de aplicar el mismo factor (111.320 m/°) a latitud y
  longitud por igual — a la latitud de Valencia (39,47°N), 1° de
  longitud son ~85.934 m reales, no 111.320 m, así que ese atajo
  sobreestimaría casi un 30% las distancias con componente este-oeste.
- La extracción de coordenadas de bici/Valenbisi recorre todos los ZIPs
  disponibles sin cortes anticipados, e imprime el nº de sensores
  encontrados.
- La corrección de coordenadas corruptas de ruido se aplica en origen,
  en `01_extraccion.py`; este script solo hace una comprobación pasiva:
  si detecta coordenadas de ruido fuera del rango geográfico de
  Valencia, avisa por si la corrección de `01` no cubre el caso.

Genera:
  - eda_grafo/coords_espiras_bici_v2.csv
  - eda_grafo/coords_valenbici.csv
  - master/asignaciones_espaciales.csv
"""

import zipfile
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from config import RAIZ, PROC, GRAPH_DIR, EDA, MASTER_DIR

# ── Latitud de referencia de Valencia, para proyectar longitud a metros ──
LAT_REF = 39.47


# ══════════════════════════════════════════════════════════════
# Normalización de identificadores heterogéneos (int, float con .0,
# str) a string limpio.
# ══════════════════════════════════════════════════════════════
def normalize_id(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s.endswith('.0'):
        s = s[:-2]
    try:
        return str(int(float(s)))
    except (ValueError, TypeError):
        return s


# ══════════════════════════════════════════════════════════════
# Proyección local lat/lon -> metros (corrige la anisotropía este-oeste)
# ══════════════════════════════════════════════════════════════
def to_local_meters(lat, lon, lat_ref=LAT_REF):
    """
    Proyecta coordenadas geográficas a un plano local aproximado en
    metros, usando cos(lat_ref) para escalar la longitud. Válido para
    distancias cortas (< unas pocas decenas de km), como es el caso aquí
    (todo dentro del término municipal de Valencia).
    """
    y = np.asarray(lat, dtype=np.float64) * 111_320.0
    x = np.asarray(lon, dtype=np.float64) * 111_320.0 * np.cos(np.radians(lat_ref))
    return np.column_stack([x, y])


# ══════════════════════════════════════════════════════════════
# Coordenadas — espiras de bici (idpm + geo_point_2d, sin cortes)
# ══════════════════════════════════════════════════════════════
def extraer_coords_bici():
    carpeta = RAIZ / 'espires_bici_VLC-main'
    zip_files = sorted(carpeta.glob('*.zip'))
    print(f'Coords espiras bici — procesando {len(zip_files)} ZIPs (sin corte)...')

    coords = {}
    for i, zf in enumerate(zip_files):
        try:
            with zipfile.ZipFile(zf, 'r') as z:
                csv_names = [n for n in z.namelist() if n.endswith('.csv')]
                if not csv_names:
                    continue
                with z.open(csv_names[0]) as f:
                    df = pd.read_csv(f, sep=';', nrows=200, encoding='utf-8',
                                      on_bad_lines='skip')
                    cols = {c.lower().strip(): c for c in df.columns}
                    id_col = cols.get('idpm')
                    geo_col = cols.get('geo_point_2d') or cols.get('geo point')
                    if not id_col or not geo_col:
                        continue
                    for _, row in df.iterrows():
                        sid, geo = row[id_col], row[geo_col]
                        if pd.isna(geo) or pd.isna(sid):
                            continue
                        try:
                            sid_int = int(float(sid))
                            parts = str(geo).replace(',', ' ').split()
                            lat, lon = float(parts[0]), float(parts[1])
                            if 39.0 < lat < 40.0 and -1.0 < lon < 0.0:
                                coords[sid_int] = (lat, lon)
                        except (ValueError, IndexError):
                            continue
        except Exception:
            continue
        if (i + 1) % 100 == 0:
            print(f'    ZIP {i+1}/{len(zip_files)} — {len(coords)} sensores hasta ahora')

    df_coords = pd.DataFrame([
        {'id_sensor': k, 'lat': v[0], 'lon': v[1]} for k, v in coords.items()
    ]).sort_values('id_sensor').reset_index(drop=True)

    df_coords.to_csv(EDA / 'coords_espiras_bici_v2.csv', index=False)
    print(f'  Guardado: coords_espiras_bici_v2.csv ({len(df_coords)} sensores)')
    return df_coords


# ══════════════════════════════════════════════════════════════
# Coordenadas — estaciones Valenbici (doble formato, sin cortes)
# ══════════════════════════════════════════════════════════════
def extraer_coords_valenbici():
    carpeta = RAIZ / 'valenbici-master'
    zip_files = sorted(carpeta.glob('*.zip'))
    print(f'Coords valenbici — procesando {len(zip_files)} ZIPs (sin corte)...')

    coords = {}
    for i, zf in enumerate(zip_files):
        try:
            with zipfile.ZipFile(zf, 'r') as z:
                csv_names = [n for n in z.namelist() if n.endswith('.csv')]
                if not csv_names:
                    continue
                with z.open(csv_names[0]) as f:
                    df = pd.read_csv(f, sep=None, engine='python', encoding='latin-1')

                    if 'number_' in df.columns:
                        id_col = 'number_'          # formato viejo (2022-2023)
                    elif 'Numero' in df.columns:
                        id_col = 'Numero'            # formato nuevo (2024+)
                    else:
                        continue

                    if 'geo_point_2d' not in df.columns:
                        continue

                    for _, row in df.iterrows():
                        sid_raw, geo = row[id_col], row['geo_point_2d']
                        if pd.isna(geo) or pd.isna(sid_raw):
                            continue
                        try:
                            sid = int(float(sid_raw))
                            parts = str(geo).replace(',', ' ').split()
                            lat, lon = float(parts[0]), float(parts[1])
                            if 39.0 < lat < 40.0 and -1.0 < lon < 0.0:
                                coords[sid] = (lat, lon)
                        except (ValueError, IndexError):
                            continue
        except Exception:
            continue
        if (i + 1) % 150 == 0:
            print(f'    ZIP {i+1}/{len(zip_files)} — {len(coords)} estaciones hasta ahora')

    df_coords = pd.DataFrame([
        {'id_estacion': k, 'lat': v[0], 'lon': v[1]} for k, v in coords.items()
    ]).sort_values('id_estacion').reset_index(drop=True)

    df_coords.to_csv(EDA / 'coords_valenbici.csv', index=False)
    print(f'  Guardado: coords_valenbici.csv ({len(df_coords)} estaciones)')
    return df_coords


# ══════════════════════════════════════════════════════════════
# Coordenadas — ruido (con corrección opcional de coords corruptas)
# ══════════════════════════════════════════════════════════════
def cargar_coords_ruido():
    df_ruido_coords = pd.read_parquet(PROC / 'ruido.parquet',
                                       columns=['id_sensor_ruido', 'lat', 'lon'])
    coords_ruido = (df_ruido_coords
                     .groupby('id_sensor_ruido')[['lat', 'lon']]
                     .first()
                     .reset_index())

    print(f'  Coords ruido: {len(coords_ruido)} sensores')

    # Chequeo de seguridad pasivo: la corrección de coordenadas corruptas
    # ya se aplica en origen, en 01_extraccion.py. Si aquí aparecen
    # coordenadas fuera del rango de Valencia, revisar extraer_ruido() en
    # 01_extraccion.py en vez de parchearlo aquí.
    fuera_rango = coords_ruido[
        ~coords_ruido['lat'].between(39.0, 40.0) |
        ~coords_ruido['lon'].between(-1.0, 0.0)
    ]
    if len(fuera_rango) > 0:
        print(f'    ⚠️ {len(fuera_rango)} sensores de ruido con coords fuera de rango '
              f'de Valencia — revisar extraer_ruido() en 01_extraccion.py.')

    return coords_ruido


# ══════════════════════════════════════════════════════════════
# Asignación espacial por cKDTree, en plano local proyectado a metros
# ══════════════════════════════════════════════════════════════
def asignar_espacialmente():
    node_order = pd.read_csv(GRAPH_DIR / 'node_order_1035.csv')
    espira_to_tramo = pd.read_csv(GRAPH_DIR / 'espira_to_tramo.csv')
    # Solo id_tramo se normaliza — igual que hace 04_fusion_master.py de
    # verdad. id_sensor/id_estacion se mantienen como enteros.
    espira_to_tramo['id_tramo'] = espira_to_tramo['id_tramo'].apply(normalize_id)

    node_order_ids = node_order.copy()

    coords_bici = pd.read_csv(EDA / 'coords_espiras_bici_v2.csv')
    coords_valen = pd.read_csv(EDA / 'coords_valenbici.csv')

    coords_ruido = cargar_coords_ruido()

    print(f'\n  Nodos (espiras tráfico): {len(node_order_ids)}')
    print(f'  Coords espiras bici: {len(coords_bici)}')
    print(f'  Coords valenbici: {len(coords_valen)}')
    print(f'  Coords ruido: {len(coords_ruido)}')

    nodos_m = to_local_meters(node_order_ids['lat'], node_order_ids['lon'])
    asignaciones = node_order_ids.merge(espira_to_tramo, on='id_sensor', how='left')

    def asignar_fuente(df_coords, id_col, nombre_dist, nombre_id_out):
        pts_m = to_local_meters(df_coords['lat'], df_coords['lon'])
        tree = cKDTree(pts_m)
        d, idx = tree.query(nodos_m)
        out = pd.DataFrame({
            'id_sensor': node_order_ids['id_sensor'],
            nombre_id_out: df_coords.iloc[idx][id_col].values,
            nombre_dist: d,  # ya en metros, sin necesidad de *111_320
        })
        print(f'  → espira→{nombre_id_out}: dist. mediana {out[nombre_dist].median():.0f}m '
              f'(corregido; antes del fix, sesgado ~+29,5% en componente este-oeste)')
        return out

    if len(coords_bici) > 0:
        espira_a_bici = asignar_fuente(coords_bici, 'id_sensor', 'dist_bici_m', 'id_bici_cercano')
        asignaciones = asignaciones.merge(espira_a_bici, on='id_sensor', how='left')

    if len(coords_valen) > 0:
        espira_a_valen = asignar_fuente(coords_valen, 'id_estacion', 'dist_valen_m', 'id_valen_cercano')
        asignaciones = asignaciones.merge(espira_a_valen, on='id_sensor', how='left')

    if len(coords_ruido) > 0:
        espira_a_ruido = asignar_fuente(coords_ruido, 'id_sensor_ruido', 'dist_ruido_m', 'id_ruido_cercano')
        asignaciones = asignaciones.merge(espira_a_ruido, on='id_sensor', how='left')

    MASTER_DIR.mkdir(parents=True, exist_ok=True)
    asignaciones.to_csv(MASTER_DIR / 'asignaciones_espaciales.csv', index=False)
    print(f'\n  Guardado: master/asignaciones_espaciales.csv')
    print(f'  Columnas: {list(asignaciones.columns)}')
    return asignaciones


def main():
    extraer_coords_bici()
    extraer_coords_valenbici()
    asignar_espacialmente()


if __name__ == '__main__':
    main()
