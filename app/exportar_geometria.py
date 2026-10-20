"""
exportar_geometria.py — Exporta a pipeline_v2_recon/demo/geometria.geojson
la geometría que dibuja el mapa de la demo, simplificada:

- capa 'tramo' (vista B, por defecto): la línea oficial (geo_shape) de los
  293 tramos de estat_trafic a los que 02 asigna alguna espira
  (espira_to_tramo.csv), con su denominación y la lista de sensores.
- capa 'sensor' (vista A): para cada uno de los 1.035 sensores, la arista
  de OSM a la que 02 lo ajustó (mapmatching_prueba.csv sobre
  valencia_drive.graphml), recortada a ±75 m alrededor del sensor.

La geometría de los tramos se lee (solo lectura) del ZIP más reciente de
estat_trafic_VLC-main que los contenga todos. Ejecutar desde la raíz del
repositorio tras 02:
    .venv\\Scripts\\python.exe app\\exportar_geometria.py
"""

import json
import sys
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point, mapping, shape
from shapely.ops import substring

RAIZ_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ_REPO))

from config import EDA, GRAPH_DIR, MODELS_DIR, RAIZ  # noqa: E402

DEMO_DIR = MODELS_DIR.parent / 'demo'
SALIDA = DEMO_DIR / 'geometria.geojson'
RECORTE_M = 75          # metros a cada lado del sensor en la vista A
TOLERANCIA = 0.00002    # simplificación (grados, unos 2 m)
DECIMALES = 5           # unos 1 m
M_POR_GRADO_LAT = 110_540
M_POR_GRADO_LON = 111_320 * np.cos(np.radians(39.47))  # latitud de València


def _a_metros(g: LineString | Point) -> LineString | Point:
    """Proyección local equirectangular (lon, lat) → metros."""
    from shapely import transform
    return transform(g, lambda c: c * [M_POR_GRADO_LON, M_POR_GRADO_LAT])


def _a_grados(g: LineString) -> LineString:
    """Inversa de _a_metros."""
    from shapely import transform
    return transform(g, lambda c: c / [M_POR_GRADO_LON, M_POR_GRADO_LAT])


def _redondear(geom: dict[str, Any]) -> dict[str, Any]:
    """Redondea las coordenadas de una geometría GeoJSON LineString."""
    return {'type': geom['type'],
            'coordinates': [[round(x, DECIMALES), round(y, DECIMALES)]
                            for x, y in geom['coordinates']]}


def leer_tramos_oficiales(ids: set[int]) -> pd.DataFrame:
    """id_tramo, nombre y geo_shape de los tramos `ids`, del ZIP más
    reciente de estat_trafic que los contenga."""
    zips = []
    for zf in (RAIZ / 'estat_trafic_VLC-main').glob('*.zip'):
        fecha = pd.to_datetime(zf.stem, format='%d-%m-%Y', errors='coerce')
        if pd.notna(fecha):
            zips.append((fecha, zf))
    for _, zf in sorted(zips, reverse=True):
        with zipfile.ZipFile(zf) as z:
            nombre_csv = next(n for n in z.namelist() if n.endswith('.csv'))
            with z.open(nombre_csv) as f:
                df = pd.read_csv(f, sep=';', encoding='utf-8', on_bad_lines='skip')
        col_id = next(c for c in df.columns if 'Id. Tram' in c)
        col_nombre = next(c for c in df.columns if 'Denomina' in c)
        df = df.dropna(subset=[col_id, 'geo_shape'])
        df['id_tramo'] = df[col_id].astype(float).astype(int)
        df = df.drop_duplicates('id_tramo').rename(columns={col_nombre: 'nombre'})
        if ids <= set(df['id_tramo']):
            print(f'  Geometría de tramos: {zf.name}')
            return df.loc[df['id_tramo'].isin(ids), ['id_tramo', 'nombre', 'geo_shape']]
    raise RuntimeError('Ningún ZIP de estat_trafic contiene todos los tramos usados')


def features_tramos(asignacion: pd.DataFrame) -> list[dict[str, Any]]:
    """Capa 'tramo': una LineString oficial por tramo con sensores."""
    tramos = leer_tramos_oficiales(set(asignacion['id_tramo']))
    sensores = asignacion.groupby('id_tramo')['id_sensor'].apply(sorted)
    features = []
    for fila in tramos.sort_values('id_tramo').itertuples():
        linea = shape(json.loads(fila.geo_shape)).simplify(TOLERANCIA)
        features.append({
            'type': 'Feature',
            'properties': {'capa': 'tramo', 'id_tramo': int(fila.id_tramo),
                           'nombre': str(fila.nombre).strip(),
                           'sensores': [int(s) for s in sensores[fila.id_tramo]]},
            'geometry': _redondear(mapping(linea)),
        })
    return features


def recortar(arista: LineString, sensor: Point, metros: float = RECORTE_M) -> LineString:
    """Trozo de `arista` de ±`metros` alrededor de la proyección del sensor."""
    linea_m, punto_m = _a_metros(arista), _a_metros(sensor)
    d = linea_m.project(punto_m)
    trozo = substring(linea_m, max(0.0, d - metros), min(linea_m.length, d + metros))
    if trozo.geom_type != 'LineString' or trozo.length == 0:
        trozo = linea_m
    return _a_grados(trozo)


def features_sensores(asignacion: pd.DataFrame) -> list[dict[str, Any]]:
    """Capa 'sensor': la arista de OSM de cada sensor, recortada."""
    import osmnx as ox  # aquí y no arriba: los tests y la CI no necesitan osmnx
    G = ox.load_graphml(GRAPH_DIR / 'valencia_drive.graphml')
    mm = pd.read_csv(EDA / 'mapmatching_prueba.csv')
    tramo = asignacion.set_index('id_sensor')['id_tramo']
    features = []
    for f in mm.sort_values('id_sensor').itertuples():
        u, v, k = int(f.osm_u), int(f.osm_v), int(f.osm_key)
        arista = G.edges[u, v, k].get('geometry') or LineString(
            [(G.nodes[u]['x'], G.nodes[u]['y']), (G.nodes[v]['x'], G.nodes[v]['y'])])
        linea = recortar(arista, Point(f.lon, f.lat)).simplify(TOLERANCIA)
        features.append({
            'type': 'Feature',
            'properties': {'capa': 'sensor', 'id_sensor': int(f.id_sensor),
                           'id_tramo': int(tramo[f.id_sensor])},
            'geometry': _redondear(mapping(linea)),
        })
    return features


def main() -> None:
    """Genera geometria.geojson con las dos capas."""
    nodos = pd.read_csv(GRAPH_DIR / 'node_order_1035.csv')
    asignacion = pd.read_csv(GRAPH_DIR / 'espira_to_tramo.csv')
    asignacion = asignacion[asignacion['id_sensor'].isin(nodos['id_sensor'])]

    features = features_tramos(asignacion) + features_sensores(asignacion)
    with open(SALIDA, 'w', encoding='utf-8') as f:
        json.dump({'type': 'FeatureCollection', 'features': features}, f,
                  ensure_ascii=False, separators=(',', ':'))
    n = pd.Series([x['properties']['capa'] for x in features]).value_counts()
    print(f'  {SALIDA.name}: {n.get("tramo", 0)} tramos, {n.get("sensor", 0)} sensores, '
          f'{SALIDA.stat().st_size / 1e3:.0f} KB')


if __name__ == '__main__':
    main()
