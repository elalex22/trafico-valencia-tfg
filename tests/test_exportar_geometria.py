"""Tests de app/exportar_geometria.py y de geometria.geojson."""
import json

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, Point, shape

from app import exportar_geometria as eg


@pytest.fixture(scope='module')
def capas() -> dict[str, list[dict]]:
    with open(eg.SALIDA, encoding='utf-8') as f:
        features = json.load(f)['features']
    return {c: [x for x in features if x['properties']['capa'] == c] for c in ('tramo', 'sensor')}


@pytest.fixture(scope='module')
def ids_nodos() -> set[int]:
    return set(pd.read_csv(eg.DEMO_DIR / 'node_order.csv')['id_sensor'])


def test_ligero() -> None:
    assert eg.SALIDA.stat().st_size < 400_000


def test_un_sensor_por_nodo(capas, ids_nodos) -> None:
    ids = [f['properties']['id_sensor'] for f in capas['sensor']]
    assert len(ids) == 1035 and set(ids) == ids_nodos


def test_tramos_reparten_los_sensores(capas, ids_nodos) -> None:
    assert len(capas['tramo']) == 293
    sensores = [s for f in capas['tramo'] for s in f['properties']['sensores']]
    assert len(sensores) == len(set(sensores)) and set(sensores) == ids_nodos
    tramo_de = {s: f['properties']['id_tramo'] for f in capas['tramo']
                for s in f['properties']['sensores']}
    assert all(tramo_de[f['properties']['id_sensor']] == f['properties']['id_tramo']
               for f in capas['sensor'])
    assert all(f['properties']['nombre'] for f in capas['tramo'])


def test_geometrias_validas_en_valencia(capas) -> None:
    for f in capas['tramo'] + capas['sensor']:
        g = shape(f['geometry'])
        assert g.geom_type == 'LineString' and len(g.coords) >= 2 and g.length > 0
        lon, lat = np.array(g.coords).T
        assert (39.3 < lat).all() and (lat < 39.6).all() and (-0.5 < lon).all() and (lon < -0.2).all()


def test_aristas_recortadas(capas) -> None:
    largos = [eg._a_metros(shape(f['geometry'])).length for f in capas['sensor']]
    assert max(largos) <= 2 * eg.RECORTE_M + 5


def test_recortar() -> None:
    # Línea recta de ~1 km hacia el este por la latitud de València.
    linea = LineString([(-0.40, 39.47), (-0.40 + 1000 / eg.M_POR_GRADO_LON, 39.47)])
    centro = eg._a_metros(eg.recortar(linea, Point(-0.40 + 500 / eg.M_POR_GRADO_LON, 39.4701)))
    assert centro.length == pytest.approx(150, abs=0.5)
    extremo = eg._a_metros(eg.recortar(linea, Point(-0.40, 39.47)))
    assert extremo.length == pytest.approx(75, abs=0.5)
    corta = LineString([(-0.40, 39.47), (-0.40 + 50 / eg.M_POR_GRADO_LON, 39.47)])
    assert eg._a_metros(eg.recortar(corta, Point(-0.40, 39.47))).length == pytest.approx(50, abs=0.5)
