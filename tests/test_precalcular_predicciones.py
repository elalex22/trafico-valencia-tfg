"""Tests de app/precalcular_predicciones.py y de los .npz que genera."""
import numpy as np
import pandas as pd
import pytest

from app import inferencia_demo as inf
from app import precalcular_predicciones as pre


@pytest.fixture(scope='module')
def modelo():
    return inf.cargar_modelo()


@pytest.fixture(scope='module', params=inf.SEMANAS)
def guardado(request) -> tuple[str, dict[str, np.ndarray]]:
    with np.load(pre.ruta_predicciones(request.param)) as npz:
        return request.param, {k: npz[k] for k in npz.files}


def test_instantes_cuarto(guardado) -> None:
    _, p = guardado
    instantes = pd.DatetimeIndex(p['instantes'])
    assert len(instantes) == 7 * 24
    assert (instantes.minute == 15).all()
    assert (np.diff(instantes.asi8) == 3600 * 10**9).all()
    assert instantes[0].dayofweek == 0 and instantes[0].hour == 0


def test_formas_y_tipos(guardado) -> None:
    _, p = guardado
    assert p['prediccion'].shape == p['real'].shape == p['persistencia'].shape == (168, 1035)
    assert p['prediccion'].dtype == p['real'].dtype == p['persistencia'].dtype == np.float32
    assert not np.isnan(p['persistencia']).any() and (p['persistencia'] >= 0).all()
    assert p['verificable'].dtype == bool
    assert not np.isnan(p['prediccion']).any() and (p['prediccion'] >= 0).all()


def test_real_nan_donde_mascara_vacia(guardado) -> None:
    nombre, p = guardado
    semana = inf.cargar_semana(nombre)
    for k in (0, 50, 167):
        j = int(semana.instantes.get_loc(pd.Timestamp(p['instantes'][k])))
        m = semana.dataset[j][2].numpy() > 0
        assert np.array_equal(~np.isnan(p['real'][k]), m)
    assert np.array_equal(p['verificable'], ~np.isnan(p['real']).all(axis=1))


def test_coincide_con_predecir(modelo, guardado) -> None:
    nombre, p = guardado
    semana = inf.cargar_semana(nombre)
    for k in (8, 100, 160):
        r = inf.predecir(modelo, semana, pd.Timestamp(p['instantes'][k]))
        np.testing.assert_allclose(p['prediccion'][k], r.tabla['prediccion_veh_h'], rtol=1e-6)
        np.testing.assert_allclose(p['real'][k], r.tabla['real_veh_h'], rtol=1e-6)


def test_precalcular_igual_que_lo_guardado(modelo) -> None:
    """precalcular() reproduce el .npz guardado (semana normal)."""
    nuevo = pre.precalcular(modelo, inf.cargar_semana('semana_normal'))
    with np.load(pre.ruta_predicciones('semana_normal')) as npz:
        for clave in ('instantes', 'verificable'):
            assert np.array_equal(nuevo[clave], npz[clave])
        np.testing.assert_allclose(nuevo['prediccion'], npz['prediccion'], rtol=1e-6)
        np.testing.assert_array_equal(np.isnan(nuevo['real']), np.isnan(npz['real']))


def test_domingo_29_no_verificable_desde_las_9() -> None:
    with np.load(pre.ruta_predicciones('semana_san_juan')) as npz:
        instantes = pd.DatetimeIndex(npz['instantes'])
        tarde = (instantes >= pd.Timestamp('2025-06-29 09:15'))
        assert not npz['verificable'][tarde].any()
        assert npz['verificable'][instantes < pd.Timestamp('2025-06-29')].all()
