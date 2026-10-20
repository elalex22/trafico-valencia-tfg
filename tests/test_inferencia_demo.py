"""Tests de app/inferencia_demo.py sobre la muestra de pipeline_v2_recon/demo/."""
import numpy as np
import pandas as pd
import pytest
import torch

from app import inferencia_demo as inf

COLUMNAS = ['id_sensor', 'lat', 'lon', 'prediccion_veh_h', 'real_veh_h', 'error',
            'persistencia_veh_h']


@pytest.fixture(scope='module')
def modelo() -> torch.nn.Module:
    return inf.cargar_modelo()


@pytest.fixture(scope='module')
def normal() -> inf.Semana:
    return inf.cargar_semana('semana_normal')


@pytest.fixture(scope='module')
def san_juan() -> inf.Semana:
    return inf.cargar_semana('semana_san_juan')


def test_constantes_coinciden_con_config() -> None:
    import config
    assert (inf.WINDOW, inf.HORIZON) == (config.WINDOW, config.HORIZON)


@pytest.mark.parametrize('nombre, lunes', [('semana_normal', '2025-09-15'),
                                           ('semana_san_juan', '2025-06-23')])
def test_instantes_disponibles(nombre: str, lunes: str) -> None:
    instantes = inf.instantes_disponibles(inf.cargar_semana(nombre))
    assert len(instantes) == 672
    assert instantes[0] == pd.Timestamp(f'{lunes} 00:00')
    assert instantes[-1] == pd.Timestamp(lunes) + pd.Timedelta(days=6, hours=23, minutes=45)
    assert (np.diff(instantes.asi8) == 15 * 60 * 10**9).all()


def test_semana_desconocida() -> None:
    with pytest.raises(ValueError):
        inf.cargar_semana('semana_inexistente')


def test_instante_fuera_de_la_semana(modelo, normal) -> None:
    with pytest.raises(KeyError):
        inf.predecir(modelo, normal, '2025-06-02 08:00')
    with pytest.raises(IndexError):
        inf.predecir(modelo, normal, 672)


def test_tabla(modelo, normal) -> None:
    r = inf.predecir(modelo, normal, '2025-09-15 08:00')
    t = r.tabla
    assert list(t.columns) == COLUMNAS and len(t) == 1035
    assert r.instante == pd.Timestamp('2025-09-15 08:00')
    assert r.verificable and r.aviso is None
    assert (t['prediccion_veh_h'] >= 0).all()
    assert t['real_veh_h'].notna().equals(t['error'].notna())
    assert np.allclose(t['error'].dropna(),
                       (t['prediccion_veh_h'] - t['real_veh_h']).abs().dropna())
    assert t['id_sensor'].is_unique


def test_paso_entero_y_fecha_dan_lo_mismo(modelo, normal) -> None:
    j = int(normal.instantes.get_loc(pd.Timestamp('2025-09-15 08:00')))
    a = inf.predecir(modelo, normal, j).tabla
    b = inf.predecir(modelo, normal, '2025-09-15 08:00').tabla
    pd.testing.assert_frame_equal(a, b)


def test_igual_que_la_inferencia_por_lotes(modelo, normal) -> None:
    """Una ventana suelta da lo mismo que un lote de DatasetCompletoPlus,
    como en modelos/inferencia.py."""
    idx = [0, 32, 400, 671]
    x = torch.stack([normal.dataset[j][0] for j in idx])
    with torch.no_grad():
        lote = torch.expm1(modelo(x, normal.edge_index).clamp(min=0)).numpy()
    for k, j in enumerate(idx):
        sola = inf.predecir(modelo, normal, j).tabla['prediccion_veh_h'].to_numpy()
        np.testing.assert_allclose(sola, lote[k], rtol=1e-4, atol=1e-2)


def test_sin_datos_no_verificable(modelo, san_juan) -> None:
    r = inf.predecir(modelo, san_juan, '2025-06-29 20:00')
    assert not r.verificable
    assert r.aviso == inf.AVISO_SIN_DATOS
    assert r.tabla['real_veh_h'].isna().all() and r.tabla['error'].isna().all()
    assert r.tabla['prediccion_veh_h'].notna().all()


def test_persistencia_como_en_07(modelo, normal) -> None:
    """La persistencia de predecir() en toda la semana da el mismo MAE que
    persistencia_mae() de modelos/entrenamiento.py sobre esos pasos."""
    from modelos.entrenamiento import persistencia_mae
    ae, n = 0.0, 0
    for j in range(len(normal.instantes)):
        t = inf.predecir(modelo, normal, j).tabla
        v = t['real_veh_h'].notna()
        ae += (t['persistencia_veh_h'] - t['real_veh_h'])[v].abs().sum()
        n += int(v.sum())
    ds = normal.dataset
    inicio = ds.start_offset + inf.WINDOW - 1
    esperado, _ = persistencia_mae(ds.Y[inicio:], ds.M[inicio:])
    assert ae / n == pytest.approx(esperado, rel=1e-5)


def test_coinciden_tolera_diferencias_entre_cpus() -> None:
    """coinciden() acepta diferencias en el último dígito de float32 (como
    entre Windows y Linux) y rechaza diferencias reales."""
    a = np.array([55.99263, 0.0004, 8296.123, np.nan], dtype=np.float32)
    assert inf.coinciden(a, np.array([55.99261, 0.0009, 8296.13, np.nan], dtype=np.float32))
    assert not inf.coinciden(a, a + np.float32(0.5))
    assert not inf.coinciden(a, np.array([55.99263, 0.0004, 8296.123, 1.0], dtype=np.float32))
