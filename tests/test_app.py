"""Prueba de humo de app/app.py con streamlit.testing (sin navegador)."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / 'app' / 'app.py')


@pytest.fixture
def app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    return at


def test_arranca_sin_errores(app) -> None:
    assert not app.exception
    assert app.title[0].value == 'Predicción del tráfico en Valencia con redes neuronales de grafos'
    assert [t.label for t in app.tabs] == ['Mapa de predicción', 'Comparativa de modelos',
                                          'Sobre el proyecto']
    assert not app.warning


def test_aviso_sin_datos(app) -> None:
    app.sidebar.selectbox[0].select('Semana con festivo: San Juan, 23–29 jun 2025').run()
    app.sidebar.selectbox[1].select_index(6).run()   # domingo 29
    app.sidebar.selectbox[2].select_index(20).run()  # 20:15
    assert not app.exception
    assert app.warning[0].value == 'Sin datos de la fuente oficial: predicción no verificable'


def test_calculo_en_vivo_coincide(app) -> None:
    app.button[0].click().run()
    assert not app.exception
    assert 'coincide' in app.success[0].value
