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


def boton(at: AppTest, etiqueta: str):
    return next(b for b in at.button if b.label == etiqueta)


def test_arranca_sin_errores(app) -> None:
    assert not app.exception
    assert app.title[0].value == 'Predicción del tráfico en Valencia con redes neuronales de grafos'
    assert [t.label for t in app.tabs] == ['Mapa de predicción', 'Comparativa de modelos',
                                          'Sobre el proyecto']
    assert not app.warning
    assert app.select_slider[0].value == 8
    assert [m.label for m in app.metric] == [
        'Sensores con datos', 'MAE de GAT+Plus en este instante',
        'MAE de la persistencia', 'Diferencia de GAT+Plus']


def test_barra_lateral_y_pie(app) -> None:
    lateral = ' '.join(m.value for m in app.sidebar.markdown)
    assert 'linkedin.com/in/alejandronavarrodelcastillo' in lateral and 'github.com' in lateral
    pie = next(c.value for c in app.caption if 'CC BY 4.0' in c.value)
    assert 'AEMET' in pie and 'OpenStreetMap' in pie and 'politica-de-uso' in pie


def test_aviso_sin_datos(app) -> None:
    app.sidebar.selectbox[0].select('Semana con festivo: San Juan, 23–29 jun 2025').run()
    app.sidebar.selectbox[1].select_index(6).run()   # domingo 29
    app.select_slider[0].set_value(20).run()         # 20:15
    assert not app.exception
    assert app.warning[0].value == 'Sin datos de la fuente oficial: predicción no verificable'
    assert not app.metric


def test_vista_por_sensores(app) -> None:
    assert app.selectbox[0].label.startswith('Tramo')
    app.toggle[0].set_value(True).run()
    assert not app.exception
    assert app.selectbox[0].label.startswith('Sensor')


def test_reproducir_y_pausa(app) -> None:
    boton(app, '▶ Reproducir').click().run()
    assert not app.exception
    assert app.session_state['reproduciendo'] and app.session_state['hora'] == 8
    app.run()  # equivale a una ejecución del fragmento con run_every
    assert app.session_state['hora'] == 9
    boton(app, '⏸ Pausa').click().run()
    assert not app.session_state['reproduciendo']


def test_calculo_en_vivo_coincide(app) -> None:
    boton(app, 'Calcular en vivo con el modelo').click().run()
    assert not app.exception
    assert 'coincide' in app.success[0].value


def test_reproducir_para_al_final_del_dia(app) -> None:
    app.select_slider[0].set_value(22).run()
    boton(app, '▶ Reproducir').click().run()
    app.run()
    assert not app.exception
    assert app.session_state['hora'] == 23 and not app.session_state['reproduciendo']
