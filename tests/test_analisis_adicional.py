"""Coherencia de pipeline_v2_recon/demo/analisis_adicional.json (versionado)."""
import json
from pathlib import Path

import pytest

DEMO = Path(__file__).resolve().parent.parent / 'pipeline_v2_recon' / 'demo'


@pytest.fixture(scope='module')
def analisis() -> dict:
    with open(DEMO / 'analisis_adicional.json', encoding='utf-8') as f:
        return json.load(f)


def test_mejora_coherente(analisis) -> None:
    for c in analisis['casos'].values():
        esperado = (c['mae_persistencia'] - c['mae_gat_plus']) / c['mae_persistencia'] * 100
        assert c['mejora_pct'] == pytest.approx(esperado, abs=0.02)


def test_caso_completo_igual_que_metricas(analisis) -> None:
    with open(DEMO / 'metricas.json', encoding='utf-8') as f:
        metricas = json.load(f)
    assert analisis['casos']['a_test_completo']['mae_gat_plus'] == metricas['modelos']['gat_plus']['mae']


def test_subconjuntos(analisis) -> None:
    a, b, c = (analisis['casos'][k] for k in
               ('a_test_completo', 'b_solo_15', 'c_solo_15_sin_congelacion'))
    assert a['instantes'] > b['instantes'] > c['instantes']
    assert b['instantes'] == pytest.approx(a['instantes'] / 4, abs=1)
