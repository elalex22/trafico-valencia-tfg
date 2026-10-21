"""Coherencia de las cifras de metricas.json con verificacion.json y el README."""
import json
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parent.parent
DEMO = RAIZ / 'pipeline_v2_recon' / 'demo'


@pytest.fixture(scope='module')
def metricas() -> dict:
    with open(DEMO / 'metricas.json', encoding='utf-8') as f:
        return json.load(f)


@pytest.fixture(scope='module')
def sin_redondear() -> dict[str, float]:
    with open(DEMO / 'verificacion.json', encoding='utf-8') as f:
        return {m['modelo']: m['mejora_vs_persistencia_pct'] for m in json.load(f)['modelos']}


def test_mejora_a_un_decimal_sin_doble_redondeo(metricas, sin_redondear) -> None:
    """Redondear la mejora guardada a un decimal da lo mismo que redondear el
    valor sin redondear (con 2 decimales, GAT+Plus salía 25,1 en vez de 25,0)."""
    for modelo, valor in sin_redondear.items():
        guardado = metricas['modelos'][modelo]['mejora_vs_persistencia_pct']
        assert f'{guardado:.1f}' == f'{valor:.1f}', modelo


def test_cifra_de_gat_plus_en_nota_y_readme(metricas, sin_redondear) -> None:
    cifra = f'{sin_redondear["gat_plus"]:.1f}'.replace('.', ',') + ' %'
    assert cifra in metricas['nota']
    readme = (RAIZ / 'README.md').read_text(encoding='utf-8')
    assert f'| **GAT+Plus** | **{metricas["modelos"]["gat_plus"]["mae"]:.2f}'.replace('.', ',') in readme
    assert f'**{cifra}**' in readme
