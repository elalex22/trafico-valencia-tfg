"""Tests de app/escribir_metricas.py: metricas.json es reproducible a partir
de los JSON versionados."""
import json

from app import escribir_metricas as em


def entradas() -> tuple[dict, dict, dict]:
    return (em.leer_json(em.DEMO_DIR / 'verificacion.json'),
            em.leer_json(em.RESULTADOS_DIR / 'ensembles.json'),
            em.leer_json(em.RESULTADOS_DIR / 'baseline.json'))


def test_reproduce_el_metricas_versionado() -> None:
    """construir_metricas() sobre las entradas versionadas da exactamente el
    metricas.json del repositorio (y el mismo texto al serializarlo)."""
    nuevo = em.construir_metricas(*entradas())
    assert nuevo == em.leer_json(em.SALIDA)
    texto = json.dumps(nuevo, indent=2, ensure_ascii=False)
    assert texto == em.SALIDA.read_text(encoding='utf-8')


def test_un_solo_redondeo_en_la_nota() -> None:
    """Con una mejora de 25,0491 la nota dice 25,0 % (con doble redondeo
    salía 25,1 %), y el MAE conserva el cero final (28,70, no 28,7)."""
    verificacion, ensembles, baseline = entradas()
    plus = next(f for f in verificacion['modelos'] if f['modelo'] == 'gat_plus')
    plus['mejora_vs_persistencia_pct'], plus['mae_actual'] = 25.0491, 28.7
    nota = em.construir_metricas(verificacion, ensembles, baseline)['nota']
    assert 'en la reconstrucción, 28,70 veh/h y un 25,0 %' in nota


def test_coma() -> None:
    assert em.coma(28.7098, 2) == '28,71'
    assert em.coma(25.0491, 1) == '25,0'
    assert em.coma(38.3047, 2) == '38,30'
