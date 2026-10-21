"""Coherencia de los archivos de dependencias de la demo (Docker, CI y
Streamlit Community Cloud)."""
import re
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent


def lineas(ruta: Path) -> list[str]:
    """Líneas útiles (sin comentarios ni vacías) de un requirements."""
    return [x.strip() for x in ruta.read_text(encoding='utf-8').splitlines()
            if x.strip() and not x.strip().startswith('#')]


def paquetes(ruta: Path) -> dict[str, str]:
    """Paquetes fijados con == en un requirements, siguiendo los -r."""
    fijados = {}
    for x in lineas(ruta):
        if x.startswith('-r '):
            fijados |= paquetes((ruta.parent / x[3:]).resolve())
        elif '==' in x:
            nombre, version = x.split('==')
            fijados[nombre.lower()] = version
    return fijados


def test_app_requirements_remite_a_los_compartidos() -> None:
    """Community Cloud usa app/requirements.txt, que solo remite a los
    archivos compartidos con Docker y la CI."""
    assert lineas(RAIZ / 'app' / 'requirements.txt') == [
        '-r ../requirements-torch-cpu.txt', '-r ../requirements-app.txt']


def test_dependencias_de_la_demo() -> None:
    fijados = paquetes(RAIZ / 'app' / 'requirements.txt')
    assert fijados['torch'].endswith('+cpu')
    assert {'torch-geometric', 'numpy', 'streamlit', 'starlette'} <= set(fijados)
    assert not {'osmnx', 'scipy', 'pyarrow'} & set(fijados)


def test_torch_del_indice_de_cpu() -> None:
    torch_cpu = lineas(RAIZ / 'requirements-torch-cpu.txt')
    assert '--extra-index-url https://download.pytorch.org/whl/cpu' in torch_cpu


def test_dockerfile_usa_los_mismos_archivos() -> None:
    dockerfile = (RAIZ / 'Dockerfile').read_text(encoding='utf-8')
    for archivo in ('requirements-torch-cpu.txt', 'requirements-app.txt'):
        assert re.search(rf'pip install -r {re.escape(archivo)}', dockerfile), archivo
