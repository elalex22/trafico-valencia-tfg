# syntax=docker/dockerfile:1
# Imagen de la demo en Streamlit (app/app.py) con GAT+Plus en CPU.
# Construir: docker build -t trafico-valencia .
# Arrancar:  docker run -p 8501:8501 trafico-valencia

# ── Base: la misma versión de Python que el .venv del proyecto ───────────
FROM python:3.10-slim

# Sin .pyc, salida sin búfer (logs al momento) y sin caché de pip en la imagen.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /srv/trafico

# ── pip actualizado (la misma versión que el .venv) ──────────────────────
# El pip de la imagen base rechaza ruedas con nombres normalizados (p. ej.
# typing_extensions) e intenta compilarlas desde el código fuente.
RUN pip install pip==26.2.1

# ── torch solo CPU, torch-geometric y numpy (versiones del .venv) ────────
# requirements-torch-cpu.txt es el mismo archivo que usa la CI. Capa aparte:
# es la más pesada y casi nunca cambia.
COPY requirements-torch-cpu.txt .
RUN pip install -r requirements-torch-cpu.txt

# ── Resto de dependencias de la app ──────────────────────────────────────
# requirements-app.txt (pandas, streamlit, folium, streamlit-folium, plotly).
# Ni osmnx ni scipy: solo los usan los scripts del pipeline. pyarrow entra
# porque streamlit lo exige.
COPY requirements-app.txt .
RUN pip install -r requirements-app.txt

# ── Usuario sin privilegios ──────────────────────────────────────────────
RUN useradd --create-home --uid 1000 appuser

# ── Código y datos de la demo (solo lo que usa la app) ───────────────────
# app.py y inferencia_demo.py; de modelos/, solo lo que importa
# inferencia_demo.py; la muestra de la demo y únicamente los pesos de gat_plus.
COPY app/app.py app/inferencia_demo.py app/
COPY modelos/__init__.py modelos/arquitecturas.py modelos/datasets.py modelos/
COPY pipeline_v2_recon/demo/ pipeline_v2_recon/demo/
COPY pipeline_v2_recon/models/gat_plus.pt pipeline_v2_recon/models/

USER appuser

# ── Servidor: Streamlit sin navegador, accesible desde fuera del contenedor ──
EXPOSE 8501

# ── Healthcheck: la imagen slim no trae curl, se usa Python ──────────────
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=4).status == 200 else 1)"

CMD ["streamlit", "run", "app/app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]
