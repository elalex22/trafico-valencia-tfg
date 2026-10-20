"""
app.py — Demo en Streamlit de la predicción del tráfico en Valencia con
GAT+Plus.

Lee las predicciones precalculadas (app/precalcular_predicciones.py) y las
métricas de pipeline_v2_recon/demo/, así que arranca sin cargar torch; el
modelo solo se carga al pulsar "Calcular en vivo con el modelo".

Ejecutar desde la raíz del repositorio:
    streamlit run app/app.py
"""

import json
from pathlib import Path
from typing import Any

import folium
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from branca.colormap import LinearColormap
from streamlit_folium import st_folium

DEMO_DIR = Path(__file__).resolve().parent.parent / 'pipeline_v2_recon' / 'demo'
GITHUB = 'https://github.com/elalex22/trafico-valencia-tfg'
AUTOR = 'Alejandro Navarro del Castillo'
AVISO_SIN_DATOS = 'Sin datos de la fuente oficial: predicción no verificable'
GRIS = '#9e9e9e'
ESCALA = ['#1a9850', '#fee08b', '#d73027']  # verde → amarillo → rojo

SEMANAS = {
    'Semana normal: 15–21 sep 2025': 'semana_normal',
    'Semana con festivo: San Juan, 23–29 jun 2025': 'semana_san_juan',
}
DIAS = ['lunes', 'martes', 'miércoles', 'jueves', 'viernes', 'sábado', 'domingo']
MESES = ['enero', 'febrero', 'marzo', 'abril', 'mayo', 'junio', 'julio',
         'agosto', 'septiembre', 'octubre', 'noviembre', 'diciembre']
NOMBRES_MODELOS = {
    'mlp_simple': 'MLP Simple', 'mlp_completo': 'MLP Completo',
    'gcn_simple_v2': 'GCN Simple v2', 'gcn_completo_v2': 'GCN Completo v2',
    'gcn_completo_big': 'GCN Completo Big', 'gat_completo': 'GAT Completo',
    'gat_enriched': 'GAT Enriched', 'gat_plus': 'GAT+Plus', 'gat_ultra': 'GAT Ultra',
}
NOMBRES_ENSEMBLES = {
    'promedio_3gat': 'Promedio 3 GAT', 'oracle_dia': 'Oracle por día',
    'oracle_hora': 'Oracle por hora', 'oracle_hora_dia': 'Oracle hora × día',
}


# ── Carga de datos ───────────────────────────────────────────────────────

@st.cache_data
def cargar_metricas() -> dict[str, Any]:
    """Lee metricas.json."""
    with open(DEMO_DIR / 'metricas.json', encoding='utf-8') as f:
        return json.load(f)


@st.cache_data
def cargar_nodos() -> pd.DataFrame:
    """id_sensor, lat y lon de los 1.035 sensores, en el orden de los tensores."""
    nodos = pd.read_csv(DEMO_DIR / 'node_order.csv')
    return nodos.sort_values('idx')[['id_sensor', 'lat', 'lon']].reset_index(drop=True)


@st.cache_data
def cargar_predicciones(nombre: str) -> dict[str, np.ndarray]:
    """Lee predicciones_<semana>.npz (instantes :15, predicción, real, verificable)."""
    with np.load(DEMO_DIR / f'predicciones_{nombre}.npz') as npz:
        return {k: npz[k] for k in npz.files}


@st.cache_data
def maximo_escala() -> float:
    """Tope de la escala de color, común a las dos semanas: percentil 95
    de las predicciones de los sensores con datos."""
    valores = []
    for nombre in SEMANAS.values():
        p = cargar_predicciones(nombre)
        valores.append(p['prediccion'][~np.isnan(p['real'])])
    return float(np.round(np.percentile(np.concatenate(valores), 95), -2))


@st.cache_resource(show_spinner='Cargando el modelo (torch)…')
def modelo_en_vivo() -> Any:
    """Carga gat_plus con inferencia_demo (importa torch solo aquí)."""
    import inferencia_demo
    return inferencia_demo.cargar_modelo()


@st.cache_resource
def semana_en_vivo(nombre: str) -> Any:
    """Carga la semana de la muestra para predecir en vivo."""
    import inferencia_demo
    return inferencia_demo.cargar_semana(nombre)


def fecha_larga(t: pd.Timestamp) -> str:
    """'lunes 15 de septiembre'."""
    return f'{DIAS[t.dayofweek]} {t.day} de {MESES[t.month - 1]}'


def formato(x: float, decimales: int = 0) -> str:
    """Número con formato español (separador de miles '.', decimal ',')."""
    s = f'{x:,.{decimales}f}'
    return s.replace(',', '_').replace('.', ',').replace('_', '.')


# ── Barra lateral ────────────────────────────────────────────────────────

def barra_lateral() -> tuple[str, int]:
    """Selectores de semana, día y hora. Devuelve el nombre de la semana y
    la posición del instante en sus predicciones."""
    with st.sidebar:
        st.header('Instante')
        etiqueta = st.selectbox('Semana', list(SEMANAS))
        nombre = SEMANAS[etiqueta]
        instantes = pd.DatetimeIndex(cargar_predicciones(nombre)['instantes'])

        dias = sorted(set(instantes.normalize()))
        dia = st.selectbox('Día', dias, format_func=fecha_larga)
        del_dia = np.flatnonzero(instantes.normalize() == dia)
        j = st.selectbox('Hora', del_dia, index=8,
                         format_func=lambda k: instantes[k].strftime('%H:%M'))
        st.caption('La intensidad de las espiras es horaria y el valor nuevo '
                   'llega a y cuarto: en los demás minutos el valor real es el '
                   'último que vio el modelo. Por eso la demo solo usa los '
                   'instantes :15.')

        st.divider()
        st.markdown(f'**{AUTOR}**  \nTFG del Grado en Ciencia de Datos (UPV)')
        st.markdown(f'[Código en GitHub]({GITHUB})')
    return nombre, int(j)


# ── Pestaña: mapa de predicción ──────────────────────────────────────────

def construir_mapa(nodos: pd.DataFrame, pred: np.ndarray, real: np.ndarray,
                   vmax: float) -> folium.Map:
    """Mapa con un círculo por sensor, de verde a rojo según la intensidad
    prevista y en gris si el sensor no tiene datos en ese instante."""
    mapa = folium.Map(location=[nodos.lat.mean(), nodos.lon.mean()], zoom_start=13,
                      tiles='OpenStreetMap')
    escala = LinearColormap(ESCALA, vmin=0, vmax=vmax,
                            caption='Intensidad prevista (veh/h)')
    for i, fila in nodos.iterrows():
        con_dato = not np.isnan(real[i])
        texto = (f'Sensor {fila.id_sensor}<br>Predicción: {formato(pred[i])} veh/h<br>'
                 + (f'Real: {formato(real[i])} veh/h' if con_dato else 'Sin datos del sensor'))
        color = escala(min(pred[i], vmax)) if con_dato else GRIS
        folium.CircleMarker(
            location=[fila.lat, fila.lon], radius=5 if con_dato else 3,
            color=color, fill=True, fill_color=color, fill_opacity=0.85, weight=1,
            tooltip=texto,
        ).add_to(mapa)
    escala.add_to(mapa)
    leyenda_gris = (
        '<div style="position:fixed;bottom:18px;left:12px;z-index:9999;background:white;'
        'padding:4px 8px;border-radius:4px;font-size:12px;box-shadow:0 0 3px #888">'
        f'<span style="color:{GRIS};font-size:16px">●</span> sin datos del sensor</div>')
    mapa.get_root().html.add_child(folium.Element(leyenda_gris))
    return mapa


def sensor_mas_cercano(nodos: pd.DataFrame, lat: float, lon: float) -> int:
    """id_sensor del nodo más cercano a un clic en el mapa."""
    d = (nodos.lat - lat) ** 2 + ((nodos.lon - lon) * np.cos(np.radians(lat))) ** 2
    return int(nodos.id_sensor[d.idxmin()])


def grafico_dia(p: dict[str, np.ndarray], i: int, instante: pd.Timestamp,
                id_sensor: int) -> go.Figure:
    """Serie real y predicha (instantes :15) del sensor `i` en el día de `instante`."""
    instantes = pd.DatetimeIndex(p['instantes'])
    dia = np.flatnonzero(instantes.normalize() == instante.normalize())
    x = instantes[dia]
    fig = go.Figure()
    fig.add_scatter(x=x, y=p['real'][dia, i], name='Real', mode='lines+markers',
                    line=dict(color='#333333'))
    fig.add_scatter(x=x, y=p['prediccion'][dia, i], name='Predicción GAT+Plus',
                    mode='lines+markers', line=dict(color='#d73027', dash='dot'))
    fig.add_vline(x=instante, line=dict(color=GRIS, dash='dash'))
    fig.update_layout(title=f'Sensor {id_sensor} — {fecha_larga(instante)}',
                      yaxis_title='veh/h', xaxis_tickformat='%H:%M', height=380,
                      margin=dict(l=10, r=10, t=50, b=10),
                      legend=dict(orientation='h', y=-0.15))
    return fig


def pestana_mapa(nombre: str, j: int) -> None:
    """Mapa del instante elegido, serie del sensor y cálculo en vivo."""
    nodos = cargar_nodos()
    p = cargar_predicciones(nombre)
    instante = pd.Timestamp(p['instantes'][j])
    pred, real = p['prediccion'][j], p['real'][j]
    con_dato = ~np.isnan(real)

    st.subheader(f'{fecha_larga(instante).capitalize()} de {instante.year}, '
                 f'{instante:%H:%M}')
    if not p['verificable'][j]:
        st.warning(AVISO_SIN_DATOS)
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric('Sensores con datos', f'{int(con_dato.sum())} de {len(nodos)}')
        c2.metric('MAE de GAT+Plus en este instante',
                  f'{formato(np.abs(pred - real)[con_dato].mean(), 1)} veh/h')
        c3.metric('Intensidad media real', f'{formato(real[con_dato].mean())} veh/h')

    col_mapa, col_serie = st.columns([3, 2])
    with col_mapa:
        salida = st_folium(construir_mapa(nodos, pred, real, maximo_escala()),
                           height=560, use_container_width=True,
                           returned_objects=['last_object_clicked'], key='mapa')
    clic = (salida or {}).get('last_object_clicked')
    if clic and clic != st.session_state.get('ultimo_clic'):
        st.session_state['ultimo_clic'] = clic
        st.session_state['sensor'] = sensor_mas_cercano(nodos, clic['lat'], clic['lng'])

    with col_serie:
        ids = nodos.id_sensor.tolist()
        if st.session_state.get('sensor') not in ids:
            st.session_state['sensor'] = ids[int(np.argmax(np.where(con_dato, pred, -1)))]
        id_sensor = st.selectbox('Sensor (o haz clic en el mapa)', ids, key='sensor')
        i = ids.index(id_sensor)
        st.plotly_chart(grafico_dia(p, i, instante, id_sensor), width='stretch')

    st.divider()
    if st.button('Calcular en vivo con el modelo'):
        import inferencia_demo
        r = inferencia_demo.predecir(modelo_en_vivo(), semana_en_vivo(nombre), instante)
        dif = np.abs(r.tabla['prediccion_veh_h'].to_numpy() - pred).max()
        if dif < 0.01:
            st.success(f'El modelo calculado en vivo coincide con la predicción '
                       f'precalculada (diferencia máxima {dif:.2e} veh/h en '
                       f'{len(nodos)} sensores).')
        else:
            st.error(f'La predicción en vivo difiere de la precalculada '
                     f'(diferencia máxima {formato(dif, 2)} veh/h).')
        st.dataframe(r.tabla.head(10), hide_index=True)


# ── Pestaña: comparativa de modelos ──────────────────────────────────────

def pestana_comparativa() -> None:
    """Tabla y gráfico de los 9 modelos, ensembles y notas sobre las cifras."""
    met = cargar_metricas()
    base = met['baseline_persistencia']
    tabla = pd.DataFrame([
        {'Modelo': NOMBRES_MODELOS.get(k, k), 'MAE (veh/h)': v['mae'],
         'RMSE (veh/h)': v['rmse'], 'Mejora vs persistencia (%)': v['mejora_vs_persistencia_pct']}
        for k, v in met['modelos'].items()
    ]).sort_values('MAE (veh/h)').reset_index(drop=True)
    destacado = NOMBRES_MODELOS[met['modelo_demo']]

    st.subheader('Los 9 modelos en test')
    st.caption('MAE y RMSE en veh/h sobre el test (1 de junio – 23 de septiembre de 2025). '
               f'Persistencia (Ŷ[t+1] = Y[t]): MAE {formato(base["mae"], 2)}, '
               f'RMSE {formato(base["rmse"], 2)}.')
    col_tabla, col_barras = st.columns([2, 3])
    with col_tabla:
        st.dataframe(
            tabla.style.apply(lambda f: ['font-weight: bold; background-color: #fde0dc'
                                         if f['Modelo'] == destacado else '' for _ in f], axis=1)
                 .format({'MAE (veh/h)': '{:.2f}', 'RMSE (veh/h)': '{:.2f}',
                          'Mejora vs persistencia (%)': '{:.1f}'}),
            hide_index=True, width='stretch')
    with col_barras:
        orden = tabla.iloc[::-1]
        fig = go.Figure(go.Bar(
            x=orden['MAE (veh/h)'], y=orden['Modelo'], orientation='h',
            marker_color=['#d73027' if m == destacado else '#9ecae1' for m in orden['Modelo']],
            text=[f'{v:.2f}' for v in orden['MAE (veh/h)']], textposition='outside'))
        fig.add_vline(x=base['mae'], line=dict(color='#333333', dash='dash'),
                      annotation_text=f'Persistencia {base["mae"]:.2f}',
                      annotation_position='top')
        fig.update_layout(xaxis_title='MAE en test (veh/h)', height=420,
                          xaxis_range=[0, base['mae'] * 1.12],
                          margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, width='stretch')

    st.subheader('Ensembles')
    ens = pd.DataFrame([
        {'Ensemble': NOMBRES_ENSEMBLES.get(k, k), 'MAE (veh/h)': v['mae'],
         'Mejora vs persistencia (%)': v['mejora_vs_persistencia_pct'],
         'Descripción': v['descripcion'],
         'Tipo': ('Cota optimista: elige con las etiquetas de test'
                  if v['eleccion_con_etiquetas_de_test'] else 'Desplegable')}
        for k, v in met['ensembles'].items()
    ])
    st.dataframe(ens.style.format({'MAE (veh/h)': '{:.2f}',
                                   'Mejora vs persistencia (%)': '{:.1f}'}),
                 hide_index=True, width='stretch')

    with st.expander('Sobre las cifras'):
        mem = met['memoria_tfg']
        st.markdown(f'**Reconstrucción frente a memoria.** {met["nota"]}')
        st.markdown(
            f'| | Memoria del TFG | Reconstrucción |\n|---|---|---|\n'
            f'| Persistencia (MAE) | {formato(mem["persistencia_mae"], 2)} | {formato(base["mae"], 2)} |\n'
            f'| GAT+Plus (MAE) | {formato(mem["gat_plus_mae"], 2)} | '
            f'{formato(met["modelos"]["gat_plus"]["mae"], 2)} |\n'
            f'| Mejora | {formato(mem["mejora_gat_plus_pct"], 1)} % | '
            f'{formato(met["modelos"]["gat_plus"]["mejora_vs_persistencia_pct"], 1)} % |')
        st.caption(mem['fuente'])
        st.markdown(f'**Calidad de los datos.** {met["nota_datos"]}')
        st.markdown(f'**Ensembles.** {met["nota_ensembles"]}')


# ── Pestaña: sobre el proyecto ───────────────────────────────────────────

PIPELINE = [
    ('01_extraccion', 'extrae las 6 fuentes brutas (ZIP, JSON y CSV) a Parquet.'),
    ('02_construccion_grafo', 'descarga el viario de OpenStreetMap y une las 1.035 espiras en un grafo.'),
    ('03_asignaciones_espaciales', 'asigna a cada espira la bici, Valenbisi y ruido más cercanos.'),
    ('04_fusion_master', 'fusiona las fuentes en una línea temporal de 15 min con festivos y lags.'),
    ('05_pipeline_tensores', 'crea los tensores (tiempo × nodo × feature), limpia el objetivo y normaliza.'),
    ('06_grafo_enriquecido', 'añade aristas entre sensores con tráfico muy correlacionado.'),
    ('07_baseline_persistencia', 'calcula la referencia trivial: el siguiente valor es igual al actual.'),
    ('08_entrenar_modelos', 'entrena las 9 arquitecturas (MLP, GCN y GAT).'),
    ('09_analisis_temporal', 'compara GAT+Plus con la persistencia por franja, hora y día.'),
    ('10_ensembles', 'evalúa 4 combinaciones de modelos.'),
]


def pestana_proyecto() -> None:
    """Problema, pipeline, datos y hallazgos de calidad."""
    st.subheader('El problema')
    st.markdown(
        'Predecir la intensidad de tráfico (vehículos/hora) de las 1.035 espiras de '
        'València con 15 minutos de antelación.  \n'
        'Las espiras forman un grafo según el viario, y una red neuronal de grafos '
        '(GAT) aprovecha lo que pasa en los sensores vecinos.  \n'
        'La referencia que hay que batir es la persistencia: suponer que el próximo '
        'valor será igual al actual.')

    st.subheader('El pipeline: 10 scripts')
    st.markdown('\n'.join(f'{k + 1}. `{s}.py`: {d}' for k, (s, d) in enumerate(PIPELINE)))

    st.subheader('Datos')
    st.markdown(
        '- Espiras de tráfico y estado de los tramos (Ayuntamiento de València).\n'
        '- Espiras de bicicleta y estaciones de Valenbisi.\n'
        '- Meteorología diaria de AEMET.\n'
        '- Ruido ambiental: solo 15 sensores, todos en Ruzafa.\n'
        '- Viario de OpenStreetMap.\n\n'
        f'El código se publica con licencia MIT en [GitHub]({GITHUB}).')

    st.subheader('Hallazgos de calidad de los datos')
    st.markdown(
        '- **8 meses corruptos:** de octubre de 2024 a mayo de 2025 la fuente oficial '
        'de espiras dio casi todo ceros (noviembre y diciembre, el 100 %). El portal lo '
        'reconocía y esos meses quedan fuera de los splits.\n'
        '- **Junio congelado:** del 1 al 12 de junio de 2025 todos los sensores repiten '
        'el mismo valor (11,3 días, alrededor del 10 % del test). Las métricas lo incluyen.\n'
        '- **Apagón del 29 de junio:** desde las 9:00 no hay ningún dato de tráfico; '
        'la demo marca esas horas como no verificables.\n'
        '- **Intensidad horaria:** el valor cambia una vez por hora, a y cuarto, así '
        'que en 3 de cada 4 pasos de 15 minutos el valor real es el anterior.')


# ── Página ───────────────────────────────────────────────────────────────

def main() -> None:
    """Monta la página."""
    st.set_page_config(page_title='Tráfico en Valencia con GNN', page_icon='🚦',
                       layout='wide')
    met = cargar_metricas()
    mem = met['memoria_tfg']
    mejora = met['modelos'][met['modelo_demo']]['mejora_vs_persistencia_pct']
    st.title('Predicción del tráfico en Valencia con redes neuronales de grafos')
    st.markdown(
        f'Modelo GAT+Plus entrenado con datos abiertos de València; mejora un '
        f'{formato(mejora, 1)} % a la persistencia en la reconstrucción reproducible '
        f'({formato(mem["mejora_gat_plus_pct"], 1)} % en la memoria del TFG)')

    nombre, j = barra_lateral()
    mapa, comparativa, proyecto = st.tabs(
        ['Mapa de predicción', 'Comparativa de modelos', 'Sobre el proyecto'])
    with mapa:
        pestana_mapa(nombre, j)
    with comparativa:
        pestana_comparativa()
    with proyecto:
        pestana_proyecto()


main()
