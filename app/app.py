"""
app.py — Demo en Streamlit de la predicción del tráfico en Valencia con
GAT+Plus.

Lee las predicciones precalculadas (app/precalcular_predicciones.py), la
geometría (app/exportar_geometria.py) y las métricas de
pipeline_v2_recon/demo/, así que arranca sin cargar torch; el modelo solo
se carga al pulsar "Calcular en vivo con el modelo".

El mapa se dibuja con streamlit-folium: el mapa base se crea una vez y en
cada cambio de hora solo se sustituye la capa de tráfico
(feature_group_to_add), dentro de un st.fragment que, al reproducir, se
vuelve a ejecutar cada PASO_REPRODUCCION segundos.

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
LINKEDIN = 'https://www.linkedin.com/in/alejandronavarrodelcastillo/'
AUTOR = 'Alejandro Navarro del Castillo'
LICENCIA_DATOS = ('https://www.valencia.es/web/gobierno-abierto/cas/datos-abiertos/'
                  'reutilizacion/politica-de-uso-datos-abiertos/politica-de-uso')
FUENTE_DATOS = (f'Fuente de los datos: [Ajuntament de València – Dades Obertes]'
                f'({LICENCIA_DATOS}) (licencia CC BY 4.0)')
AVISO_SIN_DATOS = 'Sin datos de la fuente oficial: predicción no verificable'
GRIS = '#9e9e9e'
ESCALA = ['#1a9850', '#fee08b', '#d73027']  # verde → amarillo → rojo
PASO_REPRODUCCION = 1.5  # segundos entre horas al reproducir
HORA_INICIAL = 8

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
    """Lee predicciones_<semana>.npz (instantes :15, predicción, real,
    persistencia y verificable)."""
    with np.load(DEMO_DIR / f'predicciones_{nombre}.npz') as npz:
        return {k: npz[k] for k in npz.files}


@st.cache_data
def cargar_geometria() -> dict[str, list[dict[str, Any]]]:
    """Lee geometria.geojson y separa las capas 'tramo' y 'sensor'."""
    with open(DEMO_DIR / 'geometria.geojson', encoding='utf-8') as f:
        features = json.load(f)['features']
    return {capa: [x for x in features if x['properties']['capa'] == capa]
            for capa in ('tramo', 'sensor')}


@st.cache_data
def posiciones_por_tramo() -> dict[int, np.ndarray]:
    """Para cada tramo, las posiciones (orden de los tensores) de sus sensores."""
    posicion = {s: i for i, s in enumerate(cargar_nodos().id_sensor)}
    return {f['properties']['id_tramo']: np.array([posicion[s] for s in f['properties']['sensores']])
            for f in cargar_geometria()['tramo']}


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


def media_con_datos(valores: np.ndarray, real: np.ndarray) -> float:
    """Media de `valores` sobre los sensores con dato real; NaN si no hay."""
    con_dato = ~np.isnan(real)
    return float(valores[con_dato].mean()) if con_dato.any() else float('nan')


# ── Barra lateral y pie ──────────────────────────────────────────────────

def barra_lateral() -> tuple[str, pd.Timestamp]:
    """Selectores de semana y día. Devuelve el nombre de la semana y el día."""
    with st.sidebar:
        st.header('Instante')
        etiqueta = st.selectbox('Semana', list(SEMANAS))
        nombre = SEMANAS[etiqueta]
        instantes = pd.DatetimeIndex(cargar_predicciones(nombre)['instantes'])
        dia = st.selectbox('Día', sorted(set(instantes.normalize())), format_func=fecha_larga)
        st.caption('La hora se elige con el control deslizante encima del mapa. '
                   'La intensidad de las espiras es horaria y el valor nuevo '
                   'llega a y cuarto: en los demás minutos el valor real es el '
                   'último que vio el modelo. Por eso la demo solo usa los '
                   'instantes :15.')

        st.divider()
        st.markdown(f'**{AUTOR}**  \nTFG del Grado en Ciencia de Datos (UPV)')
        st.markdown(f'[Código en GitHub]({GITHUB}) · [LinkedIn]({LINKEDIN})')
    return nombre, pd.Timestamp(dia)


def pie() -> None:
    """Pie de página con las fuentes y licencias."""
    st.divider()
    st.caption(f'{FUENTE_DATOS}. Meteorología: AEMET. Red viaria y mapa base: '
               '© colaboradores de OpenStreetMap. Código: licencia MIT.')


# ── Pestaña: mapa de predicción ──────────────────────────────────────────

def mapa_base(vmax: float) -> folium.Map:
    """Mapa base (teselas de OpenStreetMap, sin clave) con las leyendas.

    Se crea de nuevo en cada ejecución y no se cachea: st_folium añade la
    capa de feature_group_to_add al mapa que recibe, y con un objeto
    cacheado la capa anterior se acumula y el componente queda con 0 px de
    alto desde la segunda ejecución. No hace falta cachearlo para evitar el
    parpadeo: st_folium calcula su clave con los identificadores de folium
    normalizados, así que un mapa idéntico no se vuelve a dibujar."""
    nodos = cargar_nodos()
    mapa = folium.Map(location=[nodos.lat.mean(), nodos.lon.mean()], zoom_start=13,
                      tiles='OpenStreetMap')
    LinearColormap(ESCALA, vmin=0, vmax=vmax,
                   caption='Intensidad prevista (veh/h)').add_to(mapa)
    leyenda_gris = (
        '<div style="position:fixed;bottom:18px;left:12px;z-index:9999;background:white;'
        'padding:4px 8px;border-radius:4px;font-size:12px;box-shadow:0 0 3px #888">'
        f'<span style="display:inline-block;width:18px;height:4px;background:{GRIS};'
        'vertical-align:middle"></span> sin datos del sensor</div>')
    mapa.get_root().html.add_child(folium.Element(leyenda_gris))
    return mapa


def capa_trafico(por_tramos: bool, pred: np.ndarray, real: np.ndarray,
                 vmax: float) -> folium.FeatureGroup:
    """Capa de líneas del instante: tramos oficiales con la media de sus
    sensores con datos (vista B) o cada sensor como su arista de OSM
    (vista A). En gris, lo que no tiene datos."""
    escala = LinearColormap(ESCALA, vmin=0, vmax=vmax)
    geometria = cargar_geometria()
    features = []
    if por_tramos:
        posiciones = posiciones_por_tramo()
        for f in geometria['tramo']:
            p = f['properties']
            idx = posiciones[p['id_tramo']]
            con_dato = ~np.isnan(real[idx])
            if con_dato.any():
                v_pred, v_real = pred[idx][con_dato].mean(), real[idx][con_dato].mean()
                info = (f'<b>{p["nombre"]}</b><br>Tramo {p["id_tramo"]} · '
                        f'{int(con_dato.sum())} de {len(idx)} sensores con datos<br>'
                        f'Predicción media: {formato(v_pred)} veh/h<br>'
                        f'Real media: {formato(v_real)} veh/h')
                color = escala(min(v_pred, vmax))
            else:
                info = (f'<b>{p["nombre"]}</b><br>Tramo {p["id_tramo"]} · '
                        f'sin datos del sensor')
                color = GRIS
            features.append({**f, 'properties': {**p, 'info': info, 'color': color}})
    else:
        posicion = {s: i for i, s in enumerate(cargar_nodos().id_sensor)}
        for f in geometria['sensor']:
            p = f['properties']
            i = posicion[p['id_sensor']]
            if np.isnan(real[i]):
                info, color = f'Sensor {p["id_sensor"]}<br>sin datos del sensor', GRIS
            else:
                info = (f'Sensor {p["id_sensor"]}<br>Predicción: {formato(pred[i])} veh/h'
                        f'<br>Real: {formato(real[i])} veh/h')
                color = escala(min(pred[i], vmax))
            features.append({**f, 'properties': {**p, 'info': info, 'color': color}})

    features.sort(key=lambda f: f['properties']['color'] != GRIS)  # gris debajo
    capa = folium.FeatureGroup(name='Tráfico')
    folium.GeoJson(
        {'type': 'FeatureCollection', 'features': features},
        style_function=lambda f: {'color': f['properties']['color'], 'opacity': 0.9,
                                  'weight': 3 if f['properties']['color'] == GRIS else 5},
        highlight_function=lambda f: {'weight': 9},
        tooltip=folium.GeoJsonTooltip(fields=['info'], labels=False),
    ).add_to(capa)
    return capa


def serie_dia(p: dict[str, np.ndarray], dia: pd.Timestamp,
              idx: np.ndarray) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    """Serie real y predicha del día para los sensores `idx` (media de los
    que tienen datos en cada instante)."""
    instantes = pd.DatetimeIndex(p['instantes'])
    filas = np.flatnonzero(instantes.normalize() == dia)
    real = np.array([media_con_datos(p['real'][k, idx], p['real'][k, idx]) for k in filas])
    pred = np.array([media_con_datos(p['prediccion'][k, idx], p['real'][k, idx]) for k in filas])
    return instantes[filas], real, pred


def grafico_dia(x: pd.DatetimeIndex, real: np.ndarray, pred: np.ndarray,
                instante: pd.Timestamp, titulo: str) -> go.Figure:
    """Gráfico de la serie real y la predicha, con el instante marcado."""
    fig = go.Figure()
    fig.add_scatter(x=x, y=real, name='Real', mode='lines+markers',
                    line=dict(color='#333333'))
    fig.add_scatter(x=x, y=pred, name='Predicción GAT+Plus',
                    mode='lines+markers', line=dict(color='#d73027', dash='dot'))
    fig.add_vline(x=instante, line=dict(color=GRIS, dash='dash'))
    fig.update_layout(title=titulo, yaxis_title='veh/h', xaxis_tickformat='%H:%M',
                      height=380, margin=dict(l=10, r=10, t=50, b=10),
                      legend=dict(orientation='h', y=-0.15))
    return fig


def metricas_instante(pred: np.ndarray, real: np.ndarray, persistencia: np.ndarray,
                      n_nodos: int) -> None:
    """MAE de GAT+Plus y de la persistencia en el instante, y su diferencia."""
    con_dato = ~np.isnan(real)
    mae_gat = np.abs(pred - real)[con_dato].mean()
    mae_pers = np.abs(persistencia - real)[con_dato].mean()
    global_gat = cargar_metricas()['modelos']['gat_plus']['mae']
    ayuda = (f'Este MAE es de un solo instante y siempre a y cuarto, cuando llega el '
             f'valor horario nuevo, y en hora punta los errores en veh/h son mayores. '
             f'El global ({formato(global_gat, 2)}) promedia todo el test, incluidas '
             f'las noches y los minutos :00, :30 y :45, en los que el valor real es el '
             f'anterior y el error es casi cero.')
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Sensores con datos', f'{int(con_dato.sum())} de {n_nodos}')
    c2.metric('MAE de GAT+Plus en este instante', f'{formato(mae_gat, 1)} veh/h', help=ayuda)
    c3.metric('MAE de la persistencia', f'{formato(mae_pers, 1)} veh/h',
              help='Persistencia: suponer que el valor será igual al último observado. '
                   + ayuda)
    c4.metric('Diferencia de GAT+Plus', f'{formato((mae_gat - mae_pers) / mae_pers * 100, 1)} %',
              help='(MAE de GAT+Plus − MAE de la persistencia) / MAE de la persistencia. '
                   'Negativo: GAT+Plus se equivoca menos.')


def controles_hora() -> int:
    """Control deslizante de hora (00:15–23:15) y botón de reproducción.
    Al reproducir, avanza una hora en cada ejecución del fragmento."""
    if st.session_state.get('reproduciendo'):
        if st.session_state.get('avanzar'):
            st.session_state['hora'] = st.session_state['hora'] + 1
        st.session_state['avanzar'] = True
        if st.session_state['hora'] >= 23:
            st.session_state['reproduciendo'] = False
            st.session_state['parar'] = True

    col_slider, col_boton = st.columns([5, 1])
    with col_slider:
        hora = st.select_slider('Hora', options=list(range(24)), key='hora',
                                format_func=lambda h: f'{h:02d}:15')
    with col_boton:
        st.write('')
        if st.session_state.get('reproduciendo'):
            if st.button('⏸ Pausa', width='stretch'):
                st.session_state['reproduciendo'] = False
                st.rerun(scope='app')
        elif st.button('▶ Reproducir', width='stretch'):
            if st.session_state['hora'] >= 23:
                st.session_state['hora'] = 0
            st.session_state['reproduciendo'] = True
            st.session_state['avanzar'] = False
            st.rerun(scope='app')
    return int(hora)


def vista_mapa(nombre: str, dia: pd.Timestamp) -> None:
    """Contenido de la pestaña del mapa (se ejecuta como fragmento)."""
    nodos = cargar_nodos()
    p = cargar_predicciones(nombre)
    instantes = pd.DatetimeIndex(p['instantes'])
    hora = controles_hora()
    k = int(np.flatnonzero(instantes == dia + pd.Timedelta(hours=hora, minutes=15))[0])
    instante = instantes[k]
    pred, real = p['prediccion'][k], p['real'][k]

    st.subheader(f'{fecha_larga(instante).capitalize()} de {instante.year}, '
                 f'{instante:%H:%M}')
    if not p['verificable'][k]:
        st.warning(AVISO_SIN_DATOS)
    else:
        metricas_instante(pred, real, p['persistencia'][k], len(nodos))

    por_tramos = not st.toggle('Ver cada sensor (arista de OSM a ±75 m) en lugar de los '
                               'tramos oficiales', key='vista_sensores')
    vmax = maximo_escala()
    col_mapa, col_serie = st.columns([3, 2])
    with col_mapa:
        salida = st_folium(mapa_base(vmax), key='mapa', height=560,
                           use_container_width=True,
                           feature_group_to_add=capa_trafico(por_tramos, pred, real, vmax),
                           returned_objects=['last_active_drawing'])
    clic = (salida or {}).get('last_active_drawing')
    if clic and clic != st.session_state.get('ultimo_clic'):
        st.session_state['ultimo_clic'] = clic
        props = clic.get('properties', {})
        if 'id_tramo' in props:
            st.session_state['tramo_sel'] = int(props['id_tramo'])
        if 'id_sensor' in props:
            st.session_state['sensor_sel'] = int(props['id_sensor'])

    with col_serie:
        if por_tramos:
            tramos = {f['properties']['id_tramo']: f['properties']['nombre']
                      for f in cargar_geometria()['tramo']}
            posiciones = posiciones_por_tramo()
            if st.session_state.get('tramo_sel') not in tramos:
                medias = {t: media_con_datos(pred[i], real[i]) for t, i in posiciones.items()}
                st.session_state['tramo_sel'] = max(
                    tramos, key=lambda t: -1 if np.isnan(medias[t]) else medias[t])
            id_tramo = st.selectbox('Tramo (o haz clic en el mapa)', list(tramos),
                                    key='tramo_sel',
                                    format_func=lambda t: f'{t} · {tramos[t]}')
            idx = posiciones[id_tramo]
            titulo = f'{tramos[id_tramo]} ({len(idx)} sensores, media)'
        else:
            ids = nodos.id_sensor.tolist()
            if st.session_state.get('sensor_sel') not in ids:
                st.session_state['sensor_sel'] = ids[int(np.argmax(np.where(np.isnan(real), -1, pred)))]
            id_sensor = st.selectbox('Sensor (o haz clic en el mapa)', ids, key='sensor_sel')
            idx = np.array([ids.index(id_sensor)])
            titulo = f'Sensor {id_sensor}'
        x, serie_real, serie_pred = serie_dia(p, dia, idx)
        st.plotly_chart(grafico_dia(x, serie_real, serie_pred, instante,
                                    f'{titulo} — {fecha_larga(instante)}'),
                        width='stretch')

    if not st.session_state.get('reproduciendo'):
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

    if st.session_state.pop('parar', False):
        st.rerun(scope='app')


def pestana_mapa(nombre: str, dia: pd.Timestamp) -> None:
    """Pestaña del mapa: un fragmento que se repite al reproducir."""
    st.session_state.setdefault('hora', HORA_INICIAL)
    intervalo = PASO_REPRODUCCION if st.session_state.get('reproduciendo') else None
    st.fragment(run_every=intervalo)(vista_mapa)(nombre, dia)


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

    st.subheader('Datos y licencias')
    st.markdown(
        '- Espiras de tráfico y estado de los tramos (con su geometría oficial), '
        'espiras de bicicleta, estaciones de Valenbisi y ruido ambiental.\n'
        '- Ruido ambiental: solo 15 sensores, todos en Ruzafa.\n'
        f'- {FUENTE_DATOS}.\n'
        '- Meteorología diaria: AEMET (Agencia Estatal de Meteorología).\n'
        '- Red viaria y mapa base: © colaboradores de OpenStreetMap.\n'
        f'- El código se publica con licencia MIT en [GitHub]({GITHUB}).')

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

    nombre, dia = barra_lateral()
    mapa, comparativa, proyecto = st.tabs(
        ['Mapa de predicción', 'Comparativa de modelos', 'Sobre el proyecto'])
    with mapa:
        pestana_mapa(nombre, dia)
    with comparativa:
        pestana_comparativa()
    with proyecto:
        pestana_proyecto()
    pie()


main()
