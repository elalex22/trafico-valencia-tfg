"""
02_construccion_grafo.py — Grafo viario de Valencia + map-matching de espiras
+ construcción de aristas (regla secuencial + regla misma calle) +
asignación espira→tramo (para la feature 'estado' de estat_trafic).

Requiere: 01_extraccion.py ya ejecutado (usa espires_trafico.parquet para
filtrar espiras "vivas"; id_sensor llega como int64, igual que el resto
del pipeline real).

Genera:
  - grafo/valencia_drive.graphml
  - eda_grafo/coords_espiras_trafico_v2.csv
  - eda_grafo/mapmatching_prueba.csv
  - grafo/node_order_1035.csv
  - grafo/adj_final.npy
  - grafo/edge_index_final.npy
  - eda_grafo/coords_tramos_estat_v2.csv
  - grafo/espira_to_tramo.csv

── CAMBIOS respecto a la versión anterior, para que coincida con lo
   descrito en la memoria ──

1. CORRECCIÓN DE SESGO GEOGRÁFICO en la distancia de map-matching
   (`filtrar_vivas_y_mapmatch`). Antes: `pt.distance(geom) * 111_320`,
   igual para lat y lon. A la latitud de Valencia (39,47°N) eso
   sobreestima ~29,5% las distancias con componente este-oeste
   (verificado en sandbox). Ahora se proyecta el punto y la geometría a
   un plano local en metros (escalando la longitud por cos(lat_ref))
   antes de calcular la distancia — corrige tanto el número reportado
   como, potencialmente, qué arista se elige como "más cercana" en casos
   límite.
2. Verificaciones explícitas contra la Tabla 5.1 de la memoria: nº de
   nodos (1.035), aristas no dirigidas (2.863), grado medio (5,5),
   componentes conexos (1), tramos viarios cubiertos (295) y espiras
   medias por tramo (3,5). Si el pipeline real da otra cifra, se imprime
   un aviso explícito en vez de quedar solo en el log genérico.
"""

import time
import math
import heapq
import zipfile
from collections import defaultdict

import numpy as np
import pandas as pd
import networkx as nx
import osmnx as ox
from shapely.geometry import Point, LineString
from shapely.ops import transform as shp_transform

from config import RAIZ, PROC, GRAPH_DIR, EDA

# Latitud de referencia de Valencia, para proyectar longitud a metros sin
# el sesgo este-oeste (mismo criterio que en 03_asignaciones_espaciales.py).
LAT_REF = 39.47


def _proyectar_local(x, y, z=None):
    """Escala la longitud por cos(lat_ref); una distancia euclídea sobre
    coordenadas ya proyectadas así, multiplicada por 111.320, da metros
    reales sin sesgo este-oeste."""
    return (x * math.cos(math.radians(LAT_REF)), y)


# ══════════════════════════════════════════════════════════════
# PASO 1 — Descargar/cargar grafo viario OSMnx
# ══════════════════════════════════════════════════════════════
def obtener_grafo_osm():
    graphml_path = GRAPH_DIR / 'valencia_drive.graphml'
    if graphml_path.exists():
        print(f'  Ya existe {graphml_path.name}, cargando...')
        G = ox.load_graphml(graphml_path)
    else:
        print('  Descargando red viaria de Valencia (drive)...')
        t0 = time.time()
        G = ox.graph_from_place('Valencia, Spain', network_type='drive')
        ox.save_graphml(G, graphml_path)
        print(f'  Descarga: {time.time()-t0:.0f}s')

    print(f'  Nodos: {G.number_of_nodes():,} | Aristas: {G.number_of_edges():,}')
    return G


# ══════════════════════════════════════════════════════════════
# PASO 2 — Extraer coordenadas de espiras de tráfico (idpm + geo_point_2d)
# ══════════════════════════════════════════════════════════════
def extraer_coords_espiras():
    carpeta_zips = RAIZ / 'espires_VLC2-main'
    zip_files = sorted(carpeta_zips.glob('*.zip'))
    print(f'  Procesando {len(zip_files)} ZIPs...')

    coords = {}
    for i, zf in enumerate(zip_files):
        try:
            with zipfile.ZipFile(zf, 'r') as z:
                csv_names = [n for n in z.namelist() if n.endswith('.csv')]
                if not csv_names:
                    continue
                with z.open(csv_names[0]) as f:
                    df = pd.read_csv(f, sep=';', nrows=200, encoding='utf-8',
                                      on_bad_lines='skip')
                    cols = {c.lower().strip(): c for c in df.columns}
                    id_col = cols.get('idpm')
                    geo_col = cols.get('geo_point_2d') or cols.get('geo point')
                    if not id_col or not geo_col:
                        continue
                    for _, row in df.iterrows():
                        sid, geo = row[id_col], row[geo_col]
                        if pd.isna(geo) or pd.isna(sid):
                            continue
                        try:
                            sid_int = int(float(sid))
                            parts = str(geo).replace(',', ' ').split()
                            lat, lon = float(parts[0]), float(parts[1])
                            if 39.0 < lat < 40.0 and -1.0 < lon < 0.0:
                                coords[sid_int] = (lat, lon)
                        except (ValueError, IndexError):
                            continue
        except Exception:
            continue
        if (i + 1) % 100 == 0:
            print(f'    ZIP {i+1}/{len(zip_files)} — {len(coords)} sensores')

    df_coords = pd.DataFrame([
        {'id_sensor': k, 'lat': v[0], 'lon': v[1]} for k, v in coords.items()
    ]).sort_values('id_sensor').reset_index(drop=True)

    df_coords.to_csv(EDA / 'coords_espiras_trafico_v2.csv', index=False)
    print(f'  Guardado: coords_espiras_trafico_v2.csv ({len(df_coords)} sensores)')
    return df_coords


# ══════════════════════════════════════════════════════════════
# PASO 3 — Filtrar espiras "vivas" (ih_max > 0) + map-matching a OSM
# ══════════════════════════════════════════════════════════════
def filtrar_vivas_y_mapmatch(G, df_coords):
    df_esp = pd.read_parquet(PROC / 'espires_trafico.parquet')
    # id_sensor llega como int64 desde 01_extraccion.py (igual que el
    # pipeline real). Cast defensivo sin efecto si ya es int.
    df_esp['id_sensor'] = df_esp['id_sensor'].astype(int)

    cob = df_esp.groupby('id_sensor').agg(
        ih_max=('ih_trafico', 'max'), n=('timestamp', 'count')).reset_index()

    vivas = set(cob.loc[cob['ih_max'] > 0, 'id_sensor'].astype(int))
    df_vivas = df_coords[df_coords['id_sensor'].isin(vivas)].copy()

    print(f'  Total sensores: {len(cob)} | Vivos: {(cob["ih_max"] > 0).sum()} | '
          f'Muertos: {(cob["ih_max"] == 0).sum()} | Vivos con coords: {len(df_vivas)}')
    del df_esp

    print('  Map-matching (nearest_edges)...')
    t0 = time.time()
    lats, lons = df_vivas['lat'].values, df_vivas['lon'].values
    nearest = ox.nearest_edges(G, lons, lats)
    print(f'    Completado en {time.time()-t0:.1f}s')

    distances = []
    for i, (u, v, key) in enumerate(nearest):
        pt = Point(lons[i], lats[i])
        ed = G.edges[u, v, key]
        geom = ed.get('geometry') or LineString([
            (G.nodes[u]['x'], G.nodes[u]['y']), (G.nodes[v]['x'], G.nodes[v]['y'])])
        # Proyección local antes de medir distancia — corrige el sesgo
        # este-oeste (ver docstring del módulo).
        pt_proj = shp_transform(_proyectar_local, pt)
        geom_proj = shp_transform(_proyectar_local, geom)
        distances.append(pt_proj.distance(geom_proj) * 111_320)

    df_vivas['osm_u'] = [int(n[0]) for n in nearest]
    df_vivas['osm_v'] = [int(n[1]) for n in nearest]
    df_vivas['osm_key'] = [int(n[2]) for n in nearest]
    df_vivas['dist_edge_m'] = distances
    df_vivas.to_csv(EDA / 'mapmatching_prueba.csv', index=False)

    mediana = np.median(distances)
    p99 = np.percentile(distances, 99)
    print(f'    Dist. mediana: {mediana:.1f}m | P99: {p99:.1f}m | '
          f'>100m: {sum(1 for d in distances if d > 100)}')
    print(f'    (memoria Tabla 11.2: mediana 1,1m, P99 7,7m, 0 espiras >100m — '
          f'comparar tras corrección de sesgo geográfico)')
    return df_vivas, df_coords


def guardar_node_order(df_vivas, df_coords):
    espira_ids = sorted(df_vivas['id_sensor'].astype(int).tolist())
    n = len(espira_ids)
    node_order = pd.DataFrame({'idx': range(n), 'id_sensor': espira_ids})
    node_order = node_order.merge(df_coords, on='id_sensor', how='left')
    node_order.to_csv(GRAPH_DIR / 'node_order_1035.csv', index=False)
    print(f'  Guardado: node_order_1035.csv ({n} nodos)')
    if n != 1035:
        print(f'  ⚠️ Memoria (Tabla 5.1) declara 1.035 nodos; el pipeline real da {n}. '
              f'Actualizar la memoria o revisar el filtro de espiras "vivas".')
    return node_order


# ══════════════════════════════════════════════════════════════
# PASO 4 — Construir aristas: regla secuencial (BFS) + regla misma calle
# ══════════════════════════════════════════════════════════════
def construir_grafo_definitivo(G_dir):
    G_und = G_dir.to_undirected()
    mm = pd.read_csv(EDA / 'mapmatching_prueba.csv').sort_values('id_sensor').reset_index(drop=True)
    nodes_df = pd.read_csv(GRAPH_DIR / 'node_order_1035.csv')

    espira_ids = sorted(mm['id_sensor'].astype(int).tolist())
    n = len(espira_ids)
    id_to_idx = {sid: i for i, sid in enumerate(espira_ids)}

    espira_name, espira_edge_ends, espira_coord = {}, {}, {}
    for _, row in mm.iterrows():
        sid = int(row['id_sensor'])
        u, v = int(row['osm_u']), int(row['osm_v'])
        espira_edge_ends[sid] = (u, v)
        espira_coord[sid] = (row['lat'], row['lon'])
        if G_dir.has_edge(u, v):
            data = list(G_dir[u][v].values())[0]
            name = data.get('name', '')
            if isinstance(name, list):
                name = str(name[0])
            espira_name[sid] = str(name) if name else None
        else:
            espira_name[sid] = None

    print(f'  Espiras: {n} | Con nombre de calle: '
          f'{sum(1 for v in espira_name.values() if v)}')

    # ── Regla 1: vecinos secuenciales (BFS bloqueando en la propia arista) ──
    print('  Regla 1: vecinos secuenciales...')
    edge_to_espiras = defaultdict(list)
    for sid, (u, v) in espira_edge_ends.items():
        edge_to_espiras[frozenset([u, v])].append(sid)
    edges_con_espira = set(edge_to_espiras.keys())

    adj = defaultdict(set)
    for u, v in G_und.edges():
        adj[u].add(v)
        adj[v].add(u)

    def buscar_siguiente_espira(start_node, forbidden_edge):
        encontrados = []
        for next_node in adj[start_node]:
            edge_actual = frozenset([start_node, next_node])
            if edge_actual == forbidden_edge:
                continue
            visited = {start_node, next_node}
            queue = [(next_node, edge_actual)]
            while queue:
                current, prev_edge = queue.pop(0)
                if prev_edge in edges_con_espira:
                    encontrados.extend(edge_to_espiras[prev_edge])
                    break
                for nxt in adj[current]:
                    if nxt in visited:
                        continue
                    new_edge = frozenset([current, nxt])
                    visited.add(nxt)
                    queue.append((nxt, new_edge))
        return encontrados

    vecinos = defaultdict(set)
    t0 = time.time()
    for edge_key, sids_here in edge_to_espiras.items():
        u, v = tuple(edge_key)
        todas = set(buscar_siguiente_espira(u, edge_key)) | set(buscar_siguiente_espira(v, edge_key))
        for sid in sids_here:
            for vec in todas:
                if vec != sid:
                    vecinos[sid].add(vec)
            for other in sids_here:
                if other != sid:
                    vecinos[sid].add(other)

    aristas_secuencial = set()
    for sid_a, vecs in vecinos.items():
        ia = id_to_idx[sid_a]
        for sid_b in vecs:
            ib = id_to_idx[sid_b]
            aristas_secuencial.add((min(ia, ib), max(ia, ib)))
    print(f'    Aristas secuenciales: {len(aristas_secuencial)} en {time.time()-t0:.1f}s')

    # ── Regla 2: garantía por misma calle (2 más cercanas) ──
    print('  Regla 2: misma calle...')
    calle_a_espiras = defaultdict(list)
    for sid, name in espira_name.items():
        if name:
            calle_a_espiras[name].append(sid)

    def dist_euclid(a, b):
        la, lo = espira_coord[a]
        lb, lob = espira_coord[b]
        return ((la - lb) ** 2 + (lo - lob) ** 2) ** 0.5

    aristas_calle = set()
    for calle, sids_calle in calle_a_espiras.items():
        if len(sids_calle) < 2:
            continue
        for sid in sids_calle:
            otras = sorted([(o, dist_euclid(sid, o)) for o in sids_calle if o != sid],
                            key=lambda x: x[1])
            for other, _ in otras[:2]:
                ia, ib = id_to_idx[sid], id_to_idx[other]
                pair = (min(ia, ib), max(ia, ib))
                if pair not in aristas_secuencial:
                    aristas_calle.add(pair)
    print(f'    Aristas nuevas por misma calle: {len(aristas_calle)}')

    # ── Combinar y guardar ──
    aristas = aristas_secuencial | aristas_calle
    A = np.zeros((n, n), dtype=np.float32)
    for i, j in aristas:
        A[i, j] = 1
        A[j, i] = 1
    grado = A.sum(axis=1)

    G_esp = nx.Graph()
    G_esp.add_nodes_from(range(n))
    G_esp.add_edges_from(aristas)
    comps = list(nx.connected_components(G_esp))

    print('  ' + '=' * 58)
    print(f'  Nodos: {n} | Aristas totales: {len(aristas)} '
          f'(secuencial: {len(aristas_secuencial)}, calle: {len(aristas_calle)})')
    print(f'  Grado medio: {grado.mean():.1f} | mediana: {int(np.median(grado))} | '
          f'min/max: {int(grado.min())}/{int(grado.max())}')
    print(f'  Aislados: {int((grado == 0).sum())} | Componentes: {len(comps)} | '
          f'Componente principal: {max(len(c) for c in comps)} nodos')

    # ── Comparación explícita con la memoria (Tabla 5.1) ──
    if len(aristas) != 2863:
        print(f'  ⚠️ Memoria declara 2.863 aristas no dirigidas; pipeline real da {len(aristas)}.')
    if abs(grado.mean() - 5.5) > 0.1:
        print(f'  ⚠️ Memoria declara grado medio 5,5; pipeline real da {grado.mean():.2f}.')
    if len(comps) != 1:
        print(f'  ⚠️ Memoria declara 1 componente conexo; pipeline real da {len(comps)}.')

    np.save(GRAPH_DIR / 'adj_final.npy', A)
    src, dst = np.where(A > 0)
    np.save(GRAPH_DIR / 'edge_index_final.npy', np.stack([src, dst]).astype(np.int64))
    print(f'  Guardado: adj_final.npy, edge_index_final.npy '
          f'({len(src)} aristas dirigidas; memoria Tabla 5.1: 5.726)')

    return aristas_secuencial, aristas_calle


# ══════════════════════════════════════════════════════════════
# PASO 5 — Asignación espira → tramo (para 'estado' de estat_trafic)
# ══════════════════════════════════════════════════════════════
def extraer_coords_tramos(G):
    carpeta_zips = RAIZ / 'estat_trafic_VLC-main'
    zip_files = sorted(carpeta_zips.glob('*.zip'))
    print(f'  Procesando {len(zip_files)} ZIPs...')

    coords = {}
    for i, zf in enumerate(zip_files):
        try:
            with zipfile.ZipFile(zf, 'r') as z:
                csv_names = [n for n in z.namelist() if n.endswith('.csv')]
                if not csv_names:
                    continue
                with z.open(csv_names[0]) as f:
                    df = pd.read_csv(f, sep=';', nrows=1000, encoding='latin-1',
                                      on_bad_lines='skip')
                    id_col = next((c for c in df.columns
                                   if 'Id. Tram' in c or 'idpm' in c.lower()), None)
                    geo_col = next((c for c in df.columns
                                    if c.lower() == 'geo_point_2d'), None)
                    if not id_col or not geo_col:
                        continue
                    for _, row in df.iterrows():
                        sid, geo = row[id_col], row[geo_col]
                        if pd.isna(geo) or pd.isna(sid):
                            continue
                        try:
                            sid_int = int(float(sid))
                            parts = str(geo).replace(',', ' ').split()
                            lat, lon = float(parts[0]), float(parts[1])
                            if 39.0 < lat < 40.0 and -1.0 < lon < 0.0:
                                coords[sid_int] = (lat, lon)
                        except (ValueError, IndexError):
                            continue
        except Exception:
            continue
        if len(coords) >= 370:
            print(f'    ZIP {i+1}: 370 tramos extraídos, parando.')
            break
        if (i + 1) % 50 == 0:
            print(f'    ZIP {i+1}/{len(zip_files)} — {len(coords)} tramos')

    df_tramos = pd.DataFrame([
        {'id_tramo': k, 'lat': v[0], 'lon': v[1]} for k, v in coords.items()
    ]).sort_values('id_tramo').reset_index(drop=True)
    df_tramos.to_csv(EDA / 'coords_tramos_estat_v2.csv', index=False)
    print(f'  Tramos con coordenadas: {len(df_tramos)}')

    print(f'  Map-matching {len(df_tramos)} tramos a OSM...')
    t0 = time.time()
    lats_t, lons_t = df_tramos['lat'].values, df_tramos['lon'].values
    nearest_t = ox.nearest_edges(G, lons_t, lats_t)
    df_tramos['osm_u'] = [int(n[0]) for n in nearest_t]
    df_tramos['osm_v'] = [int(n[1]) for n in nearest_t]
    df_tramos['osm_key'] = [int(n[2]) for n in nearest_t]
    print(f'    Completado en {time.time()-t0:.1f}s')

    return df_tramos


def asignar_espira_a_tramo(G, df_tramos):
    mm = pd.read_csv(EDA / 'mapmatching_prueba.csv').sort_values('id_sensor').reset_index(drop=True)

    tramo_nodes = {int(r['id_tramo']): {int(r['osm_u']), int(r['osm_v'])}
                   for _, r in df_tramos.iterrows()}
    nodo_to_tramos = defaultdict(list)
    for tramo_id, nodes_set in tramo_nodes.items():
        for node in nodes_set:
            nodo_to_tramos[node].append(tramo_id)

    adj_osm = defaultdict(list)
    for u, v, data in G.edges(data=True):
        adj_osm[u].append((v, float(data.get('length', 100))))

    MAX_ROAD = 3000
    espira_to_tramo = {}
    print('  Asignando tramo a cada espira (Dijkstra, tope 3km)...')
    t0 = time.time()

    for count, (_, row) in enumerate(mm.iterrows()):
        sid = int(row['id_sensor'])
        starts = [int(row['osm_u']), int(row['osm_v'])]
        dist = {s: 0 for s in starts}
        heap = [(0, s) for s in starts]
        heapq.heapify(heap)
        visited = set()
        tramo_encontrado = None

        while heap:
            d, node = heapq.heappop(heap)
            if d > MAX_ROAD:
                break
            if node in visited:
                continue
            visited.add(node)
            if node in nodo_to_tramos:
                tramo_encontrado = nodo_to_tramos[node][0]
                break
            for neighbor, weight in adj_osm[node]:
                new_dist = d + weight
                if new_dist <= MAX_ROAD and neighbor not in visited:
                    if neighbor not in dist or new_dist < dist[neighbor]:
                        dist[neighbor] = new_dist
                        heapq.heappush(heap, (new_dist, neighbor))

        if tramo_encontrado is not None:
            espira_to_tramo[sid] = tramo_encontrado
        if count % 200 == 0 and count > 0:
            print(f'    {count}/{len(mm)}...')

    print(f'    Completado en {time.time()-t0:.1f}s | '
          f'Espiras con tramo: {len(espira_to_tramo)}/{len(mm)} '
          f'(sin tramo >3km: {len(mm)-len(espira_to_tramo)})')

    df_asig = pd.DataFrame([
        {'id_sensor': sid, 'id_tramo': tramo} for sid, tramo in espira_to_tramo.items()
    ]).sort_values('id_sensor').reset_index(drop=True)
    df_asig.to_csv(GRAPH_DIR / 'espira_to_tramo.csv', index=False)
    print(f'  Guardado: espira_to_tramo.csv ({len(df_asig)} espiras)')

    # ── Métricas de la Tabla 5.1 que faltaban: tramos cubiertos y
    # espiras medias por tramo ──
    n_tramos_cubiertos = df_asig['id_tramo'].nunique()
    espiras_por_tramo = len(df_asig) / n_tramos_cubiertos if n_tramos_cubiertos else float('nan')
    print(f'  Tramos viarios cubiertos: {n_tramos_cubiertos} (memoria Tabla 5.1: 295)')
    print(f'  Espiras medias por tramo: {espiras_por_tramo:.1f} (memoria Tabla 5.1: 3,5)')
    if n_tramos_cubiertos != 295:
        print(f'  ⚠️ Discrepancia con la memoria en nº de tramos cubiertos.')

    return df_asig


def main():
    G = obtener_grafo_osm()
    df_coords = extraer_coords_espiras()
    df_vivas, df_coords = filtrar_vivas_y_mapmatch(G, df_coords)
    guardar_node_order(df_vivas, df_coords)
    construir_grafo_definitivo(G)
    df_tramos = extraer_coords_tramos(G)
    asignar_espira_a_tramo(G, df_tramos)


if __name__ == '__main__':
    main()
