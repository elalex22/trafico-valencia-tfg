"""
01_extraccion.py — Extracción incremental de las 6 fuentes brutas a Parquet.

Fuentes:
  1. estat_trafic_VLC-main/    -> processed/estat_trafic.parquet     (id_tramo, estado)
  2. espires_VLC2-main/        -> processed/espires_trafico.parquet  (id_sensor, ih_trafico) [TARGET]
  3. espires_bici_VLC-main/    -> processed/espires_bici.parquet     (id_sensor, ih_bici)
  4. valenbici-master/         -> processed/valenbici.parquet        (id_estacion, available, free)
  5. datos_tiempo/ (JSON AEMET)-> processed/meteo_diario.parquet     (fecha, tmed, prec, tmin, tmax, sol)
  6. fusion_ruido_sensores/    -> processed/ruido.parquet            (id_sensor_ruido, laeq, laeq_n, lat, lon)

Ejecutar con el kernel/entorno limpio (los writers de parquet no toleran
archivos parcialmente escritos de una ejecución anterior).

Notas de implementación:

- Meteorología (AEMET): el sufijo "Ip" que aparece en algunos registros
  corruptos se elimina explícitamente antes de convertir a numérico;
  los valores se guardan en float32. Se descartan (quedan como NaN)
  temperaturas fuera de [-10, 50]°C y precipitaciones negativas.

- Ruido: las coordenadas con formato corrupto (más de un punto decimal,
  p. ej. "394.631.424" en vez de "39.4631424") se corrigen aquí, ya que
  03_asignaciones_espaciales.py las usa directamente para el cKDTree que
  asigna cada espira al sensor de ruido más cercano.

- `ih_trafico` / `ih_bici` se guardan con su valor crudo (incluyendo -1
  para fallos de sensor) e `id_sensor` / `id_estacion` como `int64`. La
  limpieza de -1 y outliers del target se hace más adelante, en
  `04_fusion_master.py` (rachas largas de cero) y `05_pipeline_tensores.py`
  (límite de -1/outliers antes del log1p), que necesitan el valor
  original intacto para poder detectarlo.
"""

import re
import gc
import json
import zipfile
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from config import RAIZ, PROC

PATRON_TS = re.compile(r'(\d{2}-\d{2}-\d{4}_\d{2}-\d{2}-\d{2})')


def ts_from_csv(nombre):
    m = PATRON_TS.search(nombre)
    if m:
        return pd.to_datetime(m.group(1), format='%d-%m-%Y_%H-%M-%S', errors='coerce')
    return None


def leer_csv_zip(f):
    try:
        return pd.read_csv(f, sep=None, engine='python', encoding='latin-1')
    except Exception:
        return None


def limpiar_processed():
    """Borra parquets de una ejecución anterior antes de regenerarlos."""
    for f in PROC.glob('*.parquet'):
        try:
            f.unlink()
            print(f'  🗑️ Borrado: {f.name}')
        except OSError:
            print(f'  ⚠️ No se pudo borrar: {f.name} — cierra cualquier programa que lo use')


# ══════════════════════════════════════════════════════════════
# Nota: normalize_id() vive en datos/utils_datos.py (usada por
# 04_fusion_master.py para id_tramo). No se duplica aquí porque este
# script ya no la necesita: id_tramo se guarda como string simple
# (04 la renormaliza igualmente al cargarla) e id_sensor/id_estacion se
# mantienen como int64.
# ══════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════
# Corrección de coordenadas corruptas (ej. ruido, con formato
# "394.631.424" en vez de "39.4631424"). Solo transforma strings con más
# de un punto decimal; un valor normal (0 o 1 puntos) se devuelve tal cual.
# ══════════════════════════════════════════════════════════════
def fix_corrupt_coord(value, integer_digits):
    s = str(value).strip()
    if s in ('', 'nan', 'None'):
        return None
    if s.count('.') <= 1:
        try:
            return float(s)
        except ValueError:
            return None
    sign = '-' if s.startswith('-') else ''
    digits = s.replace('.', '').replace('-', '')
    if len(digits) <= integer_digits:
        return None
    try:
        return float(f'{sign}{digits[:integer_digits]}.{digits[integer_digits:]}')
    except ValueError:
        return None


# ══════════════════════════════════════════════════════════════
# FUENTE 1/6: ESTAT_TRAFIC (estado de congestión por tramo)
# ══════════════════════════════════════════════════════════════
def extraer_estat_trafico():
    output = PROC / 'estat_trafic.parquet'
    output.unlink(missing_ok=True)

    schema = pa.schema([
        ('timestamp', pa.timestamp('us')),
        ('id_tramo', pa.string()),
        ('estado', pa.float32()),
    ])

    zips = sorted((RAIZ / 'estat_trafic_VLC-main').rglob('*.zip'))
    print(f'FUENTE 1/6: estat_trafic — {len(zips)} ZIPs')

    writer = pq.ParquetWriter(str(output), schema, compression='snappy')
    total, errores = 0, 0

    for i, zp in enumerate(zips):
        try:
            lista_bloque = []
            with zipfile.ZipFile(zp, 'r') as z:
                for csv_name in z.namelist():
                    if not csv_name.endswith('.csv'):
                        continue
                    ts = ts_from_csv(csv_name)
                    if ts is None:
                        continue
                    with z.open(csv_name) as f:
                        df = leer_csv_zip(f)
                        if df is None:
                            errores += 1
                            continue

                    col_id = [c for c in df.columns if 'Tram' in c or 'Tramo' in c]
                    col_st = [c for c in df.columns if 'Estat' in c or 'Estado' in c]
                    if not col_id or not col_st:
                        errores += 1
                        continue

                    out = pd.DataFrame({
                        'timestamp': ts,
                        'id_tramo': df[col_id[0]].astype(str).str.strip(),
                        'estado': pd.to_numeric(df[col_st[0]], errors='coerce').astype('float32'),
                    })
                    out = out[out['id_tramo'].notna() & (out['id_tramo'] != 'nan')]
                    if len(out) > 0:
                        lista_bloque.append(out)

            if lista_bloque:
                bloque = pd.concat(lista_bloque, ignore_index=True)
                tabla = pa.table({
                    'timestamp': pa.array(bloque['timestamp'].values, type=pa.timestamp('us')),
                    'id_tramo': pa.array(bloque['id_tramo'].values, type=pa.string()),
                    'estado': pa.array(bloque['estado'].values, type=pa.float32()),
                })
                writer.write_table(tabla)
                total += len(bloque)
                del lista_bloque, bloque, tabla
                gc.collect()
        except Exception:
            errores += 1

        if (i + 1) % 100 == 0:
            print(f'  {i+1}/{len(zips)} | {total:,} filas | {errores} err')

    writer.close()
    print(f'✅ estat_trafic.parquet: {total:,} filas | {errores} errores')


# ══════════════════════════════════════════════════════════════
# FUENTE 2/6: ESPIRES TRÁFICO (ih_trafico — variable target)
# ══════════════════════════════════════════════════════════════
def extraer_espires_trafico():
    output = PROC / 'espires_trafico.parquet'
    output.unlink(missing_ok=True)

    schema = pa.schema([
        ('timestamp', pa.timestamp('us')),
        ('id_sensor', pa.int64()),
        ('ih_trafico', pa.int32()),
    ])

    zips = sorted((RAIZ / 'espires_VLC2-main').rglob('*.zip'))
    print(f'FUENTE 2/6: espires_trafico — {len(zips)} ZIPs')

    writer = pq.ParquetWriter(str(output), schema, compression='snappy')
    total, errores = 0, 0

    for i, zp in enumerate(zips):
        try:
            lista_bloque = []
            with zipfile.ZipFile(zp, 'r') as z:
                for csv_name in z.namelist():
                    if not csv_name.endswith('.csv'):
                        continue
                    ts = ts_from_csv(csv_name)
                    if ts is None:
                        continue
                    with z.open(csv_name) as f:
                        df = leer_csv_zip(f)
                        if df is None:
                            errores += 1
                            continue

                    if 'idpm' not in df.columns or 'ih' not in df.columns:
                        errores += 1
                        continue

                    out = pd.DataFrame({
                        'timestamp': ts,
                        'id_sensor': pd.to_numeric(df['idpm'], errors='coerce').fillna(0).astype('int64'),
                        'ih_trafico': pd.to_numeric(df['ih'], errors='coerce').fillna(0).astype('int32'),
                    })
                    lista_bloque.append(out)

            if lista_bloque:
                bloque = pd.concat(lista_bloque, ignore_index=True)
                tabla = pa.table({
                    'timestamp': pa.array(bloque['timestamp'].values, type=pa.timestamp('us')),
                    'id_sensor': pa.array(bloque['id_sensor'].values, type=pa.int64()),
                    'ih_trafico': pa.array(bloque['ih_trafico'].values, type=pa.int32()),
                })
                writer.write_table(tabla)
                total += len(bloque)
                del lista_bloque, bloque, tabla
                gc.collect()
        except Exception:
            errores += 1

        if (i + 1) % 100 == 0:
            print(f'  {i+1}/{len(zips)} | {total:,} filas | {errores} err')

    writer.close()
    print(f'✅ espires_trafico.parquet: {total:,} filas | {errores} errores')


# ══════════════════════════════════════════════════════════════
# FUENTE 3/6: ESPIRES BICI
# ══════════════════════════════════════════════════════════════
def extraer_espires_bici():
    output = PROC / 'espires_bici.parquet'
    output.unlink(missing_ok=True)

    schema = pa.schema([
        ('timestamp', pa.timestamp('us')),
        ('id_sensor', pa.int64()),
        ('ih_bici', pa.int32()),
    ])

    zips = sorted((RAIZ / 'espires_bici_VLC-main').rglob('*.zip'))
    print(f'FUENTE 3/6: espires_bici — {len(zips)} ZIPs')

    writer = pq.ParquetWriter(str(output), schema, compression='snappy')
    total, errores = 0, 0

    for i, zp in enumerate(zips):
        try:
            lista_bloque = []
            with zipfile.ZipFile(zp, 'r') as z:
                for csv_name in z.namelist():
                    if not csv_name.endswith('.csv'):
                        continue
                    ts = ts_from_csv(csv_name)
                    if ts is None:
                        continue
                    with z.open(csv_name) as f:
                        df = leer_csv_zip(f)
                        if df is None:
                            errores += 1
                            continue

                    if 'idpm' not in df.columns or 'ih' not in df.columns:
                        errores += 1
                        continue

                    out = pd.DataFrame({
                        'timestamp': ts,
                        'id_sensor': pd.to_numeric(df['idpm'], errors='coerce').fillna(0).astype('int64'),
                        'ih_bici': pd.to_numeric(df['ih'], errors='coerce').fillna(0).astype('int32'),
                    })
                    lista_bloque.append(out)

            if lista_bloque:
                bloque = pd.concat(lista_bloque, ignore_index=True)
                tabla = pa.table({
                    'timestamp': pa.array(bloque['timestamp'].values, type=pa.timestamp('us')),
                    'id_sensor': pa.array(bloque['id_sensor'].values, type=pa.int64()),
                    'ih_bici': pa.array(bloque['ih_bici'].values, type=pa.int32()),
                })
                writer.write_table(tabla)
                total += len(bloque)
                del lista_bloque, bloque, tabla
                gc.collect()
        except Exception:
            errores += 1

        if (i + 1) % 100 == 0:
            print(f'  {i+1}/{len(zips)} | {total:,} filas | {errores} err')

    writer.close()
    print(f'✅ espires_bici.parquet: {total:,} filas | {errores} errores')


# ══════════════════════════════════════════════════════════════
# FUENTE 4/6: VALENBICI (doble formato: viejo 2022-23 / nuevo 2024+)
# ══════════════════════════════════════════════════════════════
def extraer_valenbici():
    output = PROC / 'valenbici.parquet'
    output.unlink(missing_ok=True)

    schema = pa.schema([
        ('timestamp', pa.timestamp('us')),
        ('id_estacion', pa.int64()),
        ('available', pa.int32()),
        ('free', pa.int32()),
    ])

    zips = sorted((RAIZ / 'valenbici-master').rglob('*.zip'))
    print(f'FUENTE 4/6: valenbici — {len(zips)} ZIPs (doble formato)')

    writer = pq.ParquetWriter(str(output), schema, compression='snappy')
    total, errores, fmt_viejo, fmt_nuevo = 0, 0, 0, 0

    for i, zp in enumerate(zips):
        try:
            lista_bloque = []
            with zipfile.ZipFile(zp, 'r') as z:
                for csv_name in z.namelist():
                    if not csv_name.endswith('.csv'):
                        continue
                    ts = ts_from_csv(csv_name)
                    if ts is None:
                        continue
                    with z.open(csv_name) as f:
                        df = leer_csv_zip(f)
                        if df is None:
                            errores += 1
                            continue

                    if 'number_' in df.columns:
                        id_col, avail_col, free_col = 'number_', 'available', 'free'
                        fmt_viejo += 1
                    elif 'Numero' in df.columns:
                        id_col = 'Numero'
                        avail_col = 'Bicis_disponibles'
                        free_col = 'Espacios_libres'
                        fmt_nuevo += 1
                    else:
                        errores += 1
                        continue

                    out = pd.DataFrame({
                        'timestamp': ts,
                        'id_estacion': pd.to_numeric(df[id_col], errors='coerce').fillna(0).astype('int64'),
                        'available': pd.to_numeric(df.get(avail_col, 0), errors='coerce').fillna(0).astype('int32'),
                        'free': pd.to_numeric(df.get(free_col, 0), errors='coerce').fillna(0).astype('int32'),
                    })
                    lista_bloque.append(out)

            if lista_bloque:
                bloque = pd.concat(lista_bloque, ignore_index=True)
                tabla = pa.table({
                    'timestamp': pa.array(bloque['timestamp'].values, type=pa.timestamp('us')),
                    'id_estacion': pa.array(bloque['id_estacion'].values, type=pa.int64()),
                    'available': pa.array(bloque['available'].values, type=pa.int32()),
                    'free': pa.array(bloque['free'].values, type=pa.int32()),
                })
                writer.write_table(tabla)
                total += len(bloque)
                del lista_bloque, bloque, tabla
                gc.collect()
        except Exception:
            errores += 1

        if (i + 1) % 150 == 0:
            print(f'  {i+1}/{len(zips)} | {total:,} filas | viejo:{fmt_viejo} nuevo:{fmt_nuevo} | {errores} err')

    writer.close()
    print(f'✅ valenbici.parquet: {total:,} filas | {errores} errores '
          f'(viejo: {fmt_viejo} CSVs, nuevo: {fmt_nuevo} CSVs)')


# ══════════════════════════════════════════════════════════════
# FUENTE 5/6: METEOROLOGÍA (JSONs AEMET, granularidad diaria)
# ══════════════════════════════════════════════════════════════
def extraer_meteo():
    print('FUENTE 5/6: meteorología')
    dir_tiempo = RAIZ / 'datos_tiempo'

    registros = []
    for f in sorted(dir_tiempo.iterdir()):
        if not f.name.endswith('.json') or f.stat().st_size < 100:
            continue
        with open(f, 'r', encoding='latin-1') as fh:
            data = json.load(fh)
        for r in data:
            registros.append({
                'fecha': r.get('fecha'),
                'tmed': str(r.get('tmed', '')).replace(',', '.'),
                'prec': str(r.get('prec', '')).replace(',', '.'),
                'tmin': str(r.get('tmin', '')).replace(',', '.'),
                'tmax': str(r.get('tmax', '')).replace(',', '.'),
                'sol': str(r.get('sol', '')).replace(',', '.'),
            })

    df_meteo = pd.DataFrame(registros)
    df_meteo['fecha'] = pd.to_datetime(df_meteo['fecha'])

    for col in ['tmed', 'prec', 'tmin', 'tmax', 'sol']:
        # Eliminar (no sustituir) el sufijo corrupto "Ip" que aparece en
        # algunos registros; luego convertir a float32 explícitamente.
        limpio = df_meteo[col].str.replace('Ip', '', regex=False)
        df_meteo[col] = pd.to_numeric(limpio, errors='coerce').astype('float32')

    df_meteo = df_meteo.dropna(subset=['fecha']).drop_duplicates(subset='fecha').sort_values('fecha')

    # ── Validación de rango: descarta >50°C o <-10°C, precipitaciones
    # negativas. Se descarta el valor puntual (NaN), no la fila entera,
    # para no perder el resto de variables de ese día. ──
    for col in ['tmed', 'tmin', 'tmax']:
        fuera_rango = df_meteo[col].notna() & ~df_meteo[col].between(-10, 50)
        if fuera_rango.any():
            print(f'  ⚠️ {int(fuera_rango.sum())} valores de {col} fuera de [-10, 50]°C '
                  f'descartados (puestos a NaN)')
        df_meteo.loc[fuera_rango, col] = np.float32('nan')

    prec_negativa = df_meteo['prec'].notna() & (df_meteo['prec'] < 0)
    if prec_negativa.any():
        print(f'  ⚠️ {int(prec_negativa.sum())} valores de precipitación negativos '
              f'descartados (puestos a NaN)')
    df_meteo.loc[prec_negativa, 'prec'] = np.float32('nan')

    df_meteo.to_parquet(PROC / 'meteo_diario.parquet', index=False)

    print(f'✅ meteo_diario.parquet: {len(df_meteo)} días '
          f'({df_meteo["fecha"].min().date()} → {df_meteo["fecha"].max().date()})')


# ══════════════════════════════════════════════════════════════
# FUENTE 6/6: RUIDO AMBIENTAL (Ruzafa)
# ══════════════════════════════════════════════════════════════
def extraer_ruido():
    print('FUENTE 6/6: ruido ambiental')
    dir_ruido = RAIZ / 'fusion_ruido_sensores'

    frames = []
    for csv_path in sorted(dir_ruido.iterdir()):
        if not csv_path.name.startswith('carrer-'):
            continue
        df = pd.read_csv(csv_path)
        df['id_sensor_ruido'] = csv_path.stem
        df['timestamp'] = pd.to_datetime(df['dateobserved'], errors='coerce')
        df = df.dropna(subset=['timestamp'])
        for col in ['laeq', 'laeq_n']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        frames.append(df[['timestamp', 'id_sensor_ruido', 'laeq', 'laeq_n', 'lat', 'lon']])

    df_ruido = pd.concat(frames, ignore_index=True).sort_values('timestamp')

    # ── Corrección de coordenadas corruptas, aplicada en origen.
    # fix_corrupt_coord() solo altera valores con más de un punto decimal
    # (claramente corruptos); el resto pasa sin cambios. ──
    n_lat_corruptas = df_ruido['lat'].astype(str).str.count(r'\.').gt(1).sum()
    n_lon_corruptas = df_ruido['lon'].astype(str).str.count(r'\.').gt(1).sum()
    df_ruido['lat'] = df_ruido['lat'].apply(lambda v: fix_corrupt_coord(v, integer_digits=2))
    df_ruido['lon'] = df_ruido['lon'].apply(lambda v: fix_corrupt_coord(v, integer_digits=1))
    print(f'  Coordenadas corruptas corregidas: {n_lat_corruptas} en lat, {n_lon_corruptas} en lon')

    df_ruido = df_ruido.dropna(subset=['lat', 'lon'])
    df_ruido.to_parquet(PROC / 'ruido.parquet', index=False)

    print(f'✅ ruido.parquet: {len(df_ruido):,} filas | '
          f'{df_ruido["id_sensor_ruido"].nunique()} sensores '
          f'({df_ruido["timestamp"].min().date()} → {df_ruido["timestamp"].max().date()})')


def verificar():
    print('=' * 60)
    print('  VERIFICACIÓN — 6 FUENTES EXTRAÍDAS')
    print('=' * 60)
    esperado_sensores = {
        'espires_trafico': ('id_sensor', 1035),
        'espires_bici': ('id_sensor', 136),
        'valenbici': ('id_estacion', 276),
        'ruido': ('id_sensor_ruido', 15),
    }
    for nombre in ['estat_trafic', 'espires_trafico', 'espires_bici',
                   'valenbici', 'meteo_diario', 'ruido']:
        path = PROC / f'{nombre}.parquet'
        if not path.exists():
            print(f'\n  ❌ {nombre}')
            continue
        df = pd.read_parquet(path)
        ts_col = 'timestamp' if 'timestamp' in df.columns else 'fecha'
        df[ts_col] = pd.to_datetime(df[ts_col])
        size = path.stat().st_size / (1024 * 1024)
        print(f'\n  ✅ {nombre}.parquet ({size:.1f} MB)')
        print(f'     {len(df):>12,} filas | {df[ts_col].min().date()} → {df[ts_col].max().date()}')
        if nombre in esperado_sensores:
            col, esperado = esperado_sensores[nombre]
            real = df[col].nunique()
            marca = '✓' if real == esperado else '⚠️'
            print(f'     {marca} sensores únicos: {real} (esperado: {esperado})')
        del df
    gc.collect()


def main():
    limpiar_processed()
    extraer_estat_trafico()
    extraer_espires_trafico()
    extraer_espires_bici()
    extraer_valenbici()
    extraer_meteo()
    extraer_ruido()
    verificar()


if __name__ == '__main__':
    main()
