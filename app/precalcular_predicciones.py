"""
precalcular_predicciones.py — Calcula con inferencia_demo.predecir() las
predicciones de gat_plus en todos los instantes "y cuarto" (:15) de cada
semana de la muestra y las guarda en pipeline_v2_recon/demo/ como
predicciones_<semana>.npz (comprimido), para que la app cargue rápido sin
importar torch.

Solo se usan los :15 porque la intensidad (ih) es horaria y cambia a y
cuarto: en :00, :30 y :45 el valor real es el último que vio el modelo.

Claves de cada .npz:
    instantes   (168,)        datetime64[ns], los :15 de la semana
    prediccion  (168, 1035)   float32, veh/h
    real        (168, 1035)   float32, veh/h; NaN donde la máscara no es válida
    verificable (168,)        bool, False si no hay ningún valor real
    persistencia (168, 1035)  float32, veh/h; Y[t], el último valor antes del instante

Ejecutar desde la raíz del repositorio:
    .venv\\Scripts\\python.exe app\\precalcular_predicciones.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from inferencia_demo import (  # noqa: E402
    DEMO_DIR,
    SEMANAS,
    Semana,
    cargar_modelo,
    cargar_semana,
    instantes_disponibles,
    predecir,
)

MINUTO = 15


def ruta_predicciones(nombre: str) -> Path:
    """Ruta del .npz de predicciones de la semana `nombre`."""
    return DEMO_DIR / f'predicciones_{nombre}.npz'


def posiciones_cuarto(semana: Semana) -> np.ndarray:
    """Posiciones (en instantes_disponibles) de los instantes :15."""
    return np.flatnonzero(instantes_disponibles(semana).minute == MINUTO)


def precalcular(modelo, semana: Semana) -> dict[str, np.ndarray]:
    """Predice todos los instantes :15 de `semana` y devuelve los arrays
    que se guardan en el .npz."""
    posiciones = posiciones_cuarto(semana)
    prediccion, real, verificable, persistencia = [], [], [], []
    for j in posiciones:
        r = predecir(modelo, semana, int(j))
        prediccion.append(r.tabla['prediccion_veh_h'].to_numpy())
        real.append(r.tabla['real_veh_h'].to_numpy())
        verificable.append(r.verificable)
        persistencia.append(r.tabla['persistencia_veh_h'].to_numpy())
    return {
        'instantes': instantes_disponibles(semana)[posiciones].to_numpy().astype('datetime64[ns]'),
        'prediccion': np.stack(prediccion).astype(np.float32),
        'real': np.stack(real).astype(np.float32),
        'verificable': np.array(verificable, dtype=bool),
        'persistencia': np.stack(persistencia).astype(np.float32),
    }


def main() -> None:
    """Genera predicciones_<semana>.npz para las dos semanas."""
    modelo = cargar_modelo()
    for nombre in SEMANAS:
        arrays = precalcular(modelo, cargar_semana(nombre))
        ruta = ruta_predicciones(nombre)
        np.savez_compressed(ruta, **arrays)
        print(f'  {ruta.name}: {len(arrays["instantes"])} instantes, '
              f'{int((~arrays["verificable"]).sum())} no verificables, '
              f'{ruta.stat().st_size / 1e6:.2f} MB')


if __name__ == '__main__':
    main()
