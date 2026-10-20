"""Carga de tensores finales, compartida por 05, 09 y 10."""
import numpy as np
from config import TENSOR_DIR


def cargar_split_para_modelo(split_name):
    return dict(
        X=np.load(TENSOR_DIR / f'X_{split_name}_norm.npy'),
        Y=np.load(TENSOR_DIR / f'Y_{split_name}_log.npy'),
        M=np.load(TENSOR_DIR / f'M_{split_name}.npy'),
        Y_lag24=np.load(TENSOR_DIR / f'Y_{split_name}_lag24_log.npy'),
        M_lag24=np.load(TENSOR_DIR / f'M_{split_name}_lag24.npy'),
        Y_lag7d=np.load(TENSOR_DIR / f'Y_{split_name}_lag7d_log.npy'),
        M_lag7d=np.load(TENSOR_DIR / f'M_{split_name}_lag7d.npy'),
    )