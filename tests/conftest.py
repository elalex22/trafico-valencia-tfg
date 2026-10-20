"""Pone la raíz del repositorio en sys.path para importar app/ y modelos/."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
