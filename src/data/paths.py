import os
from pathlib import Path
from typing import TypeVar

PathLike = TypeVar("PathLike", str, bytes, os.PathLike)

project_dir = Path(__file__).resolve().parents[2]
