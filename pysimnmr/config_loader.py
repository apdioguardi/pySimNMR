from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def load_python_config(path: Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file '{path}' does not exist")
    if path.suffix != '.py':
        raise ValueError(f"Only Python config files (*.py) are supported; got '{path.suffix}'")
    module_name = f"_pysimnmr_config_{path.stem}_{abs(hash(path))}"
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load config module from {path}")
    module = importlib.util.module_from_spec(spec)
    loader = spec.loader
    sys.modules[module_name] = module
    loader.exec_module(module)
    if not hasattr(module, "CONFIG"):
        raise AttributeError(f"Config module '{path}' must define a CONFIG dictionary")
    config = getattr(module, "CONFIG")
    if not isinstance(config, dict):
        raise TypeError(f"CONFIG in '{path}' must be a dict (got {type(config)!r})")
    return config.copy()
