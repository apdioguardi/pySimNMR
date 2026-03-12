# pySimNMR package
from .core import *  # re-export legacy core for now

from .hyperfine_distribution import (
    freq_spectrum_from_internal_field_distribution,
    save_internal_field_distribution_html,
    validate_internal_field_samples,
)

# Package version (from installed metadata). Falls back for editable/dev.
try:  # Python 3.8+
    from importlib.metadata import version as _pkg_version
except Exception:  # pragma: no cover
    _pkg_version = None

try:
    __version__ = _pkg_version("pySimNMR") if _pkg_version else "0+unknown"
except Exception:  # during editable installs or source checkout
    __version__ = "0+unknown"
