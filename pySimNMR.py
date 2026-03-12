"""Compatibility shim for legacy scripts.

Allows `import pySimNMR` to continue working by re-exporting from
the new `pysimnmr.core` module.
"""

from pysimnmr.core import *  # noqa: F401,F403

