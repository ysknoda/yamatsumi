"""
yamatsumi: CI-NEB (Climbing Image Nudged Elastic Band) transition-state search
driven by VASP.

Run as a program with ``python -m yamatsumi`` (see ``yamatsumi.cli.build_parser``
for the full option reference, or the ``yamatsumi`` console script installed
by ``pip install``).
"""
from .errors import NEBError
from .poscar import Poscar, read_poscar, write_poscar

__version__ = "1.0.0a1"

__all__ = ["NEBError", "Poscar", "read_poscar", "write_poscar", "__version__"]
