"""
Tests for examples/si7_vacancy_vasp/run_validation.py that need no VASP.

Criterion A2 must judge only the OUTCARs of the run being evaluated: archived
earlier attempts or diagnostic runs kept under the same work directory must
not decide it.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = (
    Path(__file__).parent.parent / "examples" / "si7_vacancy_vasp" / "run_validation.py"
)


def _load_script():
    spec = importlib.util.spec_from_file_location("si7_run_validation", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a2_ignores_outcars_outside_the_current_run(tmp_path):
    script = _load_script()
    for rel in (
        "relax_init",
        "neb/00",
        "neb/03/phonon",
        "attempt2_metastable_endpoints/neb/03",
        "diag/c3v",
    ):
        d = tmp_path / rel
        d.mkdir(parents=True)
        (d / "OUTCAR").write_text("")
    found = sorted(
        str(p.relative_to(tmp_path)) for p in script.current_run_outcars(tmp_path)
    )
    assert found == ["neb/00/OUTCAR", "neb/03/phonon/OUTCAR", "relax_init/OUTCAR"]
