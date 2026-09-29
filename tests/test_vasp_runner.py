"""Tests for yamatsumi.vasp_runner."""
from __future__ import annotations

import sys
import textwrap

from yamatsumi.vasp_runner import run_vasp


# Written by the fake VASP scripts below that should count as successful:
# run_vasp now also requires every ionic step's SCF to have converged
# (ADR 0003). Marker lines are verbatim from a real VASP 6.6.1 OUTCAR.
_CONVERGED_OUTCAR = (
    "------------------------ aborting loop because EDIFF is reached "
    "----------------------------------------\\n"
    "  free  energy   TOTEN  =       -10.81465686 eV\\n"
    "     LOOP+:  cpu time      2.5090: real time      1.9252\\n"
)


def _write_script(tmp_path, name, body):
    path = tmp_path / name
    path.write_text(textwrap.dedent(body))
    return path


def test_run_vasp_success(tmp_path):
    image_dir = tmp_path / "00"
    image_dir.mkdir()
    script = _write_script(tmp_path, "ok.py", f"""
        open("OUTCAR", "w").write("{_CONVERGED_OUTCAR}")
        print("all good")
    """)
    assert run_vasp(image_dir, f"{sys.executable} {script}") is True
    assert "all good" in (image_dir / "vasp_run.log").read_text()


def test_run_vasp_nonzero_exit_fails(tmp_path):
    image_dir = tmp_path / "00"
    image_dir.mkdir()
    script = _write_script(tmp_path, "fail.py", """
        import sys
        sys.exit(1)
    """)
    assert run_vasp(image_dir, f"{sys.executable} {script}") is False


def test_run_vasp_fatal_error_banner_fails_despite_exit_code_zero(tmp_path):
    """
    3f regression: a known VASP fatal-error banner in the captured log must
    fail the run even though the process itself exits with code 0 -- an
    exit-code-only check would silently accept this as success.
    """
    image_dir = tmp_path / "00"
    image_dir.mkdir()
    script = _write_script(tmp_path, "silent_fatal.py", """
        print("ZBRENT: fatal error in bracketing")
    """)
    assert run_vasp(image_dir, f"{sys.executable} {script}") is False


def test_run_vasp_timeout_kills_hung_process(tmp_path):
    """3f regression: a hung VASP process must be killed, not block forever."""
    image_dir = tmp_path / "00"
    image_dir.mkdir()
    script = _write_script(tmp_path, "hang.py", """
        import time
        time.sleep(30)
    """)
    assert run_vasp(image_dir, f"{sys.executable} {script}", timeout=0.5) is False


def test_run_vasp_no_timeout_by_default_behaviour_unaffected(tmp_path):
    """A fast, well-behaved run must still succeed when timeout is unset (None)."""
    image_dir = tmp_path / "00"
    image_dir.mkdir()
    script = _write_script(tmp_path, "ok.py", f"""
        open("OUTCAR", "w").write("{_CONVERGED_OUTCAR}")
        print("all good")
    """)
    assert run_vasp(image_dir, f"{sys.executable} {script}", timeout=None) is True


def test_run_vasp_executable_not_found(tmp_path):
    image_dir = tmp_path / "00"
    image_dir.mkdir()
    assert run_vasp(image_dir, "no_such_vasp_executable_xyz") is False


def test_run_vasp_unconverged_scf_fails_despite_exit_code_zero(tmp_path):
    """
    ADR 0003 regression: VASP exits 0 when it reaches NELM without meeting
    EDIFF (checked on VASP 6.6.1). Such a run must not count as a success.
    """
    image_dir = tmp_path / "00"
    image_dir.mkdir()
    script = _write_script(tmp_path, "unconverged.py", """
        open("OUTCAR", "w").write(
            "   NELM   =      3;   NELMIN=  2; NELMDL=  0     # of ELM steps \\n"
            "  free  energy   TOTEN  =       -10.99766482 eV\\n"
            "     LOOP+:  cpu time      0.3726: real time      0.4048\\n"
        )
    """)
    assert run_vasp(image_dir, f"{sys.executable} {script}") is False


def test_run_vasp_without_outcar_fails_despite_exit_code_zero(tmp_path):
    image_dir = tmp_path / "00"
    image_dir.mkdir()
    script = _write_script(tmp_path, "no_outcar.py", """
        print("exited cleanly but wrote nothing")
    """)
    assert run_vasp(image_dir, f"{sys.executable} {script}") is False
