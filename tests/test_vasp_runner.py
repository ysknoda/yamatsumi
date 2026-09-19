"""Tests for yamatsumi.vasp_runner."""
from __future__ import annotations

import sys
import textwrap

from yamatsumi.vasp_runner import run_vasp


def _write_script(tmp_path, name, body):
    path = tmp_path / name
    path.write_text(textwrap.dedent(body))
    return path


def test_run_vasp_success(tmp_path):
    image_dir = tmp_path / "00"
    image_dir.mkdir()
    script = _write_script(tmp_path, "ok.py", """
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
    script = _write_script(tmp_path, "ok.py", """
        print("all good")
    """)
    assert run_vasp(image_dir, f"{sys.executable} {script}", timeout=None) is True


def test_run_vasp_executable_not_found(tmp_path):
    image_dir = tmp_path / "00"
    image_dir.mkdir()
    assert run_vasp(image_dir, "no_such_vasp_executable_xyz") is False
