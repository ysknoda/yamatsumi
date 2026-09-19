"""Tests for yamatsumi.directories."""
from __future__ import annotations

import pytest

from yamatsumi.directories import (
    check_vasp_relaxation_disabled,
    get_image_dirs,
    prepare_image_directory,
)
from yamatsumi.errors import NEBError


def test_prepare_image_directory_copies_templates_and_skips_poscar(tmp_path):
    copy_dir = tmp_path / "CopyDir"
    copy_dir.mkdir()
    (copy_dir / "INCAR").write_text("ENCUT = 400\n")
    (copy_dir / "KPOINTS").write_text("Gamma\n")
    (copy_dir / "POSCAR").write_text("should not be copied\n")

    image_dir = tmp_path / "00"
    prepare_image_directory(image_dir, copy_dir)

    assert (image_dir / "INCAR").read_text() == "ENCUT = 400\n"
    assert (image_dir / "KPOINTS").read_text() == "Gamma\n"
    assert not (image_dir / "POSCAR").exists()


def test_prepare_image_directory_missing_copy_dir_raises(tmp_path):
    with pytest.raises(NEBError):
        prepare_image_directory(tmp_path / "00", tmp_path / "NoSuchCopyDir")


def test_prepare_image_directory_removes_stale_outcar_and_contcar(tmp_path):
    """
    3d regression: a leftover OUTCAR/CONTCAR from a crashed or earlier VASP
    run must be removed before new inputs are staged, so it can never be
    silently mistaken for the result of the run about to happen.
    """
    copy_dir = tmp_path / "CopyDir"
    copy_dir.mkdir()
    (copy_dir / "INCAR").write_text("ENCUT = 400\n")

    image_dir = tmp_path / "00"
    image_dir.mkdir()
    (image_dir / "OUTCAR").write_text("stale energy data\n")
    (image_dir / "CONTCAR").write_text("stale relaxed coordinates\n")

    prepare_image_directory(image_dir, copy_dir)

    assert not (image_dir / "OUTCAR").exists()
    assert not (image_dir / "CONTCAR").exists()
    assert (image_dir / "INCAR").exists()


def test_prepare_image_directory_no_op_when_no_stale_files(tmp_path):
    """Cleanup must not raise when there is nothing stale to remove."""
    copy_dir = tmp_path / "CopyDir"
    copy_dir.mkdir()
    (copy_dir / "INCAR").write_text("ENCUT = 400\n")
    prepare_image_directory(tmp_path / "00", copy_dir)  # must not raise


def test_get_image_dirs_count_and_naming():
    from pathlib import Path

    dirs = get_image_dirs(Path("/work"), num_images=5)
    assert len(dirs) == 7  # 5 intermediates + 2 endpoints
    assert [d.name for d in dirs] == ["0", "1", "2", "3", "4", "5", "6"]


def test_get_image_dirs_zero_pads_for_double_digit_totals():
    from pathlib import Path

    dirs = get_image_dirs(Path("/work"), num_images=9)
    assert [d.name for d in dirs] == [str(i).zfill(2) for i in range(11)]


# ---------------------------------------------------------------------------
# 3g: check_vasp_relaxation_disabled
# ---------------------------------------------------------------------------

def test_relaxation_check_passes_with_no_incar(tmp_path):
    check_vasp_relaxation_disabled(tmp_path)  # must not raise


def test_relaxation_check_passes_with_no_relevant_tags(tmp_path):
    (tmp_path / "INCAR").write_text("ENCUT = 400\nISMEAR = 0\n")
    check_vasp_relaxation_disabled(tmp_path)  # defaults NSW=0, IBRION=-1 -- must not raise


def test_relaxation_check_passes_with_explicit_single_point_tags(tmp_path):
    (tmp_path / "INCAR").write_text("NSW = 0\nIBRION = -1\n")
    check_vasp_relaxation_disabled(tmp_path)  # must not raise


def test_relaxation_check_raises_on_nonzero_nsw(tmp_path):
    (tmp_path / "INCAR").write_text("NSW = 100\n")
    with pytest.raises(NEBError):
        check_vasp_relaxation_disabled(tmp_path)


def test_relaxation_check_raises_on_relaxation_ibrion(tmp_path):
    (tmp_path / "INCAR").write_text("IBRION = 2\nNSW = 50\n")
    with pytest.raises(NEBError):
        check_vasp_relaxation_disabled(tmp_path)


def test_relaxation_check_override_skips_validation(tmp_path):
    (tmp_path / "INCAR").write_text("IBRION = 2\nNSW = 50\n")
    check_vasp_relaxation_disabled(tmp_path, allow_override=True)  # must not raise


def test_relaxation_check_handles_inline_comment_and_lowercase_tag(tmp_path):
    (tmp_path / "INCAR").write_text("nsw = 0 ! keep endpoints fixed\nibrion=-1\n")
    check_vasp_relaxation_disabled(tmp_path)  # must not raise


def test_relaxation_check_last_occurrence_wins(tmp_path):
    """A later re-assignment of the same tag overrides an earlier one."""
    (tmp_path / "INCAR").write_text("NSW = 50\nNSW = 0\nIBRION = -1\n")
    check_vasp_relaxation_disabled(tmp_path)  # must not raise (final NSW=0)
