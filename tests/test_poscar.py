"""
Tests for yamatsumi.poscar.

Includes the regression tests for Bug B (POSCAR scale-factor handling,
including VASP's negative-scale "target volume" convention) and Bug C
(Selective Dynamics constraint flags), plus the original B-1 malformed-input
validation behaviour.
"""
from __future__ import annotations

import numpy as np
import pytest

from yamatsumi.errors import NEBError
from yamatsumi.poscar import Poscar, read_poscar, write_poscar


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return str(path)


VASP5_TEXT = """Test comment
   1.0
   2.0 0.0 0.0
   0.0 2.0 0.0
   0.0 0.0 2.0
Si
2
Direct
0.0 0.0 0.0
0.5 0.5 0.5
"""

VASP4_TEXT = """VASP4 test
   1.0
   3.0 0.0 0.0
   0.0 3.0 0.0
   0.0 0.0 3.0
2
Direct
0.0 0.0 0.0
0.5 0.5 0.5
"""


def test_vasp5_round_trip(tmp_path):
    path = _write(tmp_path, "POSCAR", VASP5_TEXT)
    p = read_poscar(path)
    assert p.species == ["Si"]
    assert p.counts == [2]
    assert p.coord_type == "Direct"
    assert p.scale == pytest.approx(1.0)
    assert np.allclose(p.lattice, np.diag([2.0, 2.0, 2.0]))
    assert np.allclose(p.positions, [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    assert p.selective_dynamics is None

    out_path = str(tmp_path / "POSCAR_out")
    write_poscar(out_path, p)
    p2 = read_poscar(out_path)
    assert p == p2


def test_vasp4_round_trip(tmp_path):
    path = _write(tmp_path, "POSCAR", VASP4_TEXT)
    p = read_poscar(path)
    assert p.species == []
    assert p.counts == [2]
    assert np.allclose(p.lattice, np.diag([3.0, 3.0, 3.0]))


def test_bug_b_scale_factor_baked_into_lattice_cartesian(tmp_path):
    """
    Regression for Bug B: scale=2.0 with a Cartesian-coordinate POSCAR must
    resolve to the true fractional coordinate (1,1,1), not (2,2,2) -- and
    scale/lattice must be normalised (scale -> 1.0, lattice -> true 2*I).
    """
    text = (
        "Bug B cartesian test\n"
        "   2.0\n"
        "   1.0 0.0 0.0\n"
        "   0.0 1.0 0.0\n"
        "   0.0 0.0 1.0\n"
        "H\n"
        "1\n"
        "Cartesian\n"
        "1.0 1.0 1.0\n"
    )
    path = _write(tmp_path, "POSCAR", text)
    p = read_poscar(path)

    assert p.scale == pytest.approx(1.0)
    assert np.allclose(p.lattice, np.diag([2.0, 2.0, 2.0]))
    assert np.allclose(p.positions, [[1.0, 1.0, 1.0]], atol=1e-9)


def test_bug_b_negative_scale_volume_convention(tmp_path):
    """
    Regression for Bug B: a negative scale factor means |scale| is the
    target cell volume (Ang^3); here raw lattice = I (det=1), target
    volume=8.0 => true linear scale = 8**(1/3) = 2.0 => lattice = 2*I,
    lattice_volume == 8.0.
    """
    text = (
        "Negative scale test\n"
        "   -8.0\n"
        "   1.0 0.0 0.0\n"
        "   0.0 1.0 0.0\n"
        "   0.0 0.0 1.0\n"
        "H\n"
        "1\n"
        "Direct\n"
        "0.25 0.25 0.25\n"
    )
    path = _write(tmp_path, "POSCAR", text)
    p = read_poscar(path)

    assert p.scale == pytest.approx(1.0)
    assert np.allclose(p.lattice, np.diag([2.0, 2.0, 2.0]), atol=1e-9)
    assert np.allclose(p.positions, [[0.25, 0.25, 0.25]])
    assert p.lattice_volume == pytest.approx(8.0, abs=1e-8)


def test_bug_c_selective_dynamics_round_trip(tmp_path):
    """Regression for Bug C: Selective Dynamics flags must survive read+write."""
    text = (
        "SD test\n"
        "   1.0\n"
        "   5.0 0.0 0.0\n"
        "   0.0 5.0 0.0\n"
        "   0.0 0.0 5.0\n"
        "H O\n"
        "1 1\n"
        "Selective dynamics\n"
        "Direct\n"
        "0.0 0.0 0.0 T T T\n"
        "0.5 0.5 0.5 F F T\n"
    )
    path = _write(tmp_path, "POSCAR", text)
    p = read_poscar(path)

    assert p.selective_dynamics is not None
    assert p.selective_dynamics.tolist() == [[True, True, True], [False, False, True]]

    out_path = str(tmp_path / "POSCAR_out")
    write_poscar(out_path, p)
    p2 = read_poscar(out_path)
    assert p2.selective_dynamics is not None
    assert np.array_equal(p.selective_dynamics, p2.selective_dynamics)
    assert p == p2


def test_no_selective_dynamics_writes_no_block(tmp_path):
    path = _write(tmp_path, "POSCAR", VASP5_TEXT)
    p = read_poscar(path)
    out_path = str(tmp_path / "POSCAR_out")
    write_poscar(out_path, p)
    text = open(out_path).read()
    assert "Selective dynamics" not in text


# ---------------------------------------------------------------------------
# B-1 malformed-input validation (regression lock)
# ---------------------------------------------------------------------------

def test_file_too_short_raises(tmp_path):
    path = _write(tmp_path, "POSCAR", "only\nfour\nlines\nhere\n")
    with pytest.raises(NEBError):
        read_poscar(path)


def test_zero_scale_raises(tmp_path):
    text = VASP5_TEXT.replace("   1.0\n", "   0.0\n", 1)
    path = _write(tmp_path, "POSCAR", text)
    with pytest.raises(NEBError):
        read_poscar(path)


def test_singular_lattice_raises(tmp_path):
    text = (
        "singular\n"
        "   1.0\n"
        "   1.0 0.0 0.0\n"
        "   2.0 0.0 0.0\n"
        "   0.0 0.0 1.0\n"
        "H\n"
        "1\n"
        "Direct\n"
        "0.0 0.0 0.0\n"
    )
    path = _write(tmp_path, "POSCAR", text)
    with pytest.raises(NEBError):
        read_poscar(path)


def test_negative_atom_count_raises(tmp_path):
    text = VASP5_TEXT.replace("2\n", "-2\n", 1)
    path = _write(tmp_path, "POSCAR", text)
    with pytest.raises(NEBError):
        read_poscar(path)


def test_missing_position_lines_raises(tmp_path):
    text = VASP5_TEXT.replace("0.5 0.5 0.5\n", "", 1)
    path = _write(tmp_path, "POSCAR", text)
    with pytest.raises(NEBError):
        read_poscar(path)


def test_file_not_found():
    with pytest.raises(FileNotFoundError):
        read_poscar("/nonexistent/path/POSCAR")


def test_species_counts_length_mismatch_raises(tmp_path):
    """3c: a VASP5 header with 2 species symbols but 1 count must be rejected."""
    text = VASP5_TEXT.replace("Si\n2\n", "Si O\n2\n", 1)
    path = _write(tmp_path, "POSCAR", text)
    with pytest.raises(NEBError):
        read_poscar(path)


def test_unrecognised_coordinate_type_raises(tmp_path):
    """3c: a coordinate-type line that is neither Direct nor Cartesian must be rejected."""
    text = VASP5_TEXT.replace("Direct\n", "Bogus\n", 1)
    path = _write(tmp_path, "POSCAR", text)
    with pytest.raises(NEBError):
        read_poscar(path)


def test_karthesian_coordinate_type_accepted(tmp_path):
    """'Karthesian' (VASP's alternative spelling) must still be treated as Cartesian."""
    text = VASP5_TEXT.replace("Direct\n0.0 0.0 0.0\n0.5 0.5 0.5\n",
                              "Karthesian\n0.0 0.0 0.0\n1.0 1.0 1.0\n", 1)
    path = _write(tmp_path, "POSCAR", text)
    p = read_poscar(path)
    assert p.coord_type == "Direct"
    assert np.allclose(p.positions, [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])


# ---------------------------------------------------------------------------
# Poscar dataclass behaviour
# ---------------------------------------------------------------------------

def test_n_atoms_and_copy_and_equality(tmp_path):
    path = _write(tmp_path, "POSCAR", VASP5_TEXT)
    p = read_poscar(path)
    assert p.n_atoms == 2

    p2 = p.copy()
    assert p == p2
    p2.positions = p2.positions.copy()
    p2.positions[0, 0] = 0.999
    assert p != p2
    # mutating the copy must not have mutated the original
    assert p.positions[0, 0] == pytest.approx(0.0)


def test_lattice_volume(tmp_path):
    path = _write(tmp_path, "POSCAR", VASP5_TEXT)
    p = read_poscar(path)
    assert p.lattice_volume == pytest.approx(8.0)


def test_with_positions_shares_other_fields_by_reference(tmp_path):
    """
    2a: with_positions() must share lattice/species/counts by reference
    (not deep-copy them) while still producing an object whose positions
    are independent of the original.
    """
    path = _write(tmp_path, "POSCAR", VASP5_TEXT)
    p = read_poscar(path)
    new_positions = p.positions + 0.1
    p2 = p.with_positions(new_positions)

    assert p2.lattice is p.lattice
    assert p2.species is p.species
    assert p2.counts is p.counts
    assert p2.positions is new_positions
    assert np.allclose(p2.positions, p.positions + 0.1)
    assert not np.allclose(p2.positions, p.positions)
