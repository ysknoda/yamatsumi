"""
Tests for yamatsumi.outcar.

The frequency-token format asserted here (real: 'f  =', imaginary: 'f/i=',
sign carried by the token, not by a literal minus sign in the meV column)
is also the format the mock VASP fixture (tests/fixtures/mock_vasp.py) must
reproduce for the integration tests to be meaningful.
"""
from __future__ import annotations

import numpy as np
import pytest

from yamatsumi.outcar import (
    parse_eigenvectors_from_outcar,
    parse_energy_from_outcar,
    parse_forces_from_outcar,
    parse_frequencies_from_outcar,
)


def test_last_toten_wins(tmp_path):
    text = (
        "  free  energy   TOTEN  =   -100.000000 eV\n"
        "some other line\n"
        "  free  energy   TOTEN  =   -123.456789 eV\n"
    )
    path = tmp_path / "OUTCAR"
    path.write_text(text)
    assert parse_energy_from_outcar(path) == -123.456789


def test_energy_missing_returns_none(tmp_path):
    path = tmp_path / "OUTCAR"
    path.write_text("nothing relevant here\n")
    assert parse_energy_from_outcar(path) is None


def _force_block(rows):
    lines = [" TOTAL-FORCE (eV/Angst)\n", " -----------------------\n"]
    for x, y, z, fx, fy, fz in rows:
        lines.append(f"   {x:.6f} {y:.6f} {z:.6f}   {fx:.6f} {fy:.6f} {fz:.6f}\n")
    return "".join(lines)


def test_last_force_block_wins(tmp_path):
    first  = _force_block([(0, 0, 0, 1.0, 0.0, 0.0)])
    second = _force_block([(0, 0, 0, 2.0, 0.0, 0.0)])
    path = tmp_path / "OUTCAR"
    path.write_text(first + "\n" + second)
    forces = parse_forces_from_outcar(path, n_atoms=1)
    assert np.allclose(forces, [[2.0, 0.0, 0.0]])


def test_truncated_force_block_returns_none(tmp_path):
    text = _force_block([(0, 0, 0, 1.0, 0.0, 0.0)])  # only 1 atom's worth
    path = tmp_path / "OUTCAR"
    path.write_text(text)
    assert parse_forces_from_outcar(path, n_atoms=2) is None


def test_forces_missing_returns_none(tmp_path):
    path = tmp_path / "OUTCAR"
    path.write_text("no force block here\n")
    assert parse_forces_from_outcar(path, n_atoms=1) is None


_EIGENVECTOR_HEADER_TEXT = (
    " Eigenvectors and eigenvalues of the dynamical matrix\n"
    " ----------------------------------------------------\n"
    "\n"
)


def test_frequencies_real_and_imaginary_tokens_with_unit_suffix(tmp_path):
    """VASP variant that prints a trailing 'meV' unit label after the value."""
    text = _EIGENVECTOR_HEADER_TEXT + (
        "   1 f  =   12.345678 THz   77.567 2PiTHz  412.345 cm-1    51.123 meV\n"
        "   2 f/i=    1.234567 THz    7.756 2PiTHz   41.234 cm-1     5.112 meV\n"
    )
    path = tmp_path / "OUTCAR"
    path.write_text(text)
    freqs = parse_frequencies_from_outcar(path)
    assert len(freqs) == 2
    assert freqs[0] == 51.123
    assert freqs[1] == -5.112


def test_frequencies_real_and_imaginary_tokens_bare_numeric(tmp_path):
    """VASP variant with no trailing unit label; the meV value is the last token."""
    text = _EIGENVECTOR_HEADER_TEXT + (
        "   1 f  =   12.345678 THz   77.567 2PiTHz  412.345 cm-1    51.123\n"
        "   2 f/i=    1.234567 THz    7.756 2PiTHz   41.234 cm-1     5.112\n"
    )
    path = tmp_path / "OUTCAR"
    path.write_text(text)
    freqs = parse_frequencies_from_outcar(path)
    assert len(freqs) == 2
    assert freqs[0] == 51.123
    assert freqs[1] == -5.112


# ---------------------------------------------------------------------------
# 3b regression: frequency parser must anchor on the "Eigenvectors and
# eigenvalues" header rather than accepting any digit-led, >=3-token line.
# ---------------------------------------------------------------------------

def test_frequencies_ignore_position_force_block_before_anchor(tmp_path):
    """
    A POSITION/TOTAL-FORCE row with a non-negative first coordinate (e.g.
    "1.234000 0.300000 0.500000 ...") has >=3 whitespace-separated tokens and
    a leading digit character -- without the header anchor this would be
    misparsed as a spurious frequency (its last token would parse as float).
    """
    text = (
        " POSITION                                       TOTAL-FORCE (eV/Angst)\n"
        " -----------------------------------------------------------------------------------\n"
        "   1.234000 0.300000 0.500000   -0.500000 0.030000 0.000000\n"
        "   2.000000 0.100000 0.500000    0.100000 0.000000 0.000000\n"
        " -----------------------------------------------------------------------------------\n"
        "\n"
        " Eigenvectors and eigenvalues of the dynamical matrix\n"
        " ----------------------------------------------------\n"
        "\n"
        "   1 f  =   12.345678 THz   77.567 2PiTHz  412.345 cm-1    51.123 meV\n"
        "   2 f/i=    1.234567 THz    7.756 2PiTHz   41.234 cm-1     5.112 meV\n"
    )
    path = tmp_path / "OUTCAR"
    path.write_text(text)
    freqs = parse_frequencies_from_outcar(path)
    assert freqs == [pytest.approx(51.123), pytest.approx(-5.112)]


def test_frequencies_no_anchor_header_returns_empty(tmp_path):
    """Without the 'Eigenvectors and eigenvalues' header, nothing is a frequency."""
    text = "   1.234000 0.300000 0.500000   -0.500000 0.030000 0.000000\n"
    path = tmp_path / "OUTCAR"
    path.write_text(text)
    assert parse_frequencies_from_outcar(path) == []


def test_frequencies_last_eigenvector_section_wins(tmp_path):
    """Consistent with energy/forces: the LAST matching section is authoritative."""
    text = (
        " Eigenvectors and eigenvalues of the dynamical matrix\n"
        " ----------------------------------------------------\n"
        "   1 f  =   1.000000 THz 1.0 2PiTHz 1.0 cm-1 1.000000 meV\n"
        " Eigenvectors and eigenvalues of the dynamical matrix\n"
        " ----------------------------------------------------\n"
        "   1 f  =   2.000000 THz 1.0 2PiTHz 1.0 cm-1 2.000000 meV\n"
    )
    path = tmp_path / "OUTCAR"
    path.write_text(text)
    assert parse_frequencies_from_outcar(path) == [pytest.approx(2.0)]


# ---------------------------------------------------------------------------
# 3a regression: field-overflow ("********") markers must not raise.
# ---------------------------------------------------------------------------

def test_energy_overflow_field_skipped(tmp_path):
    text = (
        "  free  energy   TOTEN  =   -100.000000 eV\n"
        "  free  energy   TOTEN  =   ******** eV\n"
    )
    path = tmp_path / "OUTCAR"
    path.write_text(text)
    assert parse_energy_from_outcar(path) == pytest.approx(-100.000000)


def test_force_overflow_field_falls_back_to_previous_good_block(tmp_path):
    good = _force_block([(0, 0, 0, 1.0, 0.0, 0.0)])
    bad = (
        " TOTAL-FORCE (eV/Angst)\n"
        " -----------------------\n"
        "   0.0 0.0 0.0   ******** 0.0 0.0\n"
    )
    path = tmp_path / "OUTCAR"
    path.write_text(good + "\n" + bad)
    forces = parse_forces_from_outcar(path, n_atoms=1)
    assert np.allclose(forces, [[1.0, 0.0, 0.0]])


# ---------------------------------------------------------------------------
# 1d: parse_eigenvectors_from_outcar
# ---------------------------------------------------------------------------

def _eigenvector_block(mode_lines_and_rows):
    """mode_lines_and_rows: list of (header_line_without_newline, [[dx,dy,dz], ...])."""
    lines = list(_EIGENVECTOR_HEADER_TEXT.splitlines(keepends=True))
    for header, rows in mode_lines_and_rows:
        lines.append(header + "\n")
        lines.append("             X         Y         Z           dx          dy          dz\n")
        for k, (dx, dy, dz) in enumerate(rows):
            px, py, pz = float(k), 0.0, 0.0
            lines.append(f"      {px:.6f}  {py:.6f}  {pz:.6f}     {dx:.6f}  {dy:.6f}  {dz:.6f}\n")
        lines.append("\n")
    return "".join(lines)


def test_eigenvectors_parses_frequency_and_displacement_per_mode(tmp_path):
    text = _eigenvector_block([
        ("   1 f  =   12.345678 THz   77.567 2PiTHz  412.345 cm-1    51.123 meV",
         [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        ("   2 f/i=    1.234567 THz    7.756 2PiTHz   41.234 cm-1     5.112 meV",
         [[0.5, 0.5, 0.0], [-0.5, -0.5, 0.0]]),
    ])
    path = tmp_path / "OUTCAR"
    path.write_text(text)

    modes = parse_eigenvectors_from_outcar(path, n_atoms=2)
    assert len(modes) == 2

    freq0, evec0 = modes[0]
    assert freq0 == pytest.approx(51.123)
    assert np.allclose(evec0, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    freq1, evec1 = modes[1]
    assert freq1 == pytest.approx(-5.112)
    assert np.allclose(evec1, [[0.5, 0.5, 0.0], [-0.5, -0.5, 0.0]])


def test_eigenvectors_skips_truncated_mode_but_keeps_scanning(tmp_path):
    """A mode with fewer atom rows than n_atoms must be skipped, not crash the scan."""
    text = _eigenvector_block([
        ("   1 f  =   1.000000 THz 1.0 2PiTHz 1.0 cm-1 1.000000 meV",
         [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
    ])
    # Mode 2: header + label line only, no atom rows at all (fully truncated).
    text += (
        "   2 f  =   2.000000 THz 1.0 2PiTHz 1.0 cm-1 2.000000 meV\n"
        "             X         Y         Z           dx          dy          dz\n"
    )
    text += _eigenvector_block([
        ("   3 f  =   3.000000 THz 1.0 2PiTHz 1.0 cm-1 3.000000 meV",
         [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]),
    ]).replace(_EIGENVECTOR_HEADER_TEXT, "")  # avoid a second header block

    path = tmp_path / "OUTCAR"
    path.write_text(text)

    modes = parse_eigenvectors_from_outcar(path, n_atoms=2)
    freqs = [f for f, _ in modes]
    assert freqs == [pytest.approx(1.0), pytest.approx(3.0)]


def test_eigenvectors_no_anchor_returns_empty(tmp_path):
    path = tmp_path / "OUTCAR"
    path.write_text("no eigenvector section here\n")
    assert parse_eigenvectors_from_outcar(path, n_atoms=2) == []
