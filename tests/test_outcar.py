"""
Tests for yamatsumi.outcar.

The frequency-token format asserted here (real: 'f  =', imaginary: 'f/i=',
sign carried by the token, not by a literal minus sign in the meV column)
is also the format the mock VASP fixture (tests/fixtures/mock_vasp.py) must
reproduce for the integration tests to be meaningful.
"""
from __future__ import annotations

from pathlib import Path

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


# ---------------------------------------------------------------------------
# ADR 0003: SCF convergence detection and ionic-summary-only energies.
# The marker lines below are copied verbatim from real VASP 6.6.1 OUTCARs
# (Si2, converged and with NELM = 3); the same markers appear in 6.4.3.
# ---------------------------------------------------------------------------

_NELM_60 = "   NELM   =     60;   NELMIN=  2; NELMDL= -5     # of ELM steps \n"
_NELM_3  = "   NELM   =      3;   NELMIN=  2; NELMDL=  0     # of ELM steps \n"
_EDIFF_REACHED = (
    "------------------------ aborting loop because EDIFF is reached "
    "----------------------------------------\n"
)
_LOOP_PLUS = "     LOOP+:  cpu time      2.5090: real time      1.9252\n"


def _elec(e):
    return f"  free energy    TOTEN  =       {e:.8f} eV\n"


def _summary(e):
    return f"  free  energy   TOTEN  =       {e:.8f} eV\n"


def _converged_step(e):
    return [_elec(e + 0.1), _elec(e), _EDIFF_REACHED, _summary(e), _LOOP_PLUS]


def test_scf_converged_single_point_has_no_problem(tmp_path):
    from yamatsumi.outcar import scf_convergence_problem
    p = tmp_path / "OUTCAR"
    p.write_text("".join([_NELM_60, *_converged_step(-10.81465686)]))
    assert scf_convergence_problem(p) is None


def test_scf_unconverged_single_point_is_reported_with_nelm(tmp_path):
    """VASP 6.6.1 exits 0 here and prints no marker (checked with NELM = 3)."""
    from yamatsumi.outcar import scf_convergence_problem
    p = tmp_path / "OUTCAR"
    p.write_text("".join([_NELM_3, _elec(2.4), _elec(-10.83), _elec(-10.9977),
                          _summary(-10.9977), _LOOP_PLUS]))
    problem = scf_convergence_problem(p)
    assert problem is not None
    assert "NELM = 3" in problem


def test_scf_one_unconverged_step_among_several_is_reported(tmp_path):
    """A finite-difference phonon run has one SCF per displacement."""
    from yamatsumi.outcar import scf_convergence_problem
    p = tmp_path / "OUTCAR"
    unconverged = [_elec(-9.0), _summary(-9.0), _LOOP_PLUS]
    p.write_text("".join([_NELM_60, *_converged_step(-10.0), *unconverged,
                          *_converged_step(-10.1)]))
    problem = scf_convergence_problem(p)
    assert problem is not None
    assert "1 of 3" in problem


def test_scf_outcar_without_any_ionic_step_is_reported(tmp_path):
    from yamatsumi.outcar import scf_convergence_problem
    p = tmp_path / "OUTCAR"
    p.write_text(_NELM_60 + _elec(-9.0))
    assert scf_convergence_problem(p) is not None


def test_scf_missing_outcar_is_reported(tmp_path):
    from yamatsumi.outcar import scf_convergence_problem
    assert scf_convergence_problem(tmp_path / "OUTCAR") is not None


def test_energy_ignores_trailing_electronic_step_lines(tmp_path):
    """
    A run stopped part-way through the next ionic step leaves per-electronic-
    step TOTEN lines after the last completed step's summary. Only the
    summary line is a converged ionic-step energy.
    """
    p = tmp_path / "OUTCAR"
    p.write_text("".join([*_converged_step(-45.97327637), _elec(-46.85957269)]))
    assert parse_energy_from_outcar(p) == pytest.approx(-45.97327637)


# ---------------------------------------------------------------------------
# The phonon parsers must stop at the end of the mode list
# ---------------------------------------------------------------------------

_REAL_PHONON_EXCERPT = (
    Path(__file__).parent / "fixtures" / "real_vasp_excerpts"
    / "si8_bulk_ibrion6_vasp643_phonon.OUTCAR"
)
# Verbatim band-eigenvalue rows from the same real OUTCAR: an integer, then
# two floats, which the old filters accepted as mode headers.
_BAND_ROWS = (
    "  band No.  band energies     occupation \n"
    "      1      -6.1584      2.00000\n"
    "      2       5.9416      2.00000\n"
)


def test_frequencies_ignore_band_rows_after_the_phonon_block(tmp_path):
    p = tmp_path / "OUTCAR"
    p.write_text(_REAL_PHONON_EXCERPT.read_text() + _BAND_ROWS)
    assert len(parse_frequencies_from_outcar(p)) == 24


def test_frequencies_stop_when_a_second_mode_list_starts(tmp_path):
    """
    A second eigenvector block (e.g. 'Eigenvectors after division by
    SQRT(mass)') restarts the mode numbering at 1 and must not be appended.
    """
    block = _REAL_PHONON_EXCERPT.read_text()
    second = block.replace("Eigenvectors and eigenvalues of the dynamical matrix",
                           "Eigenvectors after division by SQRT(mass)")
    p = tmp_path / "OUTCAR"
    p.write_text(block + second)
    assert len(parse_frequencies_from_outcar(p)) == 24
    assert len(parse_eigenvectors_from_outcar(p, 8)) == 24
