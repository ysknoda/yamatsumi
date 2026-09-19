"""
Integration tests: drive the full yamatsumi pipeline (phase_init + phase_neb)
against the mock-VASP analytic potential energy surfaces (tests/fixtures/
mock_vasp.py), with no real VASP involved. These validate the actual NEB
algorithm end-to-end (does the climbing image converge to the known
analytic saddle energy?), not just code plumbing.

Both surfaces here use a simple cubic coordinate-encoding lattice, so
neither exercises Bug A (the sheared-lattice transpose bug) -- that is
covered exclusively by the dedicated sheared-lattice unit tests in
test_geometry.py / test_poscar.py. A fully green run of this file is not,
by itself, evidence that Bug A stays fixed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from yamatsumi.cli import build_parser
from yamatsumi.directories import get_image_dirs
from yamatsumi.outcar import parse_energy_from_outcar
from yamatsumi.phases.init_phase import phase_init
from yamatsumi.phases.neb_phase import phase_neb
from yamatsumi.poscar import Poscar, read_poscar, write_poscar
from yamatsumi.vasp_runner import run_vasp

MOCK_VASP = str(Path(__file__).parent / "fixtures" / "mock_vasp.py")

# Ang; large relative to the PES's O(1)-O(2) coordinate range so
# _mic_displacement's unconditional d - round(d) never mistakes a real
# displacement for a periodic-boundary crossing (a genuine trap found while
# designing this fixture, not a hypothetical one).
LATTICE_SIZE = 20.0


def _make_poscar(x: float, y: float, comment: str) -> Poscar:
    lattice = np.eye(3) * LATTICE_SIZE
    positions = np.array([[x / LATTICE_SIZE, y / LATTICE_SIZE, 0.5]])
    return Poscar(
        comment=comment, scale=1.0, lattice=lattice, species=["H"], counts=[1],
        coord_type="Direct", positions=positions,
    )


def _cli_args(tmp_path, vasp_cmd, x0, y0, x1, y1, **overrides):
    work_dir = tmp_path / "NEB_run"
    copy_dir = tmp_path / "CopyDir"
    copy_dir.mkdir()
    (copy_dir / "INCAR").write_text("# placeholder -- ignored by mock_vasp.py\n")

    init_path  = tmp_path / "POSCAR_init"
    final_path = tmp_path / "POSCAR_final"
    write_poscar(str(init_path), _make_poscar(x0, y0, "initial"))
    write_poscar(str(final_path), _make_poscar(x1, y1, "final"))

    argv = [
        "--init", str(init_path), "--final", str(final_path),
        "--copy-dir", str(copy_dir), "--work-dir", str(work_dir),
        "--vasp-cmd", vasp_cmd, "--no-show",
    ]
    for key, value in overrides.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                argv.append(flag)
        else:
            argv.extend([flag, str(value)])
    return build_parser().parse_args(argv)


def _perturb_intermediate_y(work_dir, num_images, vasp_cmd, delta_cart_y):
    """Nudge every intermediate image's y-coordinate by delta_cart_y (Ang)."""
    for idir in get_image_dirs(work_dir, num_images)[1:-1]:
        p = read_poscar(str(idir / "POSCAR"))
        cart_y = p.positions[0, 1] * LATTICE_SIZE + delta_cart_y
        p.positions[0, 1] = cart_y / LATTICE_SIZE
        write_poscar(str(idir / "POSCAR"), p)
        assert run_vasp(idir, vasp_cmd)


def test_double_well_yamatsumi_converges_to_known_saddle(tmp_path):
    """
    V(x,y)=(x^2-1)^2+y^2: exact minima (+-1,0), exact saddle (0,0), V=1.
    Symmetric endpoints would place the linear-interpolation midpoint
    exactly at the saddle already, "converging" without exercising the
    optimiser at all -- intermediate images are perturbed in y first to
    avoid that degenerate setup.
    """
    vasp_cmd = f"{sys.executable} {MOCK_VASP} --pes doublewell"
    args = _cli_args(
        tmp_path, vasp_cmd, x0=-1.0, y0=0.0, x1=1.0, y1=0.0,
        num_images=5, optimizer="fire", spring_const=5.0, step_size=0.05,
        force_tol=0.05, max_iter=100,
    )

    phase_init(args)
    _perturb_intermediate_y(Path(args.work_dir), args.num_images, vasp_cmd, delta_cart_y=0.3)
    phase_neb(args)

    image_dirs = get_image_dirs(Path(args.work_dir), args.num_images)
    energies = [parse_energy_from_outcar(idir / "OUTCAR") for idir in image_dirs]
    ci_index = int(np.argmax(energies[1:-1])) + 1

    assert energies[ci_index] == pytest.approx(1.0, abs=1e-2)

    result_file = Path(args.work_dir) / "yamatsumi_result.txt"
    assert result_file.exists()
    result_text = result_file.read_text()
    assert "TS criterion satisfied      : True" in result_text

    # 1d end-to-end: at this analytic saddle the Hessian is diag(-4, 2), so
    # the imaginary mode's unstable direction is purely along x -- and the
    # NEB path (endpoints (-1,0)/(1,0), only the y-perturbation is
    # off-axis) approaches the saddle along x too, so a correctly computed
    # alignment should land at (or extremely close to) 1.0.
    assert "Eigenvector-tangent alignment" in result_text
    align_line = next(
        ln for ln in result_text.splitlines() if "Eigenvector-tangent alignment" in ln
    )
    alignment = float(align_line.split(":")[1].split()[0])
    assert alignment == pytest.approx(1.0, abs=1e-3)


@pytest.mark.slow
def test_muller_brown_yamatsumi_converges_to_known_saddle(tmp_path):
    """
    Standard 4-Gaussian Muller-Brown surface (rescaled by /20 -- required
    for stable convergence with yamatsumi's default hyperparameters, verified
    empirically). Endpoints are the min3<->min2 pair (a single clean
    saddle), not the full 3-minimum min1<->min2 chain.
    """
    vasp_cmd = f"{sys.executable} {MOCK_VASP} --pes mullerbrown --rescale 20.0"
    args = _cli_args(
        tmp_path, vasp_cmd,
        x0=-0.050011, y0=0.466694,  # min3
        x1=0.623499,  y1=0.028038,  # min2
        num_images=7, optimizer="fire", spring_const=5.0, step_size=0.05,
        force_tol=0.05, max_iter=200,
    )

    phase_init(args)
    phase_neb(args)

    image_dirs = get_image_dirs(Path(args.work_dir), args.num_images)
    energies = [parse_energy_from_outcar(idir / "OUTCAR") for idir in image_dirs]
    ci_index = int(np.argmax(energies[1:-1])) + 1

    expected_saddle = -72.248940 / 20.0
    assert energies[ci_index] == pytest.approx(expected_saddle, abs=5e-2)
