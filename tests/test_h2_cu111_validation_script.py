"""
Tests for examples/h2_cu111_vasp/run_validation.py that need no VASP: the
slab and adsorbate geometry, the Birch-Murnaghan fit, the setting choice of
ADR 0009 and the image perturbation.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from yamatsumi.geometry import direct_to_cartesian, mic_displacement
from yamatsumi.poscar import read_poscar, write_poscar

SCRIPT = (
    Path(__file__).parent.parent / "examples" / "h2_cu111_vasp" / "run_validation.py"
)
A = 3.624  # SBH17 Table 3, PBE


@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location("h2cu_run_validation", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _min_cu_cu(p):
    cart = direct_to_cartesian(p.positions, p.lattice)
    best = np.inf
    for i in range(len(cart)):
        d = direct_to_cartesian(
            mic_displacement(p.positions[i], p.positions), p.lattice
        )
        n = np.linalg.norm(d, axis=1)
        best = min(best, n[n > 1e-6].min())
    return best


def test_slab_has_fcc_nearest_neighbour_distance_and_layer_spacing(script):
    slab = script.slab_1x1(A)
    assert _min_cu_cu(slab) == pytest.approx(A / np.sqrt(2.0), abs=1e-9)  # 2.5626 Ang
    z = np.sort(direct_to_cartesian(slab.positions, slab.lattice)[:, 2])
    assert np.diff(z) == pytest.approx([A / np.sqrt(3.0)] * 3, abs=1e-9)  # 2.0923 Ang
    assert slab.lattice[2, 2] - (z[-1] - z[0]) == pytest.approx(script.VACUUM)
    assert not slab.selective_dynamics[0].any() and slab.selective_dynamics[1:].all()


def test_supercell_keeps_bulk_distances(script):
    slab2 = script.supercell(script.slab_1x1(A), 2)
    assert slab2.n_atoms == 16
    assert _min_cu_cu(slab2) == pytest.approx(A / np.sqrt(2.0), abs=1e-9)


def test_hollows_sit_next_to_the_bridge_and_hcp_is_over_the_second_layer(script):
    slab2 = script.supercell(script.slab_1x1(A), 2)
    site = script.surface_sites(slab2)
    s = A / np.sqrt(2.0)
    assert site["nn"] == pytest.approx(s)
    for hollow in ("fcc", "hcp"):
        assert np.linalg.norm(site[hollow] - site["bridge"]) == pytest.approx(
            s * np.sqrt(3) / 6
        )
    # every hollow is s/sqrt(3) = 1.4796 Ang from its three top-layer atoms
    cart = direct_to_cartesian(slab2.positions, slab2.lattice)
    top = cart[np.isclose(cart[:, 2], site["top_z"])]
    for hollow in ("fcc", "hcp"):
        lat = np.sort(
            [
                np.linalg.norm(
                    direct_to_cartesian(
                        mic_displacement(
                            np.linalg.solve(slab2.lattice.T, t),
                            np.linalg.solve(slab2.lattice.T, site[hollow]),
                        ),
                        slab2.lattice,
                    )[:2]
                )
                for t in top
            ]
        )
        assert lat[:3] == pytest.approx([s / np.sqrt(3)] * 3, abs=1e-9)


def test_h2_over_bridge_has_the_requested_geometry(script):
    slab2 = script.supercell(script.slab_1x1(A), 2)
    ts = script.h2_over_bridge(slab2, 1.16, 1.03)
    g = script.h_geometry(ts, slab2)
    assert g["r_HH"] == pytest.approx(1.03)
    assert g["Z"] == pytest.approx(1.16)
    assert g["lateral_from_bridge"] == pytest.approx(0.0, abs=1e-9)
    assert g["tilt_deg"] == pytest.approx(0.0, abs=1e-9)
    assert ts.species == ["H", "Cu"] and ts.counts == [2, 16]
    assert ts.selective_dynamics[:2].all() and not ts.selective_dynamics[2:].any()


def test_final_state_keeps_the_atom_order_of_the_initial_one(script):
    slab2 = script.supercell(script.slab_1x1(A), 2)
    init = script.h2_over_bridge(slab2, 4.0, 0.75)
    final = script.two_h_in_hollows(slab2, 0.95)
    ci = direct_to_cartesian(init.positions[:2], init.lattice)
    cf = direct_to_cartesian(final.positions[:2], final.lattice)
    ey = script.surface_sites(slab2)["ey"]
    assert (ci[1] - ci[0]) @ ey > 0 and (cf[1] - cf[0]) @ ey > 0
    assert np.linalg.norm(cf[1] - cf[0]) == pytest.approx(A / np.sqrt(2) / np.sqrt(3))
    assert np.array_equal(init.positions[2:], final.positions[2:])


def test_birch_murnaghan_recovers_a_known_minimum(script):
    a0, b0, bp, e0 = 3.63, 0.87, 5.0, -3.7  # eV/Ang^3 bulk modulus, as for Cu
    v0 = a0**3 / 4

    def energy(a):
        eta = (v0 / (a**3 / 4)) ** (2 / 3)
        return e0 + 9 * v0 * b0 / 16 * (
            (eta - 1) ** 3 * bp + (eta - 1) ** 2 * (6 - 4 * eta)
        )

    grid = [3.56 + 0.02 * i for i in range(7)]
    assert script.birch_murnaghan_a0(grid, [energy(a) for a in grid]) == pytest.approx(
        a0, abs=1e-6
    )


def test_setting_choice_takes_the_cheapest_within_tolerance(script):
    barriers = {
        (6, 350): 0.520,
        (8, 350): 0.505,
        (11, 350): 0.502,
        (6, 450): 0.515,
        (8, 450): 0.498,
        (11, 450): 0.500,
    }
    chosen, dense = script.select_setting(barriers)
    assert dense == (11, 450)
    assert chosen == (8, 350)  # 5 meV off; (6, *) are 15-20 meV off


def test_dipole_jump_lies_between_the_far_h2_and_the_slab_image(script):
    slab2 = script.supercell(script.slab_1x1(A), 2)
    far = script.h2_over_bridge(slab2, 6.0, 0.75)
    c = far.lattice[2, 2]
    z = direct_to_cartesian(far.positions, far.lattice)[:, 2]
    jump = (script.dipol_centre(far, z[:2].mean()) * c + 0.5 * c) % c
    top = z[2:].max()
    assert z[:2].mean() < jump < z[2:].min() + c
    assert jump - z[:2].mean() > 3.0 and top < z[:2].mean()


def test_perturbation_moves_only_the_h_atoms(script, tmp_path):
    slab2 = script.supercell(script.slab_1x1(A), 2)
    base = script.h2_over_bridge(slab2, 2.0, 0.9)
    for i in range(script.NUM_IMAGES + 2):
        (tmp_path / f"{i:02d}").mkdir()
        write_poscar(str(tmp_path / f"{i:02d}" / "POSCAR"), base)
    script.perturb_h(tmp_path, 0.05, 1)
    for i in range(1, script.NUM_IMAGES + 1):
        p = read_poscar(str(tmp_path / f"{i:02d}" / "POSCAR"))
        d = direct_to_cartesian(
            mic_displacement(base.positions, p.positions), p.lattice
        )
        n = np.linalg.norm(d, axis=1)
        assert np.allclose(n[2:], 0.0, atol=1e-7)
        assert n[:2].max() == pytest.approx(0.05, abs=1e-6) and n[:2].min() > 0
