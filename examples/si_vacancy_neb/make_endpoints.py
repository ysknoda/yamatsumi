#!/usr/bin/env python3
"""
Generate CI-NEB endpoint POSCARs for a single-vacancy nearest-neighbour hop
in bulk diamond-cubic silicon, for use with the `yamatsumi` CI-NEB package.

Physical setup
--------------
Silicon crystallises in the diamond-cubic structure (space group Fd-3m,
atoms on the 8a Wyckoff site). The experimental lattice constant used here,
a = 5.431 Ang at 300 K, is from the Ioffe Institute NSM Archive
(https://www.ioffe.ru/SVA/NSM/Semicond/Si/basic.html); it agrees with the
commonly cited textbook value (e.g. Kittel, "Introduction to Solid State
Physics"). This script tiles the conventional 8-atom diamond-cubic cell
into an n x n x n supercell (default n=2 -> 64 atoms, a "Si64" cell -- a
standard, if modest by modern standards, supercell size used for
point-defect methodology testing in the literature; production formation-
/migration-energy numbers should be checked against a larger cell, e.g.
216 or 512 atoms, for finite-size convergence), removes one atom to create
a vacancy, and writes two endpoint POSCARs describing a single
nearest-neighbour vacancy hop:

  POSCAR_init : vacancy at site A.
  POSCAR_final: vacancy at site B, where B is one of site A's four
                <111>-direction tetrahedral nearest neighbours (ideal,
                unrelaxed Si-Si bond length a*sqrt(3)/4 = 2.352 Ang -- the
                well-known Si-Si bond length). The atom that occupied B in
                POSCAR_init has moved to A; every other atom is untouched.

Both endpoints keep the SAME atom ordering (the migrating atom sits at the
same array index in both files, only its position differs) because
yamatsumi's NEB interpolation/force engine acts per atom by array index,
not by nearest-neighbour matching between the two files.

IMPORTANT: these are ideal, UNRELAXED "bulk lattice minus one atom"
structures. Real interatomic forces here are of order eV/Ang, not the
sub-0.05 eV/Ang scale a CI-NEB run expects of its endpoints. Relax
POSCAR_init and POSCAR_final SEPARATELY with ordinary static DFT (ISIF=2,
fixed cell, tight EDIFFG) before handing them to `yamatsumi` -- see the
accompanying README.md for the exact next steps.
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

# Allow running this script directly (``python examples/.../make_endpoints.py``)
# without first installing yamatsumi, by adding the repository root to sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from yamatsumi.interpolate import check_endpoint_consistency  # noqa: E402
from yamatsumi.poscar import Poscar, read_poscar, write_poscar  # noqa: E402

# --- Physical constants (cited in the module docstring above) ---------
SI_LATTICE_CONST_ANG = 5.431
# Diamond-cubic conventional cell, 8 atoms at Wyckoff 8a of space group
# Fd-3m (two interpenetrating FCC sublattices offset by (1/4, 1/4, 1/4)),
# fractional coordinates of the conventional cubic cell.
_DIAMOND_BASIS_FRAC = np.array([
    [0.00, 0.00, 0.00], [0.00, 0.50, 0.50],
    [0.50, 0.00, 0.50], [0.50, 0.50, 0.00],
    [0.25, 0.25, 0.25], [0.25, 0.75, 0.75],
    [0.75, 0.25, 0.75], [0.75, 0.75, 0.25],
])


def build_supercell_positions(n: int) -> np.ndarray:
    """
    Fractional coordinates (of the n x n x n SUPERCELL, not the conventional
    cell) of every diamond-cubic lattice site, shape (8 * n**3, 3).
    """
    sites = []
    for i, j, k in itertools.product(range(n), repeat=3):
        shift = np.array([i, j, k], dtype=float)
        sites.append((_DIAMOND_BASIS_FRAC + shift) / n)
    return np.concatenate(sites, axis=0)


def nearest_neighbor(positions_frac: np.ndarray, lattice: np.ndarray, site: int) -> tuple[int, float]:
    """
    Index of `site`'s nearest OTHER lattice site under the minimum-image
    convention, and the true (MIC) Cartesian distance to it. Mirrors
    yamatsumi.geometry.mic_displacement's own "d - round(d)" convention so
    this distance is directly comparable to what yamatsumi itself will see.
    """
    d_frac = positions_frac - positions_frac[site]
    d_frac -= np.round(d_frac)
    d_cart = d_frac @ lattice
    dist = np.linalg.norm(d_cart, axis=1)
    dist[site] = np.inf
    j = int(np.argmin(dist))
    return j, float(dist[j])


def build_endpoints(
    supercell_size: int,
    lattice_const: float,
    element: str,
    vacancy_site: int | None,
) -> tuple[Poscar, Poscar, int, int, float]:
    """
    Returns (poscar_init, poscar_final, site_a, site_b, nn_distance_ang).
    """
    lattice = np.eye(3) * (lattice_const * supercell_size)
    ideal_positions = build_supercell_positions(supercell_size)
    n_sites = ideal_positions.shape[0]

    if vacancy_site is None:
        # Pick the site closest to the cell centre purely so the generated
        # structure is easy to eyeball in a viewer; under periodic boundary
        # conditions every site and every one of its 4 tetrahedral bonds is
        # symmetry-equivalent, so this choice has no physical significance.
        center = np.array([0.5, 0.5, 0.5])
        d = ideal_positions - center
        d -= np.round(d)
        vacancy_site = int(np.argmin(np.linalg.norm(d @ lattice, axis=1)))

    site_b, nn_distance = nearest_neighbor(ideal_positions, lattice, vacancy_site)
    site_a = vacancy_site

    # --- Initial state: vacancy at A; the atom that ideally sits at B is
    # still there. ---
    positions_init = np.delete(ideal_positions, site_a, axis=0)

    # --- Final state: same 63-atom array (same site deleted, same atom
    # ordering), except the atom occupying the slot that used to be site B
    # has moved to site A's position -- i.e. it hopped into the vacancy,
    # leaving a new vacancy behind at B. ---
    positions_final = positions_init.copy()
    # `site_b`'s row index shifts down by one once `site_a` is deleted, but
    # only if site_a < site_b; handle both orderings explicitly rather than
    # relying on that.
    b_row = site_b - 1 if site_b > site_a else site_b
    positions_final[b_row] = ideal_positions[site_a]

    n_atoms = n_sites - 1
    common = dict(
        scale=1.0, lattice=lattice, species=[element], counts=[n_atoms],
        coord_type="Direct",
    )
    poscar_init = Poscar(
        comment=f"{element}{n_sites} vacancy hop: init (vacancy at site {site_a})",
        positions=positions_init, **common,
    )
    poscar_final = Poscar(
        comment=f"{element}{n_sites} vacancy hop: final (vacancy at site {site_b})",
        positions=positions_final, **common,
    )
    return poscar_init, poscar_final, site_a, site_b, nn_distance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--supercell-size", type=int, default=2,
                        help="n for the n x n x n supercell of the 8-atom conventional cell "
                             "(default: 2 -> 64 sites, 63 atoms after the vacancy)")
    parser.add_argument("--lattice-constant", type=float, default=None,
                        help=f"Conventional-cell lattice constant in Ang. Defaults to "
                             f"{SI_LATTICE_CONST_ANG} (experimental Si at 300 K, Ioffe NSM "
                             "archive) for --element Si, and is required for any other "
                             "element.")
    parser.add_argument("--element", default="Si",
                        help="Species label written to the POSCAR (default: Si). For another "
                             "diamond-cubic element, also give its --lattice-constant.")
    parser.add_argument("--vacancy-site", type=int, default=None,
                        help="Lattice-site index (0-based, before vacancy removal) to vacate. "
                             "Default: the site nearest the supercell centre (physically "
                             "arbitrary under PBC, just convenient to look at).")
    parser.add_argument("--output-dir", default=str(Path(__file__).parent),
                        help="Directory to write POSCAR_init / POSCAR_final into "
                             "(default: this script's own directory)")
    args = parser.parse_args()

    # --- Input validation: refuse anything that would write a wrong pair ---
    if args.supercell_size < 1:
        parser.error(f"--supercell-size must be at least 1 (got {args.supercell_size}).")
    if args.lattice_constant is None:
        if args.element != "Si":
            parser.error(
                f"--element {args.element} needs its own --lattice-constant; the "
                f"default {SI_LATTICE_CONST_ANG} Ang is silicon's and would give "
                f"{args.element}-{args.element} bonds of silicon's length."
            )
        args.lattice_constant = SI_LATTICE_CONST_ANG
    if args.lattice_constant <= 0.0:
        parser.error(f"--lattice-constant must be positive (got {args.lattice_constant}).")
    n_sites = 8 * args.supercell_size ** 3
    if args.vacancy_site is not None and not 0 <= args.vacancy_site < n_sites:
        parser.error(
            f"--vacancy-site must be a site index from 0 to {n_sites - 1} for a "
            f"{args.supercell_size}x{args.supercell_size}x{args.supercell_size} supercell "
            f"(got {args.vacancy_site}); negative Python-style indices are not accepted."
        )

    poscar_init, poscar_final, site_a, site_b, nn_distance = build_endpoints(
        args.supercell_size, args.lattice_constant, args.element, args.vacancy_site,
    )

    # --- Verification before writing anything out --------------------
    # 1. Measured nearest-neighbour distance vs. the analytic diamond-cubic
    #    value a*sqrt(3)/4, independently derived from the basis-vector
    #    geometry, not from this script's own construction logic.
    expected_nn = args.lattice_constant * np.sqrt(3.0) / 4.0
    print(f"Supercell: {args.supercell_size}x{args.supercell_size}x{args.supercell_size} "
          f"of the 8-atom conventional cell -> {poscar_init.n_atoms + 1} sites, "
          f"{poscar_init.n_atoms} atoms after the vacancy.")
    print(f"Lattice constant a = {args.lattice_constant:.4f} Ang "
          f"-> supercell edge = {args.lattice_constant * args.supercell_size:.4f} Ang.")
    print(f"Vacancy hop: site {site_a} (init vacancy) <-> site {site_b} (final vacancy).")
    print(f"  Ideal nearest-neighbour distance (measured, MIC) : {nn_distance:.6f} Ang")
    print(f"  Ideal nearest-neighbour distance (analytic a*sqrt(3)/4): {expected_nn:.6f} Ang")
    if abs(nn_distance - expected_nn) > 1e-6:
        raise SystemExit(
            f"FAILED verification: measured NN distance {nn_distance:.6f} Ang does not "
            f"match the analytic diamond-cubic value {expected_nn:.6f} Ang -- the two "
            "chosen sites are not actually a bonded nearest-neighbour pair. Aborting "
            "without writing any output."
        )
    print("  -> matches analytic value: OK")

    # 2. The migrating atom must move by exactly the nearest-neighbour
    #    distance between the two endpoints, and every other atom must be
    #    perfectly stationary (both are supposed to be true by construction;
    #    checked here directly against the two Poscar objects rather than
    #    trusted).
    from yamatsumi.geometry import direct_to_cartesian, mic_displacement
    d_frac = mic_displacement(poscar_init.positions, poscar_final.positions)
    d_cart = direct_to_cartesian(d_frac, poscar_init.lattice)
    per_atom_disp = np.linalg.norm(d_cart, axis=1)
    n_moved = int(np.sum(per_atom_disp > 1e-9))
    if n_moved != 1:
        raise SystemExit(
            f"FAILED verification: expected exactly 1 atom to differ between "
            f"POSCAR_init and POSCAR_final, found {n_moved}. Aborting without "
            "writing any output."
        )
    moved_distance = float(per_atom_disp.max())
    print(f"  Atoms that differ between init/final: {n_moved} "
          f"(displacement {moved_distance:.6f} Ang, expected {nn_distance:.6f} Ang)")
    if abs(moved_distance - nn_distance) > 1e-6:
        raise SystemExit(
            f"FAILED verification: the migrating atom moves {moved_distance:.6f} Ang, "
            f"not the nearest-neighbour distance {nn_distance:.6f} Ang, so this is not "
            "a nearest-neighbour vacancy hop. Aborting without writing any output."
        )
    print("  -> migrating atom moves exactly one nearest-neighbour distance: OK")

    # 3. yamatsumi's own consistency check (species/counts/lattice/atom
    #    count) -- this is exactly what `phase_init` runs before
    #    interpolating, so failing it here means failing it there too.
    check_endpoint_consistency(poscar_init, poscar_final)
    print("  yamatsumi.interpolate.check_endpoint_consistency: OK")

    # --- Write output, then round-trip through yamatsumi's own reader to
    # confirm the files this script just wrote are byte-for-byte usable. ---
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    init_path = out_dir / "POSCAR_init"
    final_path = out_dir / "POSCAR_final"
    write_poscar(str(init_path), poscar_init)
    write_poscar(str(final_path), poscar_final)

    reread_init = read_poscar(str(init_path))
    reread_final = read_poscar(str(final_path))
    assert reread_init == poscar_init, "round-trip mismatch on POSCAR_init"
    assert reread_final == poscar_final, "round-trip mismatch on POSCAR_final"
    print("  round-trip read_poscar(write_poscar(...)) == original: OK")

    print(f"\nWrote {init_path}")
    print(f"Wrote {final_path}")


if __name__ == "__main__":
    main()
