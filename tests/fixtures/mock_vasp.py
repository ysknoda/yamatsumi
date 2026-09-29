#!/usr/bin/env python3
"""
Mock VASP executable for yamatsumi integration testing.

Stands in for `vasp_std` (invoked via --vasp-cmd, cwd = the image
directory): reads ./POSCAR, evaluates an analytic 2D potential energy
surface at the first atom's (x, y) Cartesian coordinate, and writes
./OUTCAR in the exact text format yamatsumi.outcar's parsers expect (TOTEN
line, TOTAL-FORCE block, phonon frequency block with the real/imaginary
distinction carried by the 'f  =' vs 'f/i=' token, not a literal sign).

Deliberately does NOT import yamatsumi: this is an independent oracle (its
own Direct->Cartesian transform, its own minimal POSCAR reader), not a
test of yamatsumi against itself.

Usage: mock_vasp.py --pes {doublewell|mullerbrown}
"""
from __future__ import annotations

import argparse

import numpy as np


def read_minimal_poscar(path="POSCAR"):
    """
    Minimal reader for the exact VASP5/Direct/scale=1.0 format
    yamatsumi.poscar.write_poscar always emits (optionally with a
    'Selective dynamics' line, which this reader tolerates but ignores).
    """
    with open(path) as fh:
        lines = [ln.rstrip("\n") for ln in fh]
    scale = float(lines[1].split()[0])
    lattice = np.array(
        [[float(x) for x in lines[i].split()[:3]] for i in range(2, 5)]
    ) * scale

    idx = 5
    tok = lines[idx].split()
    if tok and tok[0].lstrip("-").replace(".", "", 1).isdigit():
        counts = [int(x) for x in tok]
        idx += 1
    else:
        idx += 1
        counts = [int(x) for x in lines[idx].split()]
        idx += 1
    n_atoms = sum(counts)

    if lines[idx].strip() and lines[idx].strip()[0].lower() == "s":
        idx += 1
    idx += 1  # coordinate-type line ("Direct")

    positions = np.array(
        [[float(x) for x in lines[idx + i].split()[:3]] for i in range(n_atoms)]
    )
    return lattice, positions, n_atoms


def double_well(x: float, y: float):
    """V(x,y) = (x^2-1)^2 + y^2 -- minima at (+-1,0), saddle at (0,0), V(saddle)=1."""
    v    = (x ** 2 - 1.0) ** 2 + y ** 2
    dvdx = 4.0 * x * (x ** 2 - 1.0)
    dvdy = 2.0 * y
    return v, dvdx, dvdy


_MB_A  = np.array([-200.0, -100.0, -170.0, 15.0])
_MB_a  = np.array([-1.0, -1.0, -6.5, 0.7])
_MB_b  = np.array([0.0, 0.0, 11.0, 0.6])
_MB_c  = np.array([-10.0, -10.0, -6.5, 0.7])
_MB_x0 = np.array([1.0, 0.0, -0.5, -1.0])
_MB_y0 = np.array([0.0, 0.5, 1.5, 1.0])


def muller_brown(x: float, y: float, rescale: float = 20.0):
    """
    Standard Muller-Brown 4-Gaussian surface, rescaled by /rescale.
    Rescaling is necessary (not cosmetic): at its natural scale the surface
    is far too stiff for yamatsumi's default step_size/spring_const to
    optimise stably (verified empirically -- sd plateaus, quickmin can
    diverge).
    """
    dx   = x - _MB_x0
    dy   = y - _MB_y0
    expo = _MB_a * dx ** 2 + _MB_b * dx * dy + _MB_c * dy ** 2
    terms = _MB_A * np.exp(expo)
    v    = np.sum(terms)
    dvdx = np.sum(terms * (2.0 * _MB_a * dx + _MB_b * dy))
    dvdy = np.sum(terms * (_MB_b * dx + 2.0 * _MB_c * dy))
    return v / rescale, dvdx / rescale, dvdy / rescale


PES_FUNCS = {"doublewell": double_well, "mullerbrown": muller_brown}


def hessian_2d(func, x: float, y: float, h: float = 1e-4):
    """Central finite-difference Hessian, built from the analytic gradient."""
    _, gx_px, gy_px = func(x + h, y)
    _, gx_mx, gy_mx = func(x - h, y)
    _, _,     gy_py = func(x, y + h)
    _, _,     gy_my = func(x, y - h)
    hxx = (gx_px - gx_mx) / (2 * h)
    hyy = (gy_py - gy_my) / (2 * h)
    hxy = (gy_px - gy_mx) / (2 * h)
    return np.array([[hxx, hxy], [hxy, hyy]])


def write_outcar(path, v: float, atom_forces, eigenvalues, eigenvectors, n_atoms):
    """
    eigenvectors: (2, n_modes) array from np.linalg.eigh(hessian_2d) --
    column j is the 2D (x,y) eigenvector for eigenvalues[j]. Only atom 0
    carries real 2D-PES physics, so its (dx, dy) displacement is that
    eigenvector (dz=0); every other atom is a fixed spectator with an
    all-zero displacement row, mirroring how only atom 0 gets a nonzero
    force above.

    The per-mode "X Y Z dx dy dz" column-label line and one displacement
    row per atom mirror real VASP's IBRION=5 eigenvector block layout, so
    that yamatsumi.outcar.parse_eigenvectors_from_outcar (and the anchor in
    parse_frequencies_from_outcar) are exercised against a realistically
    shaped OUTCAR, not just the frequency header lines alone.
    """
    lines = [
        "   NELM   =     60;   NELMIN=  2; NELMDL= -5     # of ELM steps \n",
        (
            "------------------------ aborting loop because EDIFF is reached "
            "----------------------------------------\n"
        ),
        f"  free  energy   TOTEN  =   {v:.6f} eV\n",
        "     LOOP+:  cpu time      0.0100: real time      0.0100\n",
        "\n",
        " POSITION                                       TOTAL-FORCE (eV/Angst)\n",
        " -----------------------------------------------------------------------------------\n",
    ]
    for (px, py, pz), (fx, fy, fz) in atom_forces:
        lines.append(f"   {px:.6f} {py:.6f} {pz:.6f}   {fx:.6f} {fy:.6f} {fz:.6f}\n")
    lines += [
        " -----------------------------------------------------------------------------------\n",
        "\n",
        " Eigenvectors and eigenvalues of the dynamical matrix\n",
        " ----------------------------------------------------\n",
        "\n",
    ]
    for i, ev in enumerate(eigenvalues, start=1):
        is_imag = ev < 0.0
        mev = abs(ev) * 10.0  # deterministic (not physically calibrated) meV scale
        tag = "f/i" if is_imag else "f  "
        lines.append(
            f"   {i} {tag}=   {mev:.6f} THz   {mev * 6.283:.6f} 2PiTHz  "
            f"{mev * 8.066:.6f} cm-1    {mev:.6f} meV\n"
        )
        lines.append("             X         Y         Z           dx          dy          dz\n")
        mode_dx, mode_dy = float(eigenvectors[0, i - 1]), float(eigenvectors[1, i - 1])
        for atom_idx in range(n_atoms):
            px, py, pz = atom_forces[atom_idx][0]
            dx, dy, dz = (mode_dx, mode_dy, 0.0) if atom_idx == 0 else (0.0, 0.0, 0.0)
            lines.append(
                f"      {px:.6f}  {py:.6f}  {pz:.6f}     {dx:.6f}  {dy:.6f}  {dz:.6f}\n"
            )
        lines.append("\n")
    with open(path, "w") as fh:
        fh.writelines(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pes", choices=sorted(PES_FUNCS), required=True)
    parser.add_argument("--rescale", type=float, default=20.0)
    args = parser.parse_args()

    lattice, positions, n_atoms = read_minimal_poscar("POSCAR")
    cart = positions @ lattice
    x, y = float(cart[0, 0]), float(cart[0, 1])

    if args.pes == "mullerbrown":
        v, dvdx, dvdy = muller_brown(x, y, rescale=args.rescale)
        hess = hessian_2d(lambda a, b: muller_brown(a, b, rescale=args.rescale), x, y)
    else:
        v, dvdx, dvdy = double_well(x, y)
        hess = hessian_2d(double_well, x, y)

    eigenvalues, eigenvectors = np.linalg.eigh(hess)

    atom_forces = [((cart[0, 0], cart[0, 1], cart[0, 2]), (-dvdx, -dvdy, 0.0))]
    for i in range(1, n_atoms):
        atom_forces.append(((cart[i, 0], cart[i, 1], cart[i, 2]), (0.0, 0.0, 0.0)))

    write_outcar("OUTCAR", v, atom_forces, eigenvalues, eigenvectors, n_atoms)


if __name__ == "__main__":
    main()
