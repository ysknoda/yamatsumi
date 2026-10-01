#!/usr/bin/env python3
"""
Stage (b) real-VASP validation of yamatsumi: dissociative chemisorption of
H2 on Cu(111), compared with the SBH17 benchmark (Tchakoua, Gerrits,
Smeets, Kroes, J. Chem. Theory Comput. 2023, 19, 245-270). Every choice
below is recorded, with its source, in
docs/adr/0009-stage-b-validates-against-sbh17-h2-cu111-with-a-frozen-surface.md.

The run has two parts, so that the setting scan can be reviewed (and the
36-hour budget rule applied) before the expensive NEB starts:

  --part prep   bulk lattice constant, H2 bond length, 1x1 slab relaxation,
                the medium-algorithm scan over k-points and cutoff, the
                3x3x4 cell check, the H2-height check and the dipole check.
                Writes prep_report.json and stops.
  --part neb    endpoints, the CI-NEB (killed once inside a VASP call and
                finished with --resume), and criteria B1-B8.

Acceptance criteria, fixed before the run (ADR 0009):

  B1  max|F| < 0.02 eV/Ang on the H atoms within 300 iterations
  B2  every VASP run of the NEB reached EDIFF
  B3  1 imaginary and 0 zero modes in the 6-mode partial Hessian
  B4  eigenvector-tangent alignment >= 0.9                       (UNCERTAIN)
  B5  |E_NEB - (0.478 + D)| <= 0.03 eV, D = our medium barrier at the
      production setting minus SBH17's 0.467 eV                  (UNCERTAIN)
  B6  TS geometry: data only
  B7  killed inside a VASP call, finished with --resume, B1-B5 still met
  B8  |T*S| < 1 meV/atom in every image's final OUTCAR

Every stage is checkpointed: rerunning the script skips work whose output
already exists. Typical use (run detached):

  nohup caffeinate -i python3 examples/h2_cu111_vasp/run_validation.py \\
      --part prep --env-script ~/vasp/vasp661_env.sh > h2cu_prep.log 2>&1 &

POTCAR files are licensed and are never written into the repository: the
script concatenates <potcar-dir>/H/POTCAR and <potcar-dir>/Cu/POTCAR into
the run directories only.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from yamatsumi.directories import get_image_dirs
from yamatsumi.geometry import (
    cartesian_to_direct,
    direct_to_cartesian,
    mic_displacement,
)
from yamatsumi.outcar import parse_energy_from_outcar, scf_convergence_problem
from yamatsumi.poscar import Poscar, read_poscar, write_poscar
from yamatsumi.vasp_runner import run_vasp_for_images

# --- SBH17 reference data (J. Chem. Theory Comput. 2023, 19, 245) ----------
SBH17_HIGH_EV = 0.478  # Table 5, PBE, high algorithm
SBH17_MEDIUM_EV = 0.467  # Table 5, PBE, medium algorithm
SBH17_A_PBE = 3.624  # Table 3, PBE Cu lattice constant (Ang)
# Table 3 also lists the all-electron PBE value of Haas, Tran, Blaha,
# Phys. Rev. B 79, 085104 (2009); the check is made against it (ADR 0009).
HAAS_A_PBE = 3.632
SRP_TS_R = 1.03  # Table 2, SRP43 TS: H-H distance (Ang)
SRP_TS_Z = 1.16  # Table 2, SRP43 TS: height above the top layer (Ang)
FAR_Z = 6.0  # Section III.2: molecule 6 Ang from the surface
VACUUM = 13.0  # Table S3, SBH17 row
N_LAYERS = 4  # Table S3, SRP14 row (2x2, 4 layers)

# --- decisions of ADR 0009 --------------------------------------------------
SCAN_K = (6, 8, 11)
SCAN_ENCUT = (350, 450)
SCAN_TOL_EV = 0.010  # lightest setting within 10 meV of the densest
NEAR_Z = 4.0  # used for the NEB if within 5 meV of FAR_Z
HEIGHT_TOL_EV = 0.005
DIPOLE_TOL_EV = 0.005
LATTICE_TOL = 0.01  # Ang, against HAAS_A_PBE
BUDGET_HOURS = 36.0
PLANNED_ITERATIONS = 100  # for the time projection only
NUM_IMAGES = 5
PERTURB_ANG = 0.05
SEED = 20260930
FORCE_TOL = 0.02
MAX_ITER = 300
KILL_AT_ITERATION = 10  # B7: kill inside the VASP call of this iteration
FINAL_H_HEIGHT = 0.95  # Ang, start of the 2H relaxation only
BARRIER_TOL_EV = 0.03
RUN_OUTPUT_DIRS = ("neb",)

ELEMENTS = ("H", "Cu")  # ordered by atomic number, like the POTCAR

INCAR_BASE = """\
SYSTEM = H2 + Cu(111) (yamatsumi stage-b validation)
ENCUT  = {encut}
PREC   = Accurate
EDIFF  = 1E-6
{smear}LREAL  = .FALSE.
ISPIN  = 1
NELM   = 150
LCHARG = .FALSE.
{extra}"""
SINGLE_POINT = "LWAVE  = .FALSE.\nNSW    = 0\nIBRION = -1\nISYM   = 0\n"
NEB_POINT = "LWAVE  = .TRUE.\nNSW    = 0\nIBRION = -1\nISYM   = 0\n"
RELAX = "LWAVE  = .FALSE.\nNSW    = 200\nIBRION = 2\nISIF   = 2\nEDIFFG = -0.005\n"
SMEAR_MP = "ISMEAR = 1          ! Methfessel-Paxton, as SBH17 (Table S3)\nSIGMA  = 0.1\n"
SMEAR_MOLECULE = "ISMEAR = 0          ! isolated molecule\nSIGMA  = 0.01\n"
BULK_POINT = "LWAVE  = .FALSE.\nNSW    = 0\nIBRION = -1\n"  # symmetry kept: 21^3 k-points
RELAX_DONE_MARKER = "reached required accuracy"


def incar(encut: int, extra: str, smear: str = SMEAR_MP) -> str:
    return INCAR_BASE.format(encut=encut, extra=extra, smear=smear)


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}", flush=True)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def fcc_primitive(a: float) -> Poscar:
    lat = 0.5 * a * np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    return Poscar("fcc Cu", 1.0, lat, ["Cu"], [1], "Direct", np.zeros((1, 3)))


def slab_1x1(a: float) -> Poscar:
    """
    Unrelaxed fcc(111) slab of N_LAYERS layers in the 1x1 cell, stacked
    A-B-C-A from the bottom, with VACUUM Ang above it. a1 lies along x, a2 at
    60 degrees; the bottom layer is fixed.
    """
    s = a / np.sqrt(2.0)
    d = a / np.sqrt(3.0)
    c = (N_LAYERS - 1) * d + VACUUM
    lat = np.array(
        [[s, 0.0, 0.0], [0.5 * s, 0.5 * np.sqrt(3.0) * s, 0.0], [0.0, 0.0, c]]
    )
    z0 = 0.5
    frac = np.array(
        [[(i % 3) / 3.0, (i % 3) / 3.0, (z0 + i * d) / c] for i in range(N_LAYERS)]
    )
    sd = np.ones((N_LAYERS, 3), dtype=bool)
    sd[0] = False
    return Poscar("Cu(111) 1x1", 1.0, lat, ["Cu"], [N_LAYERS], "Direct", frac, sd)


def supercell(slab: Poscar, n: int) -> Poscar:
    """In-plane n x n repetition of a 1x1 slab; every Cu atom fixed."""
    lat = slab.lattice.copy()
    lat[0] *= n
    lat[1] *= n
    cart = direct_to_cartesian(slab.positions, slab.lattice)
    reps = [
        cart + i * slab.lattice[0] + j * slab.lattice[1]
        for i in range(n)
        for j in range(n)
    ]
    cart = np.vstack(reps)
    frac = cartesian_to_direct(cart, lat) % 1.0
    return Poscar(
        f"Cu(111) {n}x{n}",
        1.0,
        lat,
        ["Cu"],
        [len(cart)],
        "Direct",
        frac,
        np.zeros((len(cart), 3), dtype=bool),
    )


def surface_sites(slab: Poscar) -> dict:
    """
    Cartesian bridge, fcc and hcp sites next to the bridge between a top
    atom and its neighbour along a1, and the top-layer height. The hcp hollow
    is the one above a second-layer atom; that is checked, not assumed.
    """
    cart = direct_to_cartesian(slab.positions, slab.lattice)
    z = cart[:, 2]
    layers = np.unique(np.round(z, 3))
    top_z, second_z = layers[-1], layers[-2]
    top = cart[np.argmax(z)]
    a1 = slab.lattice[0] / (slab.lattice[0] @ slab.lattice[0]) ** 0.5
    s = min(
        np.linalg.norm(cart[i, :2] - top[:2])
        for i in range(len(cart))
        if abs(z[i] - top_z) < 1e-3 and np.linalg.norm(cart[i, :2] - top[:2]) > 1e-3
    )
    ey = np.array([-a1[1], a1[0], 0.0])
    bridge = top + 0.5 * s * a1
    offset = s * np.sqrt(3.0) / 6.0
    candidates = [bridge + offset * ey, bridge - offset * ey]
    second = cart[np.abs(z - second_z) < 1e-3]

    def lateral_gap(p):
        d = direct_to_cartesian(
            mic_displacement(
                cartesian_to_direct(second, slab.lattice),
                cartesian_to_direct(p[None, :], slab.lattice),
            ),
            slab.lattice,
        )
        return np.min(np.linalg.norm(d[:, :2], axis=1))

    gaps = [lateral_gap(p) for p in candidates]
    hcp_i = int(np.argmin(gaps))
    if gaps[hcp_i] > 1e-3 or gaps[1 - hcp_i] < 0.5:
        raise SystemExit(f"hollow classification failed: lateral gaps {gaps}")
    return dict(
        bridge=bridge,
        hcp=candidates[hcp_i],
        fcc=candidates[1 - hcp_i],
        ey=ey,
        top_z=top_z,
        nn=s,
    )


def with_h(slab: Poscar, h_cart: np.ndarray) -> Poscar:
    """Slab plus two H atoms (listed first, by atomic number); H free, Cu fixed."""
    cart = np.vstack([h_cart, direct_to_cartesian(slab.positions, slab.lattice)])
    frac = cartesian_to_direct(cart, slab.lattice)
    sd = np.vstack(
        [np.ones((2, 3), dtype=bool), np.zeros((slab.n_atoms, 3), dtype=bool)]
    )
    return Poscar(
        "H2 + Cu(111)",
        1.0,
        slab.lattice,
        ["H", "Cu"],
        [2, slab.n_atoms],
        "Direct",
        frac,
        sd,
    )


def h2_over_bridge(slab: Poscar, height: float, r: float) -> Poscar:
    """H2 parallel to the surface over the bridge, axis towards the two hollows."""
    site = surface_sites(slab)
    centre = site["bridge"].copy()
    centre[2] = site["top_z"] + height
    return with_h(
        slab, np.array([centre - 0.5 * r * site["ey"], centre + 0.5 * r * site["ey"]])
    )


def two_h_in_hollows(slab: Poscar, height: float) -> Poscar:
    """One H in the fcc and one in the hcp hollow next to the bridge.

    The atom order matches h2_over_bridge: H1 on the -ey side, H2 on +ey.
    """
    site = surface_sites(slab)
    pts = sorted(
        [site["fcc"], site["hcp"]], key=lambda p: (p - site["bridge"]) @ site["ey"]
    )
    return with_h(
        slab, np.array([p + [0.0, 0.0, site["top_z"] + height - p[2]] for p in pts])
    )


def h2_in_box(r: float, box=(10.0, 10.5, 11.0)) -> Poscar:
    lat = np.diag(box)
    cart = np.array([[5.0, 5.0, 5.5 - r / 2], [5.0, 5.0, 5.5 + r / 2]])
    return Poscar(
        "H2 in a box", 1.0, lat, ["H"], [2], "Direct", cartesian_to_direct(cart, lat)
    )


def h_geometry(p: Poscar, slab: Poscar) -> dict:
    """H-H distance, centre height above the top layer, lateral offset from the bridge."""
    cart = direct_to_cartesian(p.positions, p.lattice)
    h1, h2 = cart[0], cart[1]
    d = direct_to_cartesian(mic_displacement(p.positions[0], p.positions[1]), p.lattice)
    centre = h1 + 0.5 * d
    site = surface_sites(slab)
    lateral = direct_to_cartesian(
        mic_displacement(
            cartesian_to_direct(site["bridge"][None, :], p.lattice)[0],
            cartesian_to_direct(centre[None, :], p.lattice)[0],
        ),
        p.lattice,
    )
    axis = d / np.linalg.norm(d)
    return dict(
        r_HH=float(np.linalg.norm(d)),
        Z=float(centre[2] - site["top_z"]),
        lateral_from_bridge=float(np.linalg.norm(lateral[:2])),
        tilt_deg=float(np.degrees(np.arcsin(min(1.0, abs(axis[2]))))),
    )


# ---------------------------------------------------------------------------
# Numerics
# ---------------------------------------------------------------------------


def birch_murnaghan_a0(
    a_values, energies, n_atoms_cell: int = 1, cell_factor: float = 0.25
):
    """
    Equilibrium fcc lattice constant from E(a): a third-order Birch-Murnaghan
    fit, done as the standard cubic polynomial in V^(-2/3). V = cell_factor
    * a^3 is the primitive-cell volume (a^3/4 for fcc).
    """
    v = cell_factor * np.asarray(a_values) ** 3
    x = v ** (-2.0 / 3.0)
    coeff = np.polyfit(x, energies, 3)
    roots = np.roots(np.polyder(coeff))
    roots = roots[np.isreal(roots)].real
    roots = roots[(roots > x.min()) & (roots < x.max())]
    if len(roots) != 1 or np.polyval(np.polyder(coeff, 2), roots[0]) <= 0:
        raise SystemExit("E(a) has no single minimum inside the scanned range")
    return float((roots[0] ** -1.5 / cell_factor) ** (1.0 / 3.0))


def select_setting(barriers: dict, tol: float = SCAN_TOL_EV):
    """
    The cheapest (k, encut) whose barrier is within *tol* of the densest one.
    Cost is taken as k^2 * encut^1.5: k-points times plane waves.
    """
    dense = (max(k for k, _ in barriers), max(e for _, e in barriers))
    ok = [ke for ke, eb in barriers.items() if abs(eb - barriers[dense]) <= tol]
    return min(ok, key=lambda ke: ke[0] ** 2 * ke[1] ** 1.5), dense


def dipol_centre(p: Poscar, h_z: float) -> float:
    """
    DIPOL (fractional z) placing the dipole-correction discontinuity, which
    VASP puts half a cell away from DIPOL, midway between the far H2 and the
    periodic image of the slab bottom.
    """
    c = p.lattice[2, 2]
    z = direct_to_cartesian(p.positions, p.lattice)[:, 2]
    bottom_image = z.min() + c
    jump = 0.5 * (h_z + bottom_image)
    return ((jump - 0.5 * c) / c) % 1.0


# ---------------------------------------------------------------------------
# VASP plumbing
# ---------------------------------------------------------------------------


def vasp_command(args) -> str:
    cmd = args.vasp_cmd or "mpirun -np 4 $VASP_BIN/vasp_std"
    if args.env_script is None:
        if args.vasp_cmd is None:
            raise SystemExit("Give --vasp-cmd or --env-script.")
        return cmd
    env = Path(args.env_script).expanduser().resolve()
    return f"/bin/zsh -c {shlex.quote(f'source {env} && exec {cmd}')}"


def potcar_text(potcar_dir: Path, species) -> str:
    return "".join((potcar_dir / el / "POTCAR").read_text() for el in species)


def kpoints(n: int, nz: int = 1) -> str:
    return f"Gamma-centred {n}x{n}x{nz}\n0\nGamma\n{n} {n} {nz}\n0 0 0\n"


def stage(
    run_dir: Path, poscar: Poscar, incar: str, kpts: str, potcar_dir: Path
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "INCAR").write_text(incar)
    (run_dir / "KPOINTS").write_text(kpts)
    target = run_dir / "POTCAR"
    if target.exists():
        target.unlink()
    target.write_text(potcar_text(potcar_dir, poscar.species))
    write_poscar(str(run_dir / "POSCAR"), poscar)


def run(run_dir: Path, vasp_cmd: str, relax: bool = False) -> float:
    """Run VASP once in *run_dir* (skipped if already done); return the energy."""
    outcar = run_dir / "OUTCAR"
    done = scf_convergence_problem(outcar) is None and (
        not relax or RELAX_DONE_MARKER in outcar.read_text(errors="replace")
    )
    if not done:
        t0 = time.time()
        with open(run_dir / "vasp_run.log", "w") as fh:
            rc = subprocess.run(
                shlex.split(vasp_cmd), cwd=run_dir, stdout=fh, stderr=subprocess.STDOUT
            ).returncode
        problem = scf_convergence_problem(outcar)
        text = outcar.read_text(errors="replace") if outcar.exists() else ""
        if rc != 0 or problem or (relax and RELAX_DONE_MARKER not in text):
            raise SystemExit(
                f"{run_dir}: rc={rc}, SCF: {problem}, "
                f"relaxed: {RELAX_DONE_MARKER in text}"
            )
        log(f"  {run_dir.name}: {time.time() - t0:.0f} s")
    return parse_energy_from_outcar(outcar)


def outcar_stats(outcar: Path) -> dict:
    """Wall time, SCF steps and |T*S| per atom of a single-point OUTCAR."""
    text = outcar.read_text(errors="replace")
    wall = re.search(r"Elapsed time \(sec\):\s+([\d.]+)", text)
    ts = re.findall(r"EENTRO\s*=\s*([-\d.]+)", text)
    n = int(re.search(r"NIONS\s*=\s*(\d+)", text).group(1))
    return dict(
        wall_s=float(wall.group(1)) if wall else None,
        scf_steps=text.count("LOOP:"),
        abs_TS_meV_per_atom=abs(float(ts[-1])) * 1000.0 / n if ts else None,
    )


# ---------------------------------------------------------------------------
# Part 1: preparation and setting scan
# ---------------------------------------------------------------------------


def part_prep(work: Path, cmd: str, potcar_dir: Path) -> None:
    prep = work / "prep"
    report_path = work / "prep_report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}

    def save():
        report_path.write_text(json.dumps(report, indent=2))

    # Lattice constant: E(a) at fixed cells, so no Pulay stress.
    a_grid = [round(3.56 + 0.02 * i, 2) for i in range(7)]
    energies = []
    for a in a_grid:
        d = prep / "bulk" / f"a_{a:.2f}"
        if not (d / "OUTCAR").exists():
            stage(
                d,
                fcc_primitive(a),
                incar(450, BULK_POINT),
                kpoints(21, 21),
                potcar_dir,
            )
        energies.append(run(d, cmd))
    a0 = birch_murnaghan_a0(a_grid, energies)
    report["bulk"] = dict(
        a_grid=a_grid,
        energies_eV=energies,
        a0=a0,
        sbh17_a0=SBH17_A_PBE,
        difference=a0 - SBH17_A_PBE,
        all_electron_a0=HAAS_A_PBE,
        difference_all_electron=a0 - HAAS_A_PBE,
    )
    save()
    log(
        f"prep: a0 = {a0:.4f} Ang (SBH17 PBE {SBH17_A_PBE}, diff {a0 - SBH17_A_PBE:+.4f}; "
        f"all-electron {HAAS_A_PBE}, diff {a0 - HAAS_A_PBE:+.4f})"
    )
    if abs(a0 - HAAS_A_PBE) > LATTICE_TOL:
        raise SystemExit(
            "prep: a0 differs from the all-electron PBE value by more than 0.01 Ang; "
            "stopping (ADR 0009)"
        )

    # H2 bond length in a box.
    d = prep / "h2_box"
    if not (d / "OUTCAR").exists():
        stage(
            d,
            h2_in_box(0.75),
            incar(450, RELAX.replace("EDIFFG = -0.005", "EDIFFG = -0.001"), SMEAR_MOLECULE),
            kpoints(1, 1),
            potcar_dir,
        )
    run(d, cmd, relax=True)
    h2 = read_poscar(str(d / "CONTCAR"))
    r0 = float(
        np.linalg.norm(
            direct_to_cartesian(
                mic_displacement(h2.positions[0], h2.positions[1]), h2.lattice
            )
        )
    )
    report["h2_bond_length"] = r0
    save()
    log(f"prep: H2 bond length {r0:.4f} Ang")

    # Interlayer relaxation of the 1x1 slab, bottom layer fixed.
    d = prep / "slab_1x1"
    if not (d / "OUTCAR").exists():
        stage(
            d,
            slab_1x1(a0),
            incar(450, RELAX),
            kpoints(24),
            potcar_dir,
        )
    run(d, cmd, relax=True)
    slab = read_poscar(str(d / "CONTCAR"))
    z = np.sort(direct_to_cartesian(slab.positions, slab.lattice)[:, 2])
    d_bulk = a0 / np.sqrt(3.0)
    report["slab"] = dict(
        layer_z=z.tolist(),
        d_bulk=d_bulk,
        d12_change_percent=100.0 * ((z[-1] - z[-2]) / d_bulk - 1.0),
        sbh17_pbe_d12_change_percent=-0.26,
    )
    save()
    log(
        f"prep: d12 change {report['slab']['d12_change_percent']:+.2f} % (SBH17 PBE -0.26 %)"
    )

    slab2 = supercell(slab, 2)
    ts_geom = h2_over_bridge(slab2, SRP_TS_Z, SRP_TS_R)
    far_geom = h2_over_bridge(slab2, FAR_Z, r0)
    log(f"prep: TS geometry {h_geometry(ts_geom, slab2)}")
    log(f"prep: far geometry {h_geometry(far_geom, slab2)}")

    # Medium-algorithm scan, cheapest first.
    scan = report.setdefault("scan", {})
    settings = sorted(
        ((k, e) for k in SCAN_K for e in SCAN_ENCUT),
        key=lambda ke: ke[0] ** 2 * ke[1] ** 1.5,
    )
    for k, encut in settings:
        key = f"k{k}_e{encut}"
        res = {}
        for name, geom in (("ts", ts_geom), ("far", far_geom)):
            d = prep / "scan" / key / name
            if not (d / "OUTCAR").exists():
                stage(
                    d,
                    geom,
                    incar(encut, SINGLE_POINT),
                    kpoints(k),
                    potcar_dir,
                )
            res[f"E_{name}"] = run(d, cmd)
            res[f"{name}_stats"] = outcar_stats(d / "OUTCAR")
        res["barrier"] = res["E_ts"] - res["E_far"]
        scan[key] = res
        save()
        log(f"prep: scan {key}: medium barrier {res['barrier']:.4f} eV")

    barriers = {
        (k, e): scan[f"k{k}_e{e}"]["barrier"] for k in SCAN_K for e in SCAN_ENCUT
    }
    (k_sel, e_sel), dense = select_setting(barriers)
    report["selected"] = dict(
        k=k_sel,
        encut=e_sel,
        dense=list(dense),
        barrier=barriers[(k_sel, e_sel)],
        D=barriers[(k_sel, e_sel)] - SBH17_MEDIUM_EV,
    )
    save()
    log(
        f"prep: selected k={k_sel}, ENCUT={e_sel}; D = {report['selected']['D']:+.4f} eV"
    )

    sel_incar = incar(e_sel, SINGLE_POINT)
    checks = report.setdefault("checks", {})

    # H2 height: 4 Ang against 6 Ang.
    d = prep / "height_4A"
    if not (d / "OUTCAR").exists():
        stage(
            d, h2_over_bridge(slab2, NEAR_Z, r0), sel_incar, kpoints(k_sel), potcar_dir
        )
    e4 = run(d, cmd)
    e6 = scan[f"k{k_sel}_e{e_sel}"]["E_far"]
    checks["height"] = dict(
        E_4A_minus_E_6A_eV=e4 - e6,
        neb_start_height=NEAR_Z if abs(e4 - e6) < HEIGHT_TOL_EV else FAR_Z,
    )
    save()
    log(f"prep: E(4 A) - E(6 A) = {1000 * (e4 - e6):+.2f} meV")

    # Dipole correction on both points of the medium barrier.
    dip = {}
    for name, geom in (("ts", ts_geom), ("far", far_geom)):
        d = prep / "dipole" / name
        if not (d / "OUTCAR").exists():
            h_z = direct_to_cartesian(geom.positions, geom.lattice)[:2, 2].mean()
            extra = f"IDIPOL = 3\nLDIPOL = .TRUE.\nDIPOL  = 0.5 0.5 {dipol_centre(far_geom, h_z):.4f}\n"
            stage(d, geom, sel_incar + extra, kpoints(k_sel), potcar_dir)
        dip[name] = run(d, cmd)
    shift = (dip["ts"] - dip["far"]) - report["selected"]["barrier"]
    checks["dipole"] = dict(
        barrier_shift_eV=shift, use_dipole=abs(shift) >= DIPOLE_TOL_EV
    )
    save()
    log(f"prep: dipole correction shifts the barrier by {1000 * shift:+.2f} meV")

    # Cell size: 3x3x4 with 4x4x1 k-points against 2x2x4 with 6x6x1.
    slab3 = supercell(slab, 3)
    e3 = {}
    for name, geom in (
        ("ts", h2_over_bridge(slab3, SRP_TS_Z, SRP_TS_R)),
        ("far", h2_over_bridge(slab3, FAR_Z, r0)),
    ):
        d = prep / "cell_3x3" / name
        if not (d / "OUTCAR").exists():
            stage(d, geom, sel_incar, kpoints(4), potcar_dir)
        e3[name] = run(d, cmd)
    b3 = e3["ts"] - e3["far"]
    b2 = scan[f"k6_e{e_sel}"]["barrier"]
    checks["cell"] = dict(barrier_3x3_k4=b3, barrier_2x2_k6=b2, difference_eV=b3 - b2)
    save()
    log(f"prep: 3x3 - 2x2 medium barrier {1000 * (b3 - b2):+.2f} meV")

    # Time projection of the NEB at the selected setting.
    st = scan[f"k{k_sel}_e{e_sel}"]["ts_stats"]
    per_step = st["wall_s"] / max(st["scf_steps"], 1)
    hours = PLANNED_ITERATIONS * NUM_IMAGES * 10 * per_step / 3600.0
    report["projection"] = dict(
        seconds_per_scf_step=per_step,
        scf_steps_per_image=10,
        iterations=PLANNED_ITERATIONS,
        hours=hours,
        within_budget=hours <= BUDGET_HOURS,
    )
    save()
    log(
        f"prep: NEB projected at {hours:.1f} h ({'within' if hours <= BUDGET_HOURS else 'OVER'} "
        f"the {BUDGET_HOURS:.0f} h budget); prep done"
    )


# ---------------------------------------------------------------------------
# Part 2: endpoints, NEB, evaluation
# ---------------------------------------------------------------------------


def perturb_h(neb: Path, max_ang: float, seed: int) -> None:
    """Seeded random displacement of at most *max_ang* on the free (H) atoms."""
    rng = np.random.default_rng(seed)
    for idir in get_image_dirs(neb, NUM_IMAGES)[1:-1]:
        p = read_poscar(str(idir / "POSCAR"))
        free = p.selective_dynamics.all(axis=1)
        d = np.zeros_like(p.positions)
        v = rng.normal(size=(int(free.sum()), 3))
        v *= (rng.uniform(0.0, 1.0, size=len(v)) / np.linalg.norm(v, axis=1))[:, None]
        v *= max_ang / np.linalg.norm(v, axis=1).max()
        d[free] = v
        cart = direct_to_cartesian(p.positions, p.lattice) + d
        write_poscar(
            str(idir / "POSCAR"), p.with_positions(cartesian_to_direct(cart, p.lattice))
        )


def yamatsumi_cmd(argv) -> list:
    return [sys.executable, "-m", "yamatsumi", *argv]


def kill_inside_vasp(proc: subprocess.Popen, neb: Path, log_path: Path) -> dict:
    """
    Wait until iteration KILL_AT_ITERATION is running VASP, then kill the
    whole process group (yamatsumi, mpirun and VASP) part-way through it.
    """
    csv_path = neb / "convergence_history.csv"
    while proc.poll() is None:
        rows = list(csv.DictReader(open(csv_path))) if csv_path.exists() else []
        tail = (
            log_path.read_text(errors="replace").splitlines()[-1:]
            if log_path.exists()
            else []
        )
        if len(rows) >= KILL_AT_ITERATION and tail and "Launching VASP:" in tail[0]:
            time.sleep(30)
            victim = tail[0].split("Launching VASP:")[-1].strip()
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            return dict(killed_at_iteration=len(rows) + 1, image=victim)
        time.sleep(5)
    return dict(killed_at_iteration=None, image=None, exit_code=proc.returncode)


def vasp_running_in(directory: Path) -> list:
    """PIDs of vasp_std processes whose working directory lies under *directory*.

    mpirun may start VASP outside the killed process group, so the group
    check alone could miss a VASP that keeps writing an OUTCAR the resumed
    run is about to read.
    """
    pids = subprocess.run(["pgrep", "-x", "vasp_std"], capture_output=True,
                          text=True).stdout.split()
    found = []
    for pid in pids:
        out = subprocess.run(["lsof", "-a", "-d", "cwd", "-p", pid, "-Fn"],
                             capture_output=True, text=True).stdout
        cwd = next((ln[1:] for ln in out.splitlines() if ln.startswith("n")), "")
        if cwd and Path(cwd).resolve().is_relative_to(directory.resolve()):
            found.append(pid)
    return found


def part_neb(work: Path, cmd: str, potcar_dir: Path, approved_hours: float = 0.0,
             step_size: float | None = None) -> None:
    report = json.loads((work / "prep_report.json").read_text())
    projection = report.get("projection")
    if projection is None:
        raise SystemExit("neb: prep is unfinished")
    if not projection["within_budget"] and projection["hours"] > approved_hours:
        raise SystemExit(
            f"neb: projected {projection['hours']:.1f} h is over the {BUDGET_HOURS:.0f} h "
            "budget; decide with the user and pass --approved-hours (ADR 0009)"
        )
    k, encut = report["selected"]["k"], report["selected"]["encut"]
    use_dipole = report["checks"]["dipole"]["use_dipole"]
    if use_dipole:
        raise SystemExit(
            "neb: the dipole check asks for the correction; not planned in ADR 0009"
        )
    height = report["checks"]["height"]["neb_start_height"]
    r0 = report["h2_bond_length"]
    slab2 = supercell(read_poscar(str(work / "prep" / "slab_1x1" / "CONTCAR")), 2)

    ends = {}
    for name, geom in (
        ("init", h2_over_bridge(slab2, height, r0)),
        ("final", two_h_in_hollows(slab2, FINAL_H_HEIGHT)),
    ):
        d = work / "endpoints" / name
        if not (d / "OUTCAR").exists():
            stage(
                d,
                geom,
                incar(encut, RELAX + "ISYM   = 0\n"),
                kpoints(k),
                potcar_dir,
            )
        run(d, cmd, relax=True)
        ends[name] = read_poscar(str(d / "CONTCAR"))
        log(f"neb: {name} endpoint {h_geometry(ends[name], slab2)}")
    for name, p in ends.items():
        write_poscar(str(work / f"POSCAR_{name}"), p)

    neb = work / "neb"
    copy_dir = work / "CopyDir"
    copy_dir.mkdir(exist_ok=True)
    (copy_dir / "INCAR").write_text(incar(encut, NEB_POINT))
    (copy_dir / "KPOINTS").write_text(kpoints(k))
    pot = copy_dir / "POTCAR"
    if pot.exists():
        pot.unlink()
    pot.write_text(potcar_text(potcar_dir, ELEMENTS))
    common = [
        "--init",
        str(work / "POSCAR_init"),
        "--final",
        str(work / "POSCAR_final"),
        "--copy-dir",
        str(copy_dir),
        "--work-dir",
        str(neb),
        "--vasp-cmd",
        cmd,
        "--num-images",
        str(NUM_IMAGES),
        "--force-tol",
        str(FORCE_TOL),
        "--no-show",
    ]
    if step_size is not None:
        # ADR 0009 Amendment 3: raised to the VTST MAXMOVE after iteration 11
        # of the first run showed the 0.05 Ang default barely moving the path.
        common += ["--step-size", str(step_size)]
        log(f"neb: --step-size {step_size}")
    log_path = work / "yamatsumi.log"
    state_path = work / "neb_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}

    if "kill" not in state:
        if (
            subprocess.run(
                yamatsumi_cmd(["--phase", "init", "--dry-run", *common]),
                cwd=REPO,
                stdout=open(log_path, "a"),
                stderr=subprocess.STDOUT,
            ).returncode
            != 0
        ):
            raise SystemExit("neb: staging failed; see yamatsumi.log")
        perturb_h(neb, PERTURB_ANG, SEED)
        log(
            f"neb: H atoms of the intermediate images perturbed by up to {PERTURB_ANG} Ang"
        )
        run_vasp_for_images(get_image_dirs(neb, NUM_IMAGES), cmd)
        log(
            f"neb: part 1, to be killed inside a VASP call of iteration {KILL_AT_ITERATION + 1}"
        )
        proc = subprocess.Popen(
            yamatsumi_cmd(["--phase", "neb", "--max-iter", str(MAX_ITER), *common]),
            cwd=REPO,
            stdout=open(log_path, "a"),
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        state["kill"] = kill_inside_vasp(proc, neb, log_path)
        state_path.write_text(json.dumps(state, indent=2))
        log(f"neb: killed: {state['kill']}")
        time.sleep(10)
        leftovers = subprocess.run(
            ["pgrep", "-g", str(proc.pid)], capture_output=True, text=True
        ).stdout.split()
        if leftovers:
            raise SystemExit(f"neb: processes of the killed group survive: {leftovers}")
        strays = vasp_running_in(neb)
        if strays:
            raise SystemExit(f"neb: VASP still runs in {neb} after the kill: {strays}")

    if not (neb / "yamatsumi_result.txt").exists():
        log("neb: part 2, --resume")
        rc = subprocess.run(
            yamatsumi_cmd(
                ["--phase", "neb", "--resume", "--max-iter", str(MAX_ITER), *common]
            ),
            cwd=REPO,
            stdout=open(log_path, "a"),
            stderr=subprocess.STDOUT,
        ).returncode
        log(f"neb: finished with exit code {rc}")


def _result_fields(text: str) -> dict:
    fields = {}
    for line in text.splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            fields[key.strip()] = value.strip()
    return fields


def evaluate(work: Path) -> dict:
    prep = json.loads((work / "prep_report.json").read_text())
    state = json.loads((work / "neb_state.json").read_text())
    neb = work / "neb"
    report: dict = {"criteria": {}, "data": {}}
    crit, data = report["criteria"], report["data"]

    rows = list(csv.DictReader(open(neb / "convergence_history.csv")))
    iters = [int(r["iteration"]) for r in rows]
    max_f = [float(r["max_force_eV_per_Ang"]) for r in rows]
    result_path = neb / "yamatsumi_result.txt"
    res = _result_fields(result_path.read_text()) if result_path.exists() else {}
    data.update(iterations=len(rows), final_max_force=max_f[-1] if max_f else None)
    crit["B1_converged"] = bool(
        result_path.exists() and max_f and max_f[-1] < FORCE_TOL
    )

    outcars = sorted(p for sub in RUN_OUTPUT_DIRS for p in (work / sub).rglob("OUTCAR"))
    problems = {str(p.relative_to(work)): scf_convergence_problem(p) for p in outcars}
    data["outcars_checked"] = len(outcars)
    data["scf_problems"] = {k: v for k, v in problems.items() if v}
    crit["B2_all_scf_converged"] = not data["scf_problems"]

    n_imag = int(res.get("Imaginary frequencies found", -1))
    n_zero = next(
        (int(v) for key, v in res.items() if key.startswith("Zero modes")), -1
    )
    data.update(n_imaginary=n_imag, n_zero_modes=n_zero)
    crit["B3_one_imaginary_no_zero"] = n_imag == 1 and n_zero == 0

    align = res.get("Eigenvector-tangent alignment")
    data["eigenvector_tangent_alignment"] = float(align.split()[0]) if align else None
    crit["B4_alignment"] = bool(align) and data["eigenvector_tangent_alignment"] >= 0.9

    fwd = res.get("Forward  activation barrier")
    e_neb = float(fwd.split()[0]) if fwd else None
    target = SBH17_HIGH_EV + prep["selected"]["D"]
    data.update(barrier_eV=e_neb, target_eV=target, D_eV=prep["selected"]["D"])
    crit["B5_barrier"] = e_neb is not None and abs(e_neb - target) <= BARRIER_TOL_EV

    slab2 = supercell(read_poscar(str(work / "prep" / "slab_1x1" / "CONTCAR")), 2)
    if (neb / "TS_POSCAR").exists():
        data["ts_geometry"] = h_geometry(read_poscar(str(neb / "TS_POSCAR")), slab2)

    data["kill"] = state.get("kill")
    data["iteration_numbers_continuous"] = iters == list(range(1, len(iters) + 1))
    crit["B7_resume_after_kill_inside_vasp"] = bool(
        state.get("kill", {}).get("killed_at_iteration")
        and data["iteration_numbers_continuous"]
        and all(
            crit[c]
            for c in (
                "B1_converged",
                "B2_all_scf_converged",
                "B3_one_imaginary_no_zero",
                "B4_alignment",
                "B5_barrier",
            )
        )
    )

    ts_vals = [
        outcar_stats(d / "OUTCAR")["abs_TS_meV_per_atom"]
        for d in get_image_dirs(neb, NUM_IMAGES)
    ]
    data["max_abs_TS_meV_per_atom"] = max(ts_vals) if None not in ts_vals else None
    crit["B8_entropy_small"] = (
        data["max_abs_TS_meV_per_atom"] is not None
        and data["max_abs_TS_meV_per_atom"] < 1.0
    )
    report["passed"] = all(crit.values())
    return report


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--part",
        choices=["prep", "neb", "evaluate"],
        required=True,
        help="Which part to run (see above)",
    )
    p.add_argument(
        "--work-dir",
        default=str(Path(__file__).parent / "run"),
        help="Run directory (default: examples/h2_cu111_vasp/run, gitignored)",
    )
    p.add_argument(
        "--env-script",
        default=None,
        help="Shell script that sets VASP_BIN and VASP_POTCAR_DIR",
    )
    p.add_argument(
        "--vasp-cmd",
        default=None,
        help="VASP command (default: 'mpirun -np 4 $VASP_BIN/vasp_std')",
    )
    p.add_argument(
        "--potcar-dir",
        default=None,
        help="Directory holding H/POTCAR and Cu/POTCAR "
        "(default: $VASP_POTCAR_DIR from --env-script)",
    )
    p.add_argument(
        "--step-size", type=float, default=None,
        help="Passed to yamatsumi --step-size (default: yamatsumi's own default)",
    )
    p.add_argument(
        "--approved-hours", type=float, default=0.0,
        help="NEB wall time the user approved beyond the 36 h budget "
        "(default: 0, i.e. none); the NEB part refuses a longer projection",
    )
    args = p.parse_args()

    work = Path(args.work_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    if args.part != "evaluate":
        potcar_dir = args.potcar_dir or (
            subprocess.run(
                [
                    "/bin/zsh",
                    "-c",
                    f"source {shlex.quote(str(Path(args.env_script).expanduser()))}"
                    ' && printf %s "$VASP_POTCAR_DIR"',
                ],
                capture_output=True,
                text=True,
                check=True,
            ).stdout
            if args.env_script
            else None
        )
        if not potcar_dir:
            raise SystemExit("Give --potcar-dir or --env-script.")
        potcar_dir = Path(potcar_dir).expanduser()
        for el in ELEMENTS:
            if not (potcar_dir / el / "POTCAR").is_file():
                raise SystemExit(f"No POTCAR at {potcar_dir / el}")
        cmd = vasp_command(args)
        log(f"VASP command: {cmd}")
        if args.part == "prep":
            part_prep(work, cmd, potcar_dir)
            return
        part_neb(work, cmd, potcar_dir, args.approved_hours, args.step_size)

    report = evaluate(work)
    (work / "validation_report.json").write_text(json.dumps(report, indent=2))
    for name, ok in report["criteria"].items():
        log(f"{'PASS' if ok else 'FAIL'}  {name}")
    log(f"overall: {'PASS' if report['passed'] else 'FAIL'}")


if __name__ == "__main__":
    main()
