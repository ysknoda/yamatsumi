#!/usr/bin/env python3
"""
Stage (a) real-VASP validation of yamatsumi: a single vacancy hop in the
8-site conventional diamond Si cell (7 atoms).

This checks the software against real VASP, not the physics of the Si
vacancy: at one vacancy per 8 sites the barrier has no physical meaning.
What makes the cell useful is symmetry. The two endpoints are related by the
inversion centre at the midpoint of the hopping bond, so without fitting
anything we know that the endpoint energies must agree, that the
transition state must sit at the bond midpoint, and that it must have
exactly one imaginary mode along the path. The acceptance criteria below
were fixed before the first run and must not be relaxed afterwards to make
a run pass.

  A1  NEB converges: max|F_NEB| < --force-tol within --max-iter
  A2  every VASP run reached EDIFF (yamatsumi stops otherwise; re-checked)
  A3  |E_init - E_final| < 1 meV              (symmetry-equivalent endpoints)
  A4  migrating atom within 0.02 Ang of the bond midpoint in TS_POSCAR
  A5  exactly 1 imaginary mode and 3 zero modes (full Hessian)
  A6  eigenvector-tangent alignment >= 0.9
  A7  the run is interrupted after --interrupt-after iterations and finished
      with --phase neb --resume; iteration numbering must be continuous and
      the resumed run must itself satisfy A1-A6
  A8  |T*S| < 1 meV/atom in every image's final OUTCAR

A4 (0.02 Ang) and A6 (0.9) are below 90% confidence: how far the geometry
can sit from the saddle at a 0.02 eV/Ang force tolerance depends on the
unknown curvature there. A failure must be reported and explained, not
tuned away.

Stages are checkpointed: rerunning the script skips any stage whose output
already exists, and an interrupted NEB is continued with --resume.

Typical use (about 4 to 11 hours on a 4-core laptop; run it detached):

  nohup caffeinate -i python3 examples/si7_vacancy_vasp/run_validation.py \\
      --env-script ~/vasp/vasp661_env.sh > si7_validation.log 2>&1 &

POTCAR files are licensed and are never written into the repository: the
script copies <potcar-dir>/Si/POTCAR into the run directory only.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from yamatsumi.directories import get_image_dirs
from yamatsumi.geometry import direct_to_cartesian, mic_displacement
from yamatsumi.outcar import (
    parse_energy_from_outcar,
    scf_convergence_problem,
)
from yamatsumi.poscar import read_poscar

NUM_IMAGES = 4  # even, so no image starts on the symmetric saddle
N_ATOMS = 7

# Shared electronic settings. Every value is fixed in docs/real_vasp_validation.md.
INCAR_COMMON = """\
SYSTEM = Si7 vacancy hop (yamatsumi stage-a validation)
ENCUT  = 400        ! 1.63 x ENMAX (245.345 eV) of PAW_PBE Si 05Jan2001
PREC   = Accurate
EDIFF  = 1E-6
ISMEAR = 0          ! Gaussian smearing, VASP Wiki "Smearing technique"
SIGMA  = 0.02       ! 0.05 gave |T*S| = 0.92 meV/atom at the ideal geometry
LREAL  = .FALSE.    ! VASP Wiki: real-space projection only above ~30 atoms
ISYM   = 0          ! symmetry must come out of the calculation, not be imposed
ISPIN  = 1
NELM   = 100
LCHARG = .FALSE.
"""
INCAR_RELAX = (
    INCAR_COMMON
    + """\
LWAVE  = .FALSE.
NSW    = 100
IBRION = 2
ISIF   = 2          ! fixed cell
EDIFFG = -0.005     ! eV/Ang
"""
)
INCAR_NEB = (
    INCAR_COMMON
    + """\
LWAVE  = .TRUE.     ! reuse each image's WAVECAR in the next NEB iteration
NSW    = 0          ! single point: yamatsumi moves the atoms
IBRION = -1
"""
)
# Gamma-centred 4x4x4 on the 5.431 Ang cell: k spacing ~0.29 1/Ang, the same
# density as 2x2x2 on the 64-site cell of examples/si_vacancy_neb.
KPOINTS = "Gamma-centred 4x4x4\n0\nGamma\n4 4 4\n0 0 0\n"

RELAX_DONE_MARKER = "reached required accuracy"


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}", flush=True)


def vasp_command(args) -> str:
    """The VASP command, wrapped so that the environment script is sourced."""
    cmd = args.vasp_cmd
    if cmd is None:
        if args.env_script is None:
            raise SystemExit(
                "Give --vasp-cmd, or --env-script so the default "
                "'mpirun -np 4 $VASP_BIN/vasp_std' can be resolved."
            )
        cmd = "mpirun -np 4 $VASP_BIN/vasp_std"
    if args.env_script is None:
        return cmd
    env = Path(args.env_script).expanduser().resolve()
    return f"/bin/zsh -c {shlex.quote(f'source {env} && exec {cmd}')}"


def stage_inputs(run_dir: Path, incar: str, potcar: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "INCAR").write_text(incar)
    (run_dir / "KPOINTS").write_text(KPOINTS)
    shutil.copy(potcar, run_dir / "POTCAR")


def prepare(work: Path) -> None:
    ideal = work / "ideal"
    if (ideal / "POSCAR_final").exists():
        log("prepare: ideal endpoints already present, skipping")
        return
    script = REPO / "examples" / "si_vacancy_neb" / "make_endpoints.py"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--supercell-size",
            "1",
            "--output-dir",
            str(ideal),
        ],
        check=True,
    )
    log("prepare: ideal Si7 endpoints written")


def relax(work: Path, vasp_cmd: str, potcar: Path) -> None:
    for label in ("init", "final"):
        out = work / f"POSCAR_{label}"
        if out.exists():
            log(f"relax: {out.name} already present, skipping")
            continue
        run_dir = work / f"relax_{label}"
        stage_inputs(run_dir, INCAR_RELAX, potcar)
        shutil.copy(work / "ideal" / f"POSCAR_{label}", run_dir / "POSCAR")
        log(f"relax: relaxing the {label} endpoint")
        with open(run_dir / "vasp_run.log", "w") as fh:
            rc = subprocess.run(
                shlex.split(vasp_cmd), cwd=run_dir, stdout=fh, stderr=subprocess.STDOUT
            ).returncode
        problem = scf_convergence_problem(run_dir / "OUTCAR")
        text = (
            (run_dir / "OUTCAR").read_text(errors="replace")
            if (run_dir / "OUTCAR").exists()
            else ""
        )
        if rc != 0 or problem is not None or RELAX_DONE_MARKER not in text:
            raise SystemExit(
                f"relax_{label} failed: rc={rc}, SCF: {problem}, "
                f"ionic convergence reached: {RELAX_DONE_MARKER in text}"
            )
        shutil.copy(run_dir / "CONTCAR", out)
        log(
            f"relax: {label} done, E = {parse_energy_from_outcar(run_dir / 'OUTCAR'):.8f} eV"
        )


def yamatsumi(argv: list[str], log_path: Path) -> int:
    with open(log_path, "a") as fh:
        return subprocess.run(
            [sys.executable, "-m", "yamatsumi", *argv],
            cwd=REPO,
            stdout=fh,
            stderr=subprocess.STDOUT,
        ).returncode


def run_neb(work: Path, vasp_cmd: str, potcar: Path, args) -> None:
    neb = work / "neb"
    if (neb / "yamatsumi_result.txt").exists():
        log("neb: result already present, skipping")
        return
    copy_dir = work / "CopyDir"
    stage_inputs(copy_dir, INCAR_NEB, potcar)
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
        vasp_cmd,
        "--num-images",
        str(NUM_IMAGES),
        "--force-tol",
        str(args.force_tol),
        "--no-show",
    ]
    log_path = work / "yamatsumi.log"
    if not (neb / "convergence_history.csv").exists():
        log(f"neb: part 1, stopping after {args.interrupt_after} iterations (A7)")
        yamatsumi(
            ["--phase", "all", "--max-iter", str(args.interrupt_after), *common],
            log_path,
        )
    if (neb / "yamatsumi_result.txt").exists():
        log("neb: converged before the interruption point, so A7 is not exercised")
        return
    log(f"neb: part 2, --resume up to --max-iter {args.max_iter}")
    rc = yamatsumi(
        ["--phase", "neb", "--resume", "--max-iter", str(args.max_iter), *common],
        log_path,
    )
    log(f"neb: finished with exit code {rc}")


def _result_fields(text: str) -> dict:
    fields, freqs, in_freqs = {}, [], False
    for line in text.splitlines():
        if in_freqs:
            parts = line.split()
            if len(parts) >= 3 and parts[0].isdigit() and parts[2] == "meV":
                freqs.append(float(parts[1]))
                continue
            in_freqs = False
        if "All frequencies (meV)" in line:
            in_freqs = True
        elif ":" in line:
            key, _, value = line.partition(":")
            fields[key.strip()] = value.strip()
    fields["frequencies"] = freqs
    return fields


def evaluate(work: Path, args) -> dict:
    neb = work / "neb"
    report: dict = {"criteria": {}, "data": {}}
    crit, data = report["criteria"], report["data"]

    rows = list(csv.DictReader(open(neb / "convergence_history.csv")))
    iters = [int(r["iteration"]) for r in rows]
    max_f = [float(r["max_force_eV_per_Ang"]) for r in rows]
    result_path = neb / "yamatsumi_result.txt"
    res = _result_fields(result_path.read_text()) if result_path.exists() else {}
    data.update(iterations=len(rows), final_max_force=max_f[-1] if max_f else None)

    crit["A1_converged"] = bool(
        result_path.exists() and max_f and max_f[-1] < args.force_tol
    )

    outcars = sorted(work.rglob("OUTCAR"))
    problems = {str(p.relative_to(work)): scf_convergence_problem(p) for p in outcars}
    data["outcars_checked"] = len(outcars)
    data["scf_problems"] = {k: v for k, v in problems.items() if v}
    crit["A2_all_scf_converged"] = not data["scf_problems"]

    image_dirs = get_image_dirs(neb, NUM_IMAGES)
    e0 = parse_energy_from_outcar(image_dirs[0] / "OUTCAR")
    e1 = parse_energy_from_outcar(image_dirs[-1] / "OUTCAR")
    data["endpoint_energy_difference_meV"] = abs(e0 - e1) * 1000.0
    crit["A3_endpoints_degenerate"] = data["endpoint_energy_difference_meV"] < 1.0

    init, final = (
        read_poscar(str(work / "POSCAR_init")),
        read_poscar(str(work / "POSCAR_final")),
    )
    d = direct_to_cartesian(
        mic_displacement(init.positions, final.positions), init.lattice
    )
    k = int(np.argmax(np.linalg.norm(d, axis=1)))
    ts_path = neb / "TS_POSCAR"
    if ts_path.exists():
        ts = read_poscar(str(ts_path))
        midpoint = init.positions[k] + 0.5 * mic_displacement(
            init.positions[k], final.positions[k]
        )
        off = direct_to_cartesian(
            mic_displacement(midpoint, ts.positions[k]), init.lattice
        )
        data["migrating_atom"] = k + 1
        data["ts_distance_from_bond_midpoint_Ang"] = float(np.linalg.norm(off))
        crit["A4_ts_at_bond_midpoint"] = (
            data["ts_distance_from_bond_midpoint_Ang"] < 0.02
        )
    else:
        crit["A4_ts_at_bond_midpoint"] = False

    n_imag = int(res.get("Imaginary frequencies found", -1))
    n_zero = next(
        (int(v) for key, v in res.items() if key.startswith("Zero modes")), -1
    )
    data.update(n_imaginary=n_imag, n_zero_modes=n_zero)
    crit["A5_one_imaginary_three_zero"] = n_imag == 1 and n_zero == 3

    alignment = (
        float(res["Eigenvector-tangent alignment"].split()[0])
        if "Eigenvector-tangent alignment" in res
        else None
    )
    data["eigenvector_tangent_alignment"] = alignment
    crit["A6_alignment"] = alignment is not None and alignment >= 0.9

    data["iteration_numbers_continuous"] = iters == list(range(1, len(iters) + 1))
    data["resume_exercised"] = len(iters) > args.interrupt_after
    crit["A7_resume"] = bool(
        data["iteration_numbers_continuous"]
        and data["resume_exercised"]
        and all(
            crit[c]
            for c in (
                "A1_converged",
                "A3_endpoints_degenerate",
                "A4_ts_at_bond_midpoint",
                "A5_one_imaginary_three_zero",
                "A6_alignment",
            )
        )
    )

    ts_per_atom = []
    for idir in image_dirs:
        lines = [
            ln
            for ln in (idir / "OUTCAR").read_text(errors="replace").splitlines()
            if "EENTRO" in ln
        ]
        ts_per_atom.append(abs(float(lines[-1].split("=")[1])) * 1000.0 / N_ATOMS)
    data["max_abs_TS_meV_per_atom"] = max(ts_per_atom)
    crit["A8_entropy_small"] = data["max_abs_TS_meV_per_atom"] < 1.0

    # ADR 0006 calibration data: the three smallest |nu| are the translations.
    freqs = sorted(res.get("frequencies", []), key=abs)
    if len(freqs) >= 5:
        acoustic = max(abs(f) for f in freqs[:3])
        real = [f for f in freqs[3:] if f > 0]
        imag = [f for f in freqs[3:] if f < 0]
        proposal = float(np.ceil(3.0 * acoustic * 10.0) / 10.0)
        data["zero_mode_calibration"] = dict(
            largest_translational_abs_meV=acoustic,
            smallest_real_mode_meV=min(real) if real else None,
            imaginary_modes_meV=imag,
            proposed_zero_mode_tol_meV=proposal,
            proposal_below_smallest_real_mode=bool(real and proposal < min(real)),
            proposal_below_imaginary_magnitude=bool(
                imag and proposal < min(abs(f) for f in imag)
            ),
        )

    data["barriers_eV"] = dict(
        forward=res.get("Forward  activation barrier"),
        backward=res.get("Backward activation barrier"),
    )
    report["passed"] = all(crit.values())
    return report


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--vasp-cmd",
        default=None,
        help="VASP command (default with --env-script: 'mpirun -np 4 $VASP_BIN/vasp_std')",
    )
    p.add_argument(
        "--env-script",
        default=None,
        help="Shell script sourced (by zsh) before every VASP run, e.g. ~/vasp/vasp661_env.sh",
    )
    p.add_argument(
        "--potcar-dir",
        default=os.environ.get("VASP_POTCAR_DIR"),
        help="PAW library holding Si/POTCAR (default: $VASP_POTCAR_DIR)",
    )
    p.add_argument(
        "--work-dir",
        default=str(Path(__file__).parent / "run"),
        help="Where every run directory goes (default: ./run next to this script, gitignored)",
    )
    p.add_argument(
        "--force-tol",
        type=float,
        default=0.02,
        help="A1 tolerance in eV/Ang (default: 0.02)",
    )
    p.add_argument(
        "--max-iter", type=int, default=300, help="NEB iteration budget (default: 300)"
    )
    p.add_argument(
        "--interrupt-after",
        type=int,
        default=10,
        help="Iterations before the deliberate interruption tested by A7 (default: 10)",
    )
    p.add_argument(
        "--evaluate-only", action="store_true", help="Only evaluate an existing run"
    )
    args = p.parse_args()

    work = Path(args.work_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    if not args.evaluate_only:
        potcar_dir = args.potcar_dir or (
            None
            if args.env_script is None
            else subprocess.run(
                [
                    "/bin/zsh",
                    "-c",
                    f"source {shlex.quote(str(Path(args.env_script).expanduser()))} "
                    '&& printf %s "$VASP_POTCAR_DIR"',
                ],
                capture_output=True,
                text=True,
                check=True,
            ).stdout
        )
        if not potcar_dir:
            raise SystemExit("Give --potcar-dir or set VASP_POTCAR_DIR.")
        potcar = Path(potcar_dir).expanduser() / "Si" / "POTCAR"
        if not potcar.is_file():
            raise SystemExit(f"No POTCAR at {potcar}")
        titel = next(
            (
                ln.split("=", 1)[1].strip()
                for ln in potcar.read_text().splitlines()
                if "TITEL" in ln
            ),
            "?",
        )
        log(f"POTCAR: {titel}  (copied into run directories only)")
        cmd = vasp_command(args)
        log(f"VASP command: {cmd}")
        prepare(work)
        relax(work, cmd, potcar)
        run_neb(work, cmd, potcar, args)

    report = evaluate(work, args)
    (work / "validation_report.json").write_text(json.dumps(report, indent=2))
    for name, ok in report["criteria"].items():
        log(f"{'PASS' if ok else 'FAIL'}  {name}")
    log(
        f"overall: {'PASS' if report['passed'] else 'FAIL'}   (details: {work / 'validation_report.json'})"
    )


if __name__ == "__main__":
    main()
