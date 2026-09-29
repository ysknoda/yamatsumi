# Real-VASP validation procedure

The automated test suite (`tests/`) validates `yamatsumi`'s logic and the
CI-NEB algorithm against analytic potential surfaces through a mock VASP
executable, and its OUTCAR parsers against byte-for-byte excerpts of real
VASP output (`tests/fixtures/real_vasp_excerpts/`, see
[ADR 0007](adr/0007-real-vasp-output-enters-the-repo-only-as-excerpts.md)).
That catches regressions in the code, but not problems that only a real
DFT potential energy surface exposes. This document covers the end-to-end
check against real VASP.

## Stage (a): automated software validation on Si7

`examples/si7_vacancy_vasp/run_validation.py` runs the whole pipeline on a
single vacancy hop in the 8-site conventional diamond Si cell (7 atoms):
endpoint generation, endpoint relaxation with VASP, a CI-NEB run that is
deliberately interrupted and then resumed, the finite-difference phonon
check, and an automatic evaluation. The cell is far too small for a
physically meaningful barrier. It is used because the two endpoints are
related by the inversion centre at the midpoint of the hopping bond, which
predicts several results exactly without any reference data.

Acceptance criteria, fixed before the first run and not to be relaxed
afterwards to make a run pass:

| # | Criterion | Threshold | Why |
|---|---|---|---|
| A1 | NEB converges | max\|F_NEB\| < 0.02 eV/Ang within 300 iterations | Tight enough for A4 to be meaningful |
| A2 | Every VASP run reaches EDIFF | no exceptions | [ADR 0003](adr/0003-unconverged-scf-is-a-hard-failure.md); yamatsumi stops otherwise, and every OUTCAR is re-checked |
| A3 | Endpoints are degenerate | \|E_init - E_final\| < 1 meV | Symmetry-equivalent; only the relaxation residual differs |
| A4 | TS at the bond midpoint | migrating atom within 0.02 Ang (UNCERTAIN) | Inversion centre |
| A5 | Imaginary modes | exactly 1, plus exactly 3 zero modes | First-order saddle; translational invariance ([ADR 0006](adr/0006-imaginary-mode-counting-uses-a-zero-mode-threshold.md)) |
| A6 | Eigenvector-tangent alignment | >= 0.9 (UNCERTAIN) | Unstable direction matches the intended path |
| A7 | Resume | stopped after 10 iterations, finished with `--resume`; continuous iteration numbering and A1-A6 still met | `--resume` on real VASP |
| A8 | Smearing entropy | \|T*S\| < 1 meV/atom in every image | VASP Wiki, "Smearing technique" |

A4 and A6 are marked UNCERTAIN (below 90% confidence): how close the
climbing image gets to the saddle at a 0.02 eV/Ang force tolerance depends
on the curvature there, which is unknown beforehand. A miss is reported
and explained, never tuned away.

Settings (`run_validation.py` holds the exact INCAR): PAW_PBE Si 05Jan2001,
ENCUT = 400 eV (1.63 x ENMAX 245.345 eV, read from the POTCAR), PREC =
Accurate, EDIFF = 1E-6, ISMEAR = 0 with SIGMA = 0.02, LREAL = .FALSE. (7
atoms; the VASP Wiki recommends real-space projection only above about 30),
ISYM = 0 so that symmetry has to come out of the calculation rather than be
imposed, NELM = 100, LWAVE = .TRUE. so that each image restarts from its
previous wavefunction, and a Gamma-centred 4x4x4 mesh (about 0.29 1/Ang,
the density of 2x2x2 on the 64-site cell). SIGMA was lowered from 0.05
before the first run, because the ideal geometry already gave
\|T*S\| = 0.92 meV/atom, close to the A8 limit. Endpoints are relaxed at
fixed cell with IBRION = 2, ISIF = 2, EDIFFG = -0.005 eV/Ang. The NEB uses 4
intermediate images, an even number, so that no image starts on the
symmetric saddle and the climbing image really has to climb.

How to run on the machine used here (Apple M4, 4 performance cores, VASP
6.6.1 built against Homebrew Open MPI):

```bash
nohup caffeinate -i python3 examples/si7_vacancy_vasp/run_validation.py \
    --env-script ~/vasp/vasp661_env.sh > si7_validation.log 2>&1 &
```

The yamatsumi driver itself runs without MPI; only VASP runs under
`mpirun -np 4`. The driver's mpi4py is linked against a different Open MPI
build (conda 5.0.8) from the one VASP uses (Homebrew 5.0.11), so running
the driver under `mpirun` as well would nest two MPI stacks for no gain.
The script sources the environment script before every VASP call,
because that script sets `OMP_NUM_THREADS = OPENBLAS_NUM_THREADS = 1`
(otherwise OpenBLAS threads oversubscribe the MPI ranks) but does not put
`$VASP_BIN` on `PATH`. Each stage is checkpointed, so rerunning the command
continues an interrupted run.

### Stage (a) results

Pending: the first run started on 2026-09-29.

## Physics check on your own system: vacancy migration in a bulk FCC metal

A single vacancy hopping to a neighbouring lattice site (equivalently, the
neighbouring atom hopping into the vacancy) is one of the simplest, most
commonly used validation systems for NEB implementations: small unit cell,
a single migrating atom, no surface-relaxation subtleties, and DFT-GGA
migration barriers for common FCC metals (Cu, Al, Ni, Ag, ...) are widely
published, so a rough sanity-check number is easy to find in the
literature for whichever metal you use.

1. **Build the endpoints** (e.g. with ASE or pymatgen, both already used
   elsewhere in this ecosystem):
   - A bulk FCC supercell (3x3x3 or larger; too small a cell will show
     spurious self-interaction between periodic images of the vacancy).
   - `POSCAR_init`: remove one atom to create the vacancy, at site A.
   - `POSCAR_final`: same supercell, but the vacancy is at a
     nearest-neighbour site B instead (i.e. the atom that was at B in the
     initial structure is now at A, and vice versa for the vacancy).
   - Relax both endpoints fully (ISIF=2 or 3, tight EDIFFG) *before*
     handing them to `yamatsumi` -- `yamatsumi` interpolates and optimises the
     path between whatever endpoints it is given; it does not relax them
     first.

2. **Prepare `CopyDir`** with `INCAR`, `KPOINTS`, `POTCAR` for a
   single-point calculation (no ionic relaxation flags -- `yamatsumi` moves
   the atoms itself between VASP calls). A reasonable starting `INCAR`:
   ```
   ISTART = 0
   ICHARG = 2
   ENCUT  = <converged for your POTCAR>
   EDIFF  = 1E-6
   ISMEAR = 1 ; SIGMA = 0.2   # or 0 for insulators
   NSW    = 0                 # single-point only; yamatsumi drives the ions
   LREAL  = Auto
   ```

3. **Run**:
   ```bash
   yamatsumi --phase all --num-images 5 --optimizer fire \
          --vasp-cmd "mpirun -np <N> vasp_std" \
          --init POSCAR_init --final POSCAR_final --work-dir NEB_run
   ```
   Increase `--num-images` (7-9) if the energy profile looks
   under-resolved near the barrier top.

4. **What to check**:
   - The energy profile (`NEB_run/energy_profile.png`,
     `NEB_run/yamatsumi_result.txt`) should show a single smooth barrier
     between the two (near-degenerate, for a monatomic vacancy hop) end
     states, with `max|F_NEB|` converging below `--force-tol`.
   - The transition-state phonon check
     (`Imaginary frequencies found` in `yamatsumi_result.txt`) should report
     **exactly 1** imaginary frequency. More than one usually means the
     path has not fully relaxed onto the true minimum-energy path in the
     directions perpendicular to it (try more images, a tighter
     `--force-tol`, or check `k`-point/`ENCUT` convergence); zero usually
     means the climbing image has not yet reached the true saddle.
   - The forward activation barrier should be in the right *ballpark* for
     your metal and functional (commonly cited GGA vacancy-migration
     barriers are roughly 0.6-1.0 eV for FCC transition/noble metals such
     as Cu, Ni, Al, Ag) -- treat this as a sanity-check magnitude, not an
     exact target: the precise value depends on your functional, POTCAR,
     supercell size, and k-point mesh, all of which you control
     independently of `yamatsumi`.
   - When exactly one imaginary frequency is found, `yamatsumi_result.txt`
     also reports `Eigenvector-tangent alignment` (the imaginary mode's
     displacement pattern compared against the NEB tangent at the climbing
     image; 1.0 = perfectly aligned). It should be reasonably close to 1.0
     for a well-resolved path; a low value despite a single imaginary
     frequency suggests the saddle's unstable direction points somewhere
     other than the intended vacancy-hop coordinate (e.g. an
     under-resolved path, or a nearby but different saddle).
     The eigenvector parser has been checked against a real VASP 6.4.3
     OUTCAR (24 modes, all eigenvector norms 1 to 1e-6). The same file
     showed that VASP labels the three translational modes of a periodic
     cell `f/i`, which is why modes with |nu| up to `--zero-mode-tol` are
     reported as zero modes rather than imaginary modes. Check that a full
     Hessian reports exactly three of them.

## Cross-checking MPI-parallel image dispatch against serial execution

If you plan to run this driver itself under `mpirun` (i.e.
`mpirun -n K python -m yamatsumi ...`, distributing NEB images across ranks;
see README), check once on your own cluster that it gives the same results
as a plain serial run. `tests/test_mpi_integration.py` does this with the
mock VASP whenever `mpirun` and mpi4py are available. On the machine used
for stage (a) it runs, and it is how a race between ranks reading and
rewriting image files was found and fixed. It cannot check your cluster's
filesystem, though:

1. Run the reference system above once serially (as in step 3).
2. Re-run it (fresh `--work-dir`) with `mpirun -n <num_images> python -m
   yamatsumi ...` and the same CLI flags otherwise.
3. Compare `NEB_run/convergence_history.csv` and the final barrier in
   `yamatsumi_result.txt` between the two runs -- they should match to
   floating-point precision (same number of iterations, same energies).
   A mismatch would indicate either a real bug (please report it) or that
   your cluster's filesystem is not consistently shared across the ranks
   `yamatsumi` was launched on (`yamatsumi` assumes every rank can read every
   other rank's freshly-written POSCAR/OUTCAR files without a caching
   delay -- true for a single workstation and for essentially all HPC
   parallel filesystems, but worth confirming on unfamiliar hardware).
