# Real-VASP validation procedure

The automated test suite (`tests/`) validates `yamatsumi`'s logic and the
CI-NEB algorithm against analytic potential surfaces through a mock VASP
executable, and its OUTCAR parsers against byte-for-byte excerpts of real
VASP output (`tests/fixtures/real_vasp_excerpts/`, see
[ADR 0007](adr/0007-real-vasp-output-enters-the-repo-only-as-excerpts.md)).
That catches regressions in the code, but not problems that only a real
DFT potential energy surface exposes. This document covers the end-to-end
check against real VASP.

> **Run one VASP job at a time.** Twice during stage (b) a VASP run ended
> with exit code 1 and no message the moment another session started its
> own `mpirun -np 4` on the same machine. Whether two concurrent Open MPI
> runs interfere is UNCERTAIN, but the coincidence was exact both times.

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
| A1 | NEB converges | max\|F_NEB\| < 0.02 eV/Ang within 300 iterations | Tight enough for A5 and A6 to be meaningful |
| A2 | Every VASP run reaches EDIFF | no exceptions | [ADR 0003](adr/0003-unconverged-scf-is-a-hard-failure.md); yamatsumi stops otherwise, and every OUTCAR is re-checked |
| A3 | Endpoints are degenerate | \|E_init - E_final\| < 1 meV | Symmetry-equivalent; only the relaxation residual differs |
| A4 | Below the symmetric saddle | forward barrier < 1.3013 eV (replaced before attempt 4; see below) | The first-order saddle lies off the hop axis, below the three-imaginary-mode saddle of attempt 3 |
| A5 | Imaginary modes | exactly 1, plus exactly 3 zero modes | First-order saddle; translational invariance ([ADR 0006](adr/0006-imaginary-mode-counting-uses-a-zero-mode-threshold.md)) |
| A6 | Eigenvector-tangent alignment | >= 0.9 (UNCERTAIN) | Unstable direction matches the intended path |
| A7 | Resume | stopped after 10 iterations, finished with `--resume`; continuous iteration numbering and A1-A6 still met | `--resume` on real VASP |
| A8 | Smearing entropy | \|T*S\| < 1 meV/atom in every image | VASP Wiki, "Smearing technique" |

A6 is marked UNCERTAIN (below 90% confidence): how close the climbing
image gets to the saddle at a 0.02 eV/Ang force tolerance depends on the
curvature there, which is unknown beforehand. A miss is reported and
explained, never tuned away.

**A4 was replaced once, on 2026-09-29 at 22:37, before attempt 4 had
produced any result, with the user's approval.** It used to require the
migrating atom within 0.02 Ang of the bond midpoint, on the assumption that
the inversion centre is the saddle. Attempt 3 showed that point, with the
path held on the hop axis, to be a saddle with three imaginary modes. The
first-order saddle lies off the axis: either on a C2 axis through the
bond midpoint (equidistant from both sites, but off the midpoint) or as a
pair of inversion-related saddles with a minimum between them. Either way
it is not at the midpoint, and it lies below the symmetric saddle. The TS
position relative to the midpoint, the bisecting plane and the hop axis is
still reported, as data only. Attempts 1-3 were judged under the old A4.

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

VASP 6.6.1, 4 MPI ranks, 2026-09-29. Run directories stay local
([ADR 0007](adr/0007-real-vasp-output-enters-the-repo-only-as-excerpts.md)).

**Attempt 1: stopped on the first NEB iteration.** The read-only POTCAR was
copied with its permission bits and could not be overwritten. Fixed in
`yamatsumi/directories.py`, which now copies file contents only.

**Attempt 2: diverged, stopped by hand.** The unrelaxed-symmetry (T_d)
endpoints were only a shallow metastable state. The end images fell 0.49 eV
into a C3v minimum, and the climbing image ran away to +1.95 eV. The
endpoints are now built from a relaxation with the migrating atom kicked
0.1 Ang along the hop, and the final one as its inversion image.

**Attempt 3: converged to a higher-order saddle.** 68 iterations, max|F|
0.019 eV/Ang, interrupted after 10 and finished with `--resume`.

| criterion | result | value |
|---|---|---|
| A1 converged | PASS | 0.0190 < 0.02 eV/Ang |
| A2 all SCF converged | PASS | 8 OUTCARs (first reported FAIL: the check read an archived attempt-2 OUTCAR; fixed) |
| A3 endpoints degenerate | PASS | 1e-5 meV |
| A4 (old) TS at bond midpoint | PASS | 0.0022 Ang (under the new A4 it fails: 1.3013 eV is the bound itself) |
| A5 one imaginary, three zero | **FAIL** | 3 imaginary, 3 zero |
| A6 alignment >= 0.9 | **FAIL** | not computed (needs exactly one imaginary mode) |
| A7 resume | **FAIL** | numbering continuous and resume exercised, but A5/A6 failed |
| A8 |T*S| < 1 meV/atom | PASS | 0.25 meV/atom |

The whole path stayed on the <111> hop axis: the migrating atom had x = y = z
in every image. Inversion-related endpoints give a C3v-symmetric
interpolated path, and the NEB forces keep that symmetry, so CI-NEB reached
the symmetric stationary point and nothing else. Its modes:

| imaginary mode | |nu| (meV) | overlap with the path tangent (computed offline) |
|---|---|---|
| reaction mode | 32.45 | 0.9717 |
| degenerate pair | 12.78, 12.78 | 0.0000 |

yamatsumi counted and rejected this saddle correctly. The phonon section is
kept as a regression fixture. The barrier of 1.30 eV is an upper bound for
this cell, and has no physical meaning at one vacancy per 8 sites in any
case.

**Attempt 4: PASS on all criteria.** The same endpoints; the intermediate
images were displaced by a seeded random vector of at most 0.05 Ang before
the first VASP call, which moved the migrating atom 0.01 to 0.03 Ang off
the axis. 122 iterations, interrupted after 10 and finished with
`--resume`. The machine slept for 3 h 44 min on an empty battery during
iteration 85; VASP resumed and finished that image normally. Judged by
`--evaluate-only` under the replaced A4 (the running driver still held the
old script).

| criterion | result | value |
|---|---|---|
| A1 converged | PASS | 0.0185 < 0.02 eV/Ang |
| A2 all SCF converged | PASS | 8 OUTCARs |
| A3 endpoints degenerate | PASS | 1e-5 meV |
| A4 below the symmetric saddle | PASS | 1.1727 < 1.3013 eV |
| A5 one imaginary, three zero | PASS | 1 imaginary (21.22 meV), 3 zero |
| A6 alignment >= 0.9 | PASS | 0.9566 |
| A7 resume | PASS | continuous numbering, A1-A6 met |
| A8 |T*S| < 1 meV/atom | PASS | 0.14 meV/atom |

Around iteration 61 the path passed within 0.036 eV/Ang of the symmetric
saddle (+1.3008 eV), then left it along the unstable pair and settled
0.13 eV lower. The TS lies 0.327 Ang off the hop axis and 0.074 Ang from
the plane bisecting the bond (data only). So it is not on a C2 axis
through the midpoint, which suggests the pair of inversion-related saddles
with a minimum between them, case (ii) above. Four images cannot resolve
that minimum; this is UNCERTAIN and not part of the software check.

Zero modes at this TS: largest translational 0.097 meV, smallest real mode
5.51 meV, imaginary mode 21.22 meV. The 1.0 meV default still separates
them, with margins of about 10 and 5.5.

## Stage (b): H2 dissociation on Cu(111) against SBH17

Design, reference data and every change made along the way:
[ADR 0009](adr/0009-stage-b-validates-against-sbh17-h2-cu111-with-a-frozen-surface.md).
Script: `examples/h2_cu111_vasp/run_validation.py`. In short: PBE, 2x2x4
Cu(111) slab with every Cu atom fixed, only the six H coordinates move,
H2 starting 6 Ang above a bridge site, 5 intermediate images, compared with
the SBH17 PBE "high" barrier of 0.478 eV (J. Chem. Theory Comput. 2023, 19,
245, Table 5). VASP 6.6.1, 11x11x1 k-points, 450 eV.

### Stage (b) results

**PASS on all criteria** (final, 2026-10-07), after the FIRE fix of ADR 0002
amendment 2. An earlier verdict of this run, B1 and B7 failing because the
gas-phase images never settled, is kept in the history below.

| criterion | result | value |
|---|---|---|
| B1 NEB converged | PASS | max\|F\| 0.0199 eV/Ang at iteration 160 |
| B2 all SCF converged | PASS | every completed run reached EDIFF |
| B3 1 imaginary, 0 zero modes | PASS | 6-mode partial Hessian: one imaginary mode at 75.72 meV, no zero mode |
| B4 alignment >= 0.9 | PASS | 0.9776 |
| B5 \|E_NEB - (0.478 + D)\| <= 0.03 eV | PASS | 0.4718 eV against 0.4786 eV: -0.0068 eV |
| B6 TS geometry (data) | - | bridge (0.014 Ang off), H2 parallel (0.5 deg), r = 0.980 Ang, Z = 1.211 Ang (SRP43: 1.03, 1.16) |
| B7 resume after a kill inside VASP | PASS | killed at iteration 11 (image 01), resumed; numbering 1-160 continuous |
| B8 \|T*S\| < 1 meV/atom | PASS | at most 0.38 meV/atom |

The backward barrier is 0.4801 eV; the 2H final state lies 8 meV below
the far H2.

How the run went:

1. Iterations 1-17, `--step-size` 0.05 Ang (the default): the path barely
   moved, at most 0.008 Ang per iteration (ADR 0009 Amendment 3).
2. Iterations 18-106, 0.2 Ang (the VTST MAXMOVE): the path came down from
   +2.5 eV to a climbing image at +0.4719 eV, steady to 2 meV from
   iteration 68 on, while max\|F\| on the other images swung between 0.2
   and 1.4 eV/Ang.
3. Iterations 107-125, 0.05 Ang again: max\|F\| fell steadily from 0.455 to
   0.157 eV/Ang at iteration 121, then rose again to 0.215, carried by
   images 01 and 02 at 4-5 Ang above the surface, where the energy is flat
   to a few meV and the tangent is poorly defined.
4. With the user's approval the run was stopped after about 20 iterations
   of step 3 and the climbing image was checked on its own
   (`--part ts-phonon`). The residual force on the H atoms at its geometry
   is 0.0018 eV/Ang.
   That check gave B3-B5 as above (76.07 meV, alignment 0.9774) with B1
   and B7 failing.
5. The FIRE velocity cap was then removed (ADR 0002 amendment 2) and the
   run resumed at iteration 126 with 0.05 Ang. max\|F\| fell from 0.228 to
   0.046 eV/Ang in 12 iterations, hovered at 0.05-0.065 for a few, and
   reached 0.0199 at iteration 160. yamatsumi's own phonon check then
   reproduced the climbing-image result.

Interrupted VASP runs: one image killed at iteration 11 on purpose (B7);
one at iteration 19 that ended with rc=1 and no message the moment another
session started its own `mpirun -np 4` (the same thing happened once during
the preparation; whether two concurrent Open MPI runs interfere is
UNCERTAIN); and iteration 126, stopped on purpose for the phonon check.
yamatsumi rejected the rc=1 run, and `--resume` re-ran every unfinished
image each time.

Open point: the gas-phase oscillation was never reproduced on 2D analytic
surfaces, so whether removing the velocity cap is what cured it, rather
than the extra iterations or the FIRE state reset at the resume, is
UNCERTAIN.

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
