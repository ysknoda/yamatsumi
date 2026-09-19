# Real-VASP validation procedure

The automated test suite (`tests/`) validates `yamatsumi`'s logic and the
CI-NEB algorithm itself against analytic potential surfaces (a mock VASP
executable), since no VASP license, binary, or POTCAR is available in the
development/CI environment. That is sufficient to catch regressions in the
code, but it does not exercise a real DFT potential energy surface. This
document describes a manual follow-up validation to run once, on your own
VASP/HPC environment, before relying on this tool for production
calculations.

## Recommended reference system: vacancy migration in a bulk FCC metal

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
     **Parser-verification note**: this diagnostic's OUTCAR eigenvector
     parsing (`yamatsumi.outcar.parse_eigenvectors_from_outcar`) was written
     from documented VASP OUTCAR conventions but only tested here against a
     synthetic fixture, not a real VASP OUTCAR -- on your first real run,
     check the log for a "Could not locate the imaginary mode's eigenvector"
     warning; if it appears, the column layout differs from what the parser
     expects and the alignment value should not be trusted until that is
     fixed.

## Cross-checking MPI-parallel image dispatch against serial execution

If you plan to run with this driver itself launched under `mpirun` (i.e.
`mpirun -n K python -m yamatsumi ...`, distributing NEB images across ranks --
see README), validate once on your own cluster that this gives identical
results to a plain serial run, since that consistency could not be tested
automatically here (no `mpirun`/working MPI runtime in the development
sandbox; see `tests/test_mpi_integration.py`, which self-skips for the same
reason):

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
