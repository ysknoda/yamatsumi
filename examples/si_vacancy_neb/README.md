# Si64 vacancy-hop CI-NEB test case

A single-vacancy nearest-neighbour hop in bulk diamond-cubic silicon, set
up as a `yamatsumi` CI-NEB test calculation against **real VASP** (unlike
`examples/muller_brown_demo/`, which needs no VASP at all). This is the
semiconductor analogue of the FCC-metal vacancy-hop procedure described in
`docs/real_vasp_validation.md`.

## What has been generated

Running `make_endpoints.py` produced:

```
POSCAR_init          -- vacancy at lattice site 56
POSCAR_final         -- vacancy at lattice site 15 (a <111> nearest neighbour of 56)
CopyDir/INCAR         -- single-point evaluator settings for the NEB run itself
CopyDir/KPOINTS       -- 2x2x2 Gamma-centred mesh
```

Structure: a 2x2x2 supercell of the 8-atom diamond-cubic conventional cell
(space group Fd-3m, Wyckoff 8a), i.e. 64 lattice sites / 63 atoms after the
vacancy, with the experimental Si lattice constant a = 5.431 Ang (300 K;
[Ioffe NSM Archive](https://www.ioffe.ru/SVA/NSM/Semicond/Si/basic.html)) --
supercell edge 10.862 Ang, cubic.

The generator script verified, before writing anything:
- the two chosen sites are a genuine tetrahedral nearest-neighbour pair
  (measured MIC distance 2.351692 Ang, exactly matching the analytic
  diamond-cubic value a*sqrt(3)/4 = 2.351692 Ang -- the well-known Si-Si
  bond length);
- exactly one atom differs between the two endpoints, by exactly that
  distance;
- `yamatsumi.interpolate.check_endpoint_consistency` accepts both files;
- both files round-trip through `read_poscar`/`write_poscar` unchanged.

Separately (not part of the script, done once while preparing this
example), a dry `--phase init --dry-run` through the real `yamatsumi` CLI
confirmed Phase 1 accepts these endpoints and interpolates 5 intermediate
images with no complaint, and the minimum pairwise interatomic distance
stays pinned at 2.3517 Ang across every interpolated image (i.e. plain
linear interpolation does not route the migrating atom through another
atom here -- `--path-interpolation idpp` is available but not needed for
this single-atom hop).

**These are ideal, unrelaxed "bulk lattice minus one atom" structures.**
Real forces on them are of order eV/Ang, far above what a CI-NEB run
expects of its endpoints -- do not skip Step 1 below.

## Step 1 -- relax each endpoint independently

`yamatsumi` interpolates and optimises the path *between* whatever
endpoints it is given; it does not relax them first (see the repository
README). Relax `POSCAR_init` and `POSCAR_final` **separately**, each as an
ordinary static point-defect calculation, with the supercell **volume and
shape fixed** (only atomic positions relax) -- this is the standard
fixed-volume convention for point-defect calculations, so that the
computed migration barrier is not contaminated by a spurious
pressure/elastic-relaxation contribution from letting a single defective
cell find its own volume:

```
ISIF   = 2      # relax ions only; keep the cell (lattice constant) fixed
IBRION = 2      # or 3; either is fine for a simple ionic relaxation
NSW    = 100
EDIFFG = -0.01  # eV/Ang -- tight enough that residual forces are small
                # relative to the NEB force-convergence threshold you plan
                # to use (--force-tol, default 0.05 eV/Ang)
```
(plus the same `ENCUT`/`PREC`/`ISMEAR`/`SIGMA`/`ISPIN` lines as
`CopyDir/INCAR` -- everything except the `NSW`/`IBRION`/`ISIF` block above,
which `CopyDir/INCAR` deliberately keeps at the single-point values
`yamatsumi` itself requires for the NEB step). Overwrite `POSCAR_init` /
`POSCAR_final` with each relaxation's `CONTCAR` once done.
VASP writes `CONTCAR` with fractional coordinates wrapped into [0, 1), so an
atom near a cell face can come back on the opposite face. yamatsumi unwraps
the final endpoint against the initial one when it reads them
([ADR 0001](../../docs/adr/0001-fractional-coordinates-are-never-wrapped.md)),
so the CONTCARs can be used as they are.

## Step 2 -- supply your own POTCAR

`CopyDir/POTCAR` is licensed VASP pseudopotential data and has
deliberately **not** been generated here. Copy your own Si PAW-PBE POTCAR
into `CopyDir/POTCAR` (e.g. the standard `Si` or `Si_GW` dataset from your
VASP pseudopotential library) before running anything.

## Step 3 -- run yamatsumi

From the repository root, once Steps 1-2 are done:

```bash
yamatsumi --phase all --num-images 5 --optimizer fire \
       --init examples/si_vacancy_neb/POSCAR_init \
       --final examples/si_vacancy_neb/POSCAR_final \
       --copy-dir examples/si_vacancy_neb/CopyDir \
       --work-dir examples/si_vacancy_neb/NEB_run \
       --vasp-cmd "mpirun -np <N> vasp_std"
```

Increase `--num-images` (7-9) if the barrier region looks under-resolved.

## What to check afterward

Same criteria as `docs/real_vasp_validation.md`:
- A single smooth barrier between the two (degenerate, for this
  monatomic vacancy hop) end states in `NEB_run/energy_profile.png` /
  `yamatsumi_result.txt`, with `max|F_NEB|` converged below `--force-tol`.
- `yamatsumi_result.txt` should report **exactly 1** imaginary phonon
  frequency at the climbing image.
- The forward barrier: experimentally-derived Si vacancy migration
  energies commonly cited in the literature (Watkins- and Voronkov-style
  EPR/diffusion analyses) span roughly **0.3-0.45 eV**, though the precise
  value is charge-state- and method-dependent and this is only a very
  rough sanity-check magnitude -- see the confidence note below.
- `Eigenvector-tangent alignment` close to 1.0 (see the repository
  README's "Reaction-path analysis features" section).

## Parameter sourcing and self-critique

| Parameter | Value | Source | Confidence |
|---|---|---|---|
| Si lattice constant | 5.431 Ang (300 K) | [Ioffe NSM Archive](https://www.ioffe.ru/SVA/NSM/Semicond/Si/basic.html); agrees with standard textbook value (Kittel) | High |
| Diamond-cubic basis (8a Wyckoff site) | 8 atoms per conventional cell, coords as in `make_endpoints.py` | Standard Fd-3m diamond structure; reproduces the textbook Si-Si bond length exactly (see below) | High |
| Nearest-neighbour (Si-Si bond) distance | a*sqrt(3)/4 = 2.352 Ang | Analytic geometry of the basis above; matches the well-known Si-Si bond length | High |
| Supercell size | 2x2x2 (Si64, 63 atoms with the vacancy) | Common size for point-defect *methodology* testing in the literature | Medium -- adequate for a test/workflow check; production formation/migration energies should be checked against a larger cell (216+/512 atoms) for finite-size convergence (periodic images of the vacancy interact elastically at this size) |
| Migration mechanism | Direct nearest-neighbour atomic hop between the two ideal vacancy configurations | Standard, simplest first-approximation NEB setup for vacancy migration (used analogously for FCC metals in `docs/real_vasp_validation.md`) | Medium -- the Si vacancy's true (possibly Jahn-Teller-distorted) ground-state symmetry and whether migration proceeds via a more complex bond-rearrangement path is a separately studied topic; this direct-hop path is the right starting point for a first NEB test, not necessarily the final word on the mechanism |
| ENCUT = 400 eV | `CopyDir/INCAR` | Standard "Si" PAW-PBE POTCAR has ENMAX = 245.345 eV; 400 eV is a commonly used, generously converged value in the Si point-defect literature | **UNCERTAIN** -- not checked against your actual POTCAR's ENMAX (this sandbox has no VASP/POTCAR access); verify against your POTCAR header and/or run an ENCUT convergence test |
| KPOINTS 2x2x2 Gamma-centred | `CopyDir/KPOINTS` | Reasonable test-level density for a 10.86 Ang cubic supercell (larger real-space cell needs proportionally fewer k-points than the primitive cell) | Medium -- adequate for a workflow test; do a k-point convergence test before trusting absolute barrier heights |
| ISMEAR=0, SIGMA=0.05 | `CopyDir/INCAR` | Standard safe choice for a semiconductor (finite gap) with a coarse k-mesh, where the tetrahedron method is unreliable | High |
| ISPIN=1 (non-spin-polarised) | `CopyDir/INCAR` | Simplifying assumption for a neutral-vacancy test calculation | Medium -- the neutral Si vacancy's spin state is charge-state- and Jahn-Teller-dependent (Watkins-model literature); re-check with ISPIN=2 if you need charge-state-resolved defect physics |
| Migration-barrier sanity range 0.3-0.45 eV | README "What to check" | Web-search-corroborated EPR/diffusion literature values (Watkins ~0.45 eV, Voronkov et al. ~0.38 eV); a DFT-GGA NEB value for the *neutral* vacancy along *this specific path* was not independently located | **UNCERTAIN** -- treat as an order-of-magnitude sanity check only, exactly as `docs/real_vasp_validation.md` already recommends for the FCC-metal case |

## Regenerating with different parameters

```bash
python3 make_endpoints.py --supercell-size 3 --vacancy-site 0
```

See `python3 make_endpoints.py -h` for all options (supercell size,
lattice constant, element label, which site to vacate, output directory).
Re-running overwrites `POSCAR_init` / `POSCAR_final` in place.
