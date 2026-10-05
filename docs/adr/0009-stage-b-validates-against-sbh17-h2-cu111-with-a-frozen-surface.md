---
status: accepted
---

# Stage (b) validates against SBH17 H2 + Cu(111) with a frozen surface

Stage (b) of the real-VASP validation computes the barrier to dissociative
chemisorption of H2 on Cu(111) with CI-NEB and compares it with the SBH17
benchmark: T. Tchakoua, N. Gerrits, E. W. F. Smeets, G.-J. Kroes,
J. Chem. Theory Comput. 2023, 19, 245-270 (doi:10.1021/acs.jctc.2c00824).

## Reference value and its definition

- PBE, Table 5: high algorithm 0.478 eV, light 0.488, medium 0.467,
  literature (CRP fit) 0.484. The "high" value is the one to compare with:
  it comes from a TS search (dimer method, forces below 5 meV/Ang) with one
  imaginary frequency checked, as yamatsumi's CI-NEB plus phonon check does.
- Section II.4: the barrier is measured from the molecule far from the
  surface, and the metal is frozen in its relaxed 0 K geometry during the TS
  search; frequencies are checked in the molecule's coordinates only.
- Table S3 (supporting information), H2 + Cu(111), all three rows:

  | source | layers | cell | ENCUT (eV) | k-points | POTCAR | smearing | vacuum (Ang) |
  |---|---|---|---|---|---|---|---|
  | SBH10 | 6 | 2x2 | 350 | 4 | PAW | Gaussian | 21 |
  | SRP14 (reference PES) | 4 | 2x2 | 350 | 8 | USPP | Fermi | 13 |
  | SBH17 (the 0.478 eV) | 5 | 3x3 | 450 | 11 | PAW | Methfessel-Paxton | 13 |

  The k-point column holds one number; reading it as NxNx1 is UNCERTAIN.
  The smearing width is not given.
- PBE Cu lattice constant 3.624 Ang (Table 3). SRP43 TS geometry: bridge,
  r = 1.03 Ang, Z = 1.16 Ang, H2 parallel to the surface (Table 2). SBH17
  gives no PBE TS geometry for this system.

## Decisions

1. Same definition as "high": every Cu atom is fixed by Selective
   Dynamics, only the six H coordinates move. The initial state is H2 over
   a bridge site, parallel to the surface, at the computed PBE bond length;
   the final state is 2H put in the fcc and hcp hollows next to that bridge
   and relaxed with Cu fixed. If the relaxation moves them elsewhere, that
   minimum is used and reported.
2. Preparation: the PBE bulk lattice constant with our settings (used if
   within 0.01 Ang of 3.624 Ang), the interlayer relaxation of a 1x1 slab
   with the bottom layer fixed, and the H2 height: the initial state sits
   at 6 Ang (SBH17's asymptote) unless 4 Ang differs by less than 5 meV,
   in which case the NEB starts at 4 Ang.
3. Settings: 2x2 cell, 4 layers, standard PAW `Cu` (22Jun2005) and `H`
   (15Jun2001), ISMEAR = 1, SIGMA = 0.1, 13 Ang vacuum. k-points and cutoff
   are chosen by a scan of the medium-algorithm barrier (two single points,
   SRP43 TS geometry and the far H2) over k in {6, 8, 11} x NxNx1 and
   ENCUT in {350, 450} eV: the lightest setting within 10 meV of the
   densest one. The same two points in a 3x3x4 cell (4x4x1 k-points) measure
   the cell-size effect. One TS single point with IDIPOL = 3, LDIPOL =
   .TRUE. measures the dipole correction; below 5 meV it is left out.
4. Budget: if the chosen setting projects the NEB beyond 36 hours, stop
   before it and decide with the user. The 10 meV rule is not relaxed.
5. NEB: 5 intermediate images; the H atoms of the intermediate images get
   a seeded random displacement of at most 0.05 Ang, because the plane
   holding the H2 axis and the surface normal is a mirror plane and a path
   held in it can converge to a higher-order saddle (Si7 attempt 3).

## Acceptance criteria, fixed before the run

| # | criterion | threshold |
|---|---|---|
| B1 | NEB converges | max\|F\| < 0.02 eV/Ang on the H atoms within 300 iterations |
| B2 | every VASP run reaches EDIFF | no exceptions |
| B3 | imaginary and zero modes | 1 imaginary, 0 zero modes in the 6-mode partial Hessian |
| B4 | eigenvector-tangent alignment | >= 0.9 (UNCERTAIN) |
| B5 | barrier | \|E_NEB - (0.478 + D)\| <= 0.03 eV (UNCERTAIN) |
| B6 | TS geometry | data only |
| B7 | resume | killed inside a VASP call, finished with --resume, B1-B5 met |
| B8 | smearing entropy | \|T*S\| < 1 meV/atom in every image |

D is our medium-algorithm barrier at the production settings in the 2x2x4
cell minus SBH17's 0.467 eV. Both are single points at the same fixed
geometry, so D carries every difference of settings and cell, and B5
tests what remains: CI-NEB against the dimer-method TS. The 0.03 eV window
is an estimate. B6 is data only because no PBE TS geometry is published,
and a geometric criterion built on an assumed saddle failed in Si7 (ADR
0008).

## Considered Options

- Relax the upper Cu layers along the path. Rejected: the result could not
  be compared with SBH17, which freezes the surface.
- SBH17's own settings (3x3, 5 layers, 450 eV, 11 k-points). Rejected: a
  single point was estimated at tens of hours on this machine.
- Start H2 physisorbed near the surface. Rejected: PBE has no van der Waals
  term and gives at most a shallow well.

## Amendment 1 (2026-09-30): the lattice constant

Decision 2 said to use our PBE lattice constant if it lies within 0.01 Ang
of SBH17's 3.624 Ang. It came out at 3.6347 Ang (third-order
Birch-Murnaghan fit to E(a), 7 points, residuals below 0.02 meV), 0.0107
Ang away, and the preparation stopped there. The value does not move with
the settings:

| variant | a0 (Ang) |
|---|---|
| 21^3 k-points, MP sigma 0.1, 450 eV, standard Cu | 3.6347 |
| 29^3 k-points | 3.6354 |
| sigma 0.05 | 3.6345 |
| 600 eV | 3.6343 |
| tetrahedron method, 29^3 | 3.6353 |
| Cu_pv POTCAR | 3.6323 |

SBH17's Table 3 also lists the all-electron PBE value of Haas, Tran and
Blaha (Phys. Rev. B 79, 085104, 2009), 3.632 Ang, which ours matches to
0.003 Ang. SBH17's own 3.624 Ang is the one that stands apart; why is
UNCERTAIN. With the user's approval, after this result was known, the
check is now made against the all-electron value (still 0.01 Ang), and
3.6347 Ang is used. The lattice difference to SBH17 does not bias B5: D
is computed on our own slab and so absorbs it. From the strain dependence
of Sakong and Gross it is worth about 0.01 eV.

## Amendment 2 (2026-09-30): preparation results and the budget

Preparation (VASP 6.6.1), all decided rules applied as written:

| quantity | value |
|---|---|
| a0 | 3.6347 Ang (Amendment 1) |
| H2 bond length | 0.7506 Ang |
| d12 change of the 1x1 slab | -1.46 % (SBH17 PBE -0.26 %; data only, 4 layers) |

Medium-algorithm barrier (SRP43 TS geometry minus H2 at 6 Ang), 2x2x4:

| k-points | 350 eV | 450 eV |
|---|---|---|
| 6x6x1 | 0.5185 | 0.5320 |
| 8x8x1 | 0.3960 | 0.4095 |
| 11x11x1 | 0.4540 | 0.4676 |

- Only the densest setting lies within 10 meV of itself, so 11x11x1 and
  450 eV are used, and D = 0.4676 - 0.467 = +0.0006 eV. The cutoff shifts
  every row by the same 13.5 meV; the k-points make the barrier oscillate
  by up to 0.12 eV. Whether 11x11x1 is converged cannot be told from this
  scan, since the rule picks the densest point by construction
  (UNCERTAIN). B5 compares differences at the same setting, which absorbs
  part of that error.
- E(4 Ang) - E(6 Ang) = -5.39 meV, not below 5 meV: the NEB starts at 6 Ang.
- Dipole correction: +1.37 meV on the barrier, below 5 meV: left out.
- 3x3x4 (4x4x1) minus 2x2x4 (6x6x1) at 450 eV: -5.29 meV.
- Largest |T*S|: 0.11 meV/atom.

The NEB was projected at 72.4 h (52.1 s per SCF step, 100 iterations x
5 images x 10 steps), twice the 36 h budget. Following decision 4 the run
stopped, and the user accepted the 72 h projection (option (a); the
alternatives were 3 images, the lighter 8x8x1 setting, or starting at
4 Ang against the height rule). The NEB part now refuses to start over
budget unless --approved-hours covers the projection. The user has put
VASP runs on hold until further notice.

## Amendment 3 (2026-10-02): the NEB step size

The NEB started on 2026-10-02 01:03 with yamatsumi's default
`--step-size` of 0.05 Ang. B7 worked as designed: the run was killed inside
the VASP call of image 01 in iteration 11, and `--resume` re-ran images
01-05 (all had lost their OUTCAR) and continued. But the path barely moved:
after 10 iterations max|F| was still 4.5 eV/Ang, and no H atom was more
than 0.07 Ang from the initial linear path, 0.05 Ang of which is the
seeded perturbation. yamatsumi's FIRE caps the velocity at `--step-size`
and moves velocity x dt, so with dt starting at 0.1 an image moves at most
0.005-0.008 Ang per iteration until dt has grown. The images have to move
1-2 Ang from a linear path through a stretched H2 in vacuum (+2.5 eV).

VTST documents MAXMOVE, the role `--step-size` plays, as the maximum step
for translation, default 0.2 Ang. With the user's approval the run was
stopped at an iteration boundary and resumed from its current images with
`--step-size 0.2`; nothing else changed. Whether VTST also caps the
velocity, as yamatsumi does, is UNCERTAIN and is a separate question about
the optimizer (ADR 0002), not part of this validation.

## Amendment 4 (2026-10-06): the climbing image checked on its own

With `--step-size` 0.2 the climbing image settled at +0.4719 eV, but the
other images kept oscillating (max\|F\| 0.2-1.4 eV/Ang). The user approved,
first, resuming with 0.05 Ang for about 20 iterations and, if that did not
settle the path, checking the climbing image alone. After 19 iterations
max\|F\| had come down to 0.157 eV/Ang and risen again to 0.215, so the run
was stopped and `--part ts-phonon` ran IBRION=5 on the climbing image, with
the tangent rebuilt from the current images and the last logged energies.

Result: one imaginary mode (76.07 meV), no zero mode, alignment 0.9774,
residual H force 0.0018 eV/Ang, barrier 0.4719 eV against the target
0.4786 eV. B3, B4, B5 pass. B1 fails because the path as a whole did not
converge, and B7 fails with it, since it requires B1-B5. The criteria were
not changed. Full table: docs/real_vasp_validation.md, "Stage (b) results".
