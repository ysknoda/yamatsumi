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
