---
status: accepted, calibration pending
---

# Imaginary modes are counted above a zero-mode threshold

The transition-state test is "exactly one imaginary mode". A full
finite-difference Hessian of a periodic cell also has three translational
modes that are zero in exact arithmetic, and VASP prints them with arbitrary
sign. A real VASP 6.4.3 OUTCAR of perfect diamond Si reports all three as
`f/i` at 3e-6 meV, so the old count gave three imaginary modes on the most
stable structure possible, and the TS test could not pass on real output.

A mode now counts as imaginary only if its magnitude exceeds
`--zero-mode-tol`, and modes below it are reported separately as zero modes.
yamatsumi warns when a full Hessian gives a zero-mode count other than three.

## Considered Options

- Drop the three modes with the smallest magnitude. Rejected: a partial
  Hessian (Selective Dynamics) has no translational modes, so this would
  discard three real modes.

## Calibration

The default is provisionally 1.0 meV (about 8 cm^-1). The confidence in this
value is below 90%. The only real data point so far comes from a perfect
crystal, and a defective transition-state structure should give much larger
residuals. The default will be replaced by three times the largest
translational |nu| observed in the Si7 validation run, rounded up, provided
it stays well below both the smallest real mode and the |nu| of the
imaginary mode. This section will record the measured values.

### Si7 attempt 3 (VASP 6.6.1, 2026-09-29)

Full Hessian of the climbing image of the Si7 vacancy hop (IBRION=5,
EDIFF=1E-8, 21 modes; `examples/si7_vacancy_vasp`):

| quantity | |nu| (meV) |
|---|---|
| largest translational mode | 0.0068 |
| smallest real mode | 11.64 |
| smallest imaginary mode | 12.78 |

The current default of 1.0 meV lies about 150 times above the largest
translational mode and about 11 times below the smallest real and imaginary
modes, so it classified all 21 modes correctly. The residual was not larger
than in the perfect crystal, as expected above, but of the same small order.

The rule above would give 3 x 0.0068 meV, rounded up, as the new default.
That replacement is deferred: one system is not enough to set a default,
and the H2/Cu(111) stage-(b) run (a metal slab with a light adsorbate)
supplies the second data point. Until then the default stays at 1.0 meV,
and the status stays "calibration pending".

### Si7 attempt 4 (VASP 6.6.1, 2026-09-30)

First-order saddle of the same hop off the symmetry axis: largest
translational mode 0.097 meV, smallest real mode 5.51 meV, imaginary mode
21.22 meV. The translational residual is 14 times that of attempt 3, and
the smallest real mode half of it, so the margins of the 1.0 meV default
shrank to about 10 below and 5.5 above. A defect TS did give larger
residuals here, as first expected. The default stays at 1.0 meV.
