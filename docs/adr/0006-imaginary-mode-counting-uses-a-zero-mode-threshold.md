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
