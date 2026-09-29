---
status: accepted
---

# Optimizers follow VTST semantics, and FIRE is the default

Before this decision, `--step-size` was a normaliser rather than a cap: the
steepest-descent displacement was exactly `step_size` whatever the force, from
10 down to 1e-9 eV/Ang. Quick-Min was bit-identical to steepest descent. On
the asymmetric Muller-Brown surface both stalled at max|F| of about 4.3 eV/Ang
for 200 iterations, while FIRE converged in 103. (A symmetric double well
converged with steepest descent, which is why the tests had not caught it.)

We adopted the parameterisation of the VTST Tools optimizers
(henkelmangroup.github.io/vtsttools/optimizers.html), which is the reference
NEB implementation for VASP:

- steepest descent moves `SDALPHA * F`; `--sd-alpha`, default 0.01, as VTST;
- Quick-Min integrates with a real time step; `--quickmin-dt`, default 0.1,
  the VTST `TIMESTEP`;
- `--step-size` caps the per-atom displacement for all three optimizers, the
  role of the VTST `MAXMOVE`.

yamatsumi's FIRE defaults already matched VTST exactly (dt0 0.1, dt_max 1.0,
f_dec 0.5, f_inc 1.1, alpha0 0.1, n_min 5). FIRE becomes the default because
it is the only optimizer shown to converge on an asymmetric surface, and VTST
recommends Quick-Min or FIRE when forces are noisy, which is the normal state
of DFT forces.

## Consequences

This changes the default behaviour of an existing command line. It is accepted
because the package is an alpha release. `--step-size` keeps its default of
0.05 Ang, which is more conservative than the VTST `MAXMOVE` of 0.2 Ang.
