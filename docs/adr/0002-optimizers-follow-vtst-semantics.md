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

## Amendment: the climbing image uses the ordinary FIRE step

yamatsumi originally ran the climbing image through its own FIRE variant.
The power test used only the component of the force perpendicular to the
path, so that the uphill tangential component would not "falsely" reset the
velocity. As a result the tangential velocity was never damped. It
oscillated about the saddle, each sign flip halved `dt`, and the image froze
short of the saddle. On E = x^4/4 - x^2/2 + 0.3x + y^2/2, `dt` fell to about
1e-180 with the image 0.0285 Ang from the saddle and max|F_NEB| stuck at
0.018 eV/Ang. That residual is below a 0.02 eV/Ang tolerance, so the run
would have been reported as converged at the wrong geometry.

The climbing image now takes the ordinary FIRE step on the force that
`compute_neb_forces` already returns for it, F - 2(F.tau)tau, which is a
descent direction for the climbing objective. VTST and ASE both do this.
The same surface now converges in 97 iterations with the image 2.5e-6 Ang
from the saddle. The Muller-Brown demo converges in 26 iterations instead of
103, with the barrier unchanged to 2e-5 eV.

## Amendment 2 (2026-10-06): --step-size caps the move, not the FIRE velocity

yamatsumi's FIRE capped the velocity at `--step-size` and then the move
v*dt as well. With dt starting at 0.1 that held an image to 0.005 Ang per
iteration at the default 0.05 Ang: in the stage-(b) H2/Cu(111) run (ADR
0009) no H atom moved more than 0.07 Ang in 10 iterations against forces
of 4.5 eV/Ang. The FIRE paper (Bitzek et al., PRL 97, 170201, 2006) has no
velocity cap; ASE 3.28.0 caps only the step dt*v (maxstep), and VTST
documents MAXMOVE as the maximum step. Whether VTST also caps the velocity
in its code is UNCERTAIN (source not checked). The velocity cap is removed;
the per-atom cap on the move stays, and the default stays 0.05 Ang.

Iterations to convergence on analytic surfaces, in-process, same drivers
and seeds, before -> after (every climbing image ends on the same saddle,
energies within 1e-4 eV):

| surface | step 0.05 | step 0.2 |
|---|---|---|
| double well | 42 -> 42 | 40 -> 42 |
| Muller-Brown (/20) | 26 -> 45 | 42 -> 45 |
| asymmetric quartic | 97 -> 75 | 84 -> 75 |
| plateau, near start | 23 -> 23 | 23 -> 23 |
| plateau, far start | 76 -> 54 | 53 -> 73 |
| plateau, far start, noise | 70 -> 55 | 51 -> 74 |
| plateau + free direction, noise | 71 -> 86 | 69 -> 63 |
| plateau + 5 meV well, noise | 71 -> 55 | 57 -> 74 |
| plateau + well + free direction, noise | 81 -> 75 | 77 -> 66 |

The change is not a uniform gain. On Muller-Brown at 0.05 Ang the largest
move per step grew from 0.0086 to 0.034 Ang, FIRE resets (P <= 0) from 22
to 34, and the mean move fell from 0.0024 to 0.0016 Ang: the velocity cap
had been damping overshoot on a stiff surface. Where the path must travel
far, as at the start of H2/Cu(111), it converges sooner.
