---
status: accepted
---

# The Si7 TS criterion A4 is an energy bound, not the bond midpoint

The stage-(a) Si7 validation (`docs/real_vasp_validation.md`) fixed its
criteria before the first run and does not change them afterwards. A4
required the climbing image's migrating atom to lie within 0.02 Ang of the
midpoint of the hopping bond, because the inversion centre there relates
the two endpoints.

Attempt 3 (2026-09-29) disproved the premise. With inversion-related
endpoints the path stayed on the hop axis and converged to the midpoint,
which turned out to be a saddle with three imaginary modes: the reaction
mode (32.45 meV) and a degenerate pair orthogonal to the path (12.78 meV).
The first-order saddle lies off the axis, so it is not at the midpoint.

A4 is now "forward barrier < 1.3013 eV", the barrier of that symmetric
saddle. A first-order saddle reached by leaving it along its unstable pair
lies lower whatever its geometry. The TS distance from the midpoint, the
bisecting plane and the hop axis is still reported, as data only.

The change was made on 2026-09-29 at 22:37, while attempt 4 was still in
its first single points and before it had produced any result. This is
what separates it from tuning a criterion to a result. Attempts 1-3 stay
judged under the old A4.

## Considered Options

- Keep the old A4 and report its failure as a wrong premise. Rejected: the
  criterion would then fail on a correct result, and so test nothing.
- Drop A4 and rely on A5 and A6. Rejected: A5 and A6 check that the climbing
  image is a first-order saddle along the path, not that it is a different
  saddle from the one attempt 3 found.
