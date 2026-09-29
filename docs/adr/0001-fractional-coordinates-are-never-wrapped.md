---
status: accepted
---

# Fractional coordinates are never wrapped; wrapped endpoint input is unwrapped on read

`neb_forces.py` builds tangents and spring forces from plain Cartesian
differences between adjacent images, without the minimum-image convention.
That is correct only if no image's fractional coordinates are ever folded back
into [0, 1), so nothing in yamatsumi wraps coordinates, and nothing may be
added that does. A tidy-up that wraps coordinates would put a full lattice
vector into the tangent of any segment where an atom crosses a cell boundary.

The invariant also has to hold for input. VASP writes CONTCAR with coordinates
wrapped into [0, 1), and endpoints are normally relaxed with VASP, so an atom
at z = -0.02 in the initial endpoint can come back as z = 0.98 in the final
one. Intermediate images were already built from the MIC displacement, but the
final endpoint was used verbatim. A measured case gave max|F_NEB| = 49.0 eV/Ang
on the last image, against 0.000 when unwrapped. The final endpoint is
therefore replaced by `init + MIC(final - init)` when it is read, and every
atom that moves is logged.

## Considered Options

- Reject a wrapped final endpoint with an error. Rejected: it would stop the
  ordinary workflow of feeding in two VASP-relaxed CONTCARs on every run.
- Apply the minimum-image convention inside `neb_forces.py`. Rejected: it
  hides the problem in one consumer instead of removing it at the source, and
  every other consumer of image coordinates would have to repeat the fix.
