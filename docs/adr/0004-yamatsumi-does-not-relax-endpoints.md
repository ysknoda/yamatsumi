---
status: accepted
---

# yamatsumi does not relax endpoints

yamatsumi assumes both endpoints are already relaxed, and it will not gain a
relaxation phase. Relaxing a structure is something VASP does well on its own,
with `IBRION = 2` and `ISIF = 2`. Taking it on inside yamatsumi would mean
maintaining a second INCAR, and it would collide with the start-up check that
requires `CopyDir/INCAR` to be a single point (`NSW = 0`, `IBRION = -1`).
Users relax endpoints with VASP directly and pass in the CONTCARs, and
[ADR 0001](0001-fractional-coordinates-are-never-wrapped.md) makes that safe.
