---
status: accepted
---

# Image directories are always named with two digits

Image directories were numbered with as many digits as the last index needed,
so eight or fewer intermediate images gave `0` to `9`. That contradicted the
docstring and the convention of VASP's own `IMAGES` mode and of the VTST
scripts (`00`, `01`, ...). Names are now always at least two digits wide.

A work directory created by 1.0.0a1 with single-digit names is not resumed.
`--resume` detects that layout and stops with an explanation instead of
silently creating `00` and starting again from scratch while the user thinks
the old run is being continued. We accepted breaking those directories because
1.0.0a1 was an alpha that had not been used with real VASP.
