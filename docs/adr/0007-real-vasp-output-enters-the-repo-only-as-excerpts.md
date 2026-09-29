---
status: accepted (supersedes the earlier decision to keep real VASP output local-only)
---

# Real VASP output enters the repository only as trimmed excerpts

Parsers here were validated only against synthetic fixtures written by the
parsers' own author, and the first real OUTCAR showed a critical sign error
that those fixtures shared. Real output is therefore needed as a regression
fixture. A full OUTCAR cannot be committed, because it repeats POTCAR header
information, and POTCAR files are covered by the VASP licence.

Only the sections a parser reads, such as the phonon frequency and
eigenvector blocks, are committed, byte for byte, with the source file's
SHA-256 and line range recorded next to them. Full OUTCARs stay local and
gitignored. POTCAR files are never committed. Scripts build them at run time
from `VASP_POTCAR_DIR`.
