# Real VASP output excerpts

Byte-for-byte excerpts of real VASP output, committed as regression fixtures
under [ADR 0007](../../../docs/adr/0007-real-vasp-output-enters-the-repo-only-as-excerpts.md).
Only the section a parser reads is kept. Full OUTCARs repeat POTCAR header
information and stay local.

| File | Source | Lines | Source SHA-256 |
|---|---|---|---|
| `si8_bulk_ibrion6_vasp643_phonon.OUTCAR` | VASP 6.4.3, 8-atom conventional diamond Si, `IBRION = 6`, `NFREE = 2`, `POTIM = 0.01` | 9401-9669 (from the "Eigenvectors and eigenvalues of the dynamical matrix" header to just before "ELASTIC MODULI") | `c0acffdee2ab8632878dab87a8d911070e794bd3c304214afc97815546855aa3` |

The source is a perfect crystal at its minimum, so the correct count of
imaginary modes is zero. VASP labels the three translational modes `f/i`
at about 3e-6 meV, which is the case
[ADR 0006](../../../docs/adr/0006-imaginary-mode-counting-uses-a-zero-mode-threshold.md)
addresses.
