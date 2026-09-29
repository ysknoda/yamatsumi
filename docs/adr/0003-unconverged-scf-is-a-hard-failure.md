---
status: accepted
---

# An unconverged electronic SCF is a hard failure

When VASP reaches NELM without meeting EDIFF it exits with status 0 and
prints no error banner. yamatsumi used to accept such a run and take the last
`TOTEN` line it found, which could be the energy of an unconverged electronic
step. Truncating a real OUTCAR at electronic step 5 gave an energy 0.886 eV
off. That size of error corrupts a path without any visible sign.

A run now counts as successful only if its OUTCAR contains
`aborting loop because EDIFF is reached`. If the line is missing, yamatsumi
stops and names the image and the NELM in force. Under MPI all ranks stop
together, through the existing collective failure check. The energy is read
only from the ionic-step summary line (`free  energy   TOTEN`, two spaces),
never from the per-electronic-step lines. The marker was verified in real
OUTCARs from VASP 6.4.3 and 6.6.1.

## Considered Options

- Warn and continue. Rejected: unconverged forces would still enter the path.
- Retry once with a larger NELM. Rejected: it rewrites the user's INCAR
  silently, and raising NELM seldom fixes a system that does not converge.
