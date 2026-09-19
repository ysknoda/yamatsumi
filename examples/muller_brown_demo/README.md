# Muller-Brown CI-NEB demo (no VASP required)

Runs the full `yamatsumi` pipeline against the standard 4-Gaussian
Muller-Brown analytic potential surface -- a classic NEB-method benchmark
-- using a mock VASP executable (`tests/fixtures/mock_vasp.py`) in place of
real VASP. No license, POTCAR, or DFT calculation needed; this is meant as
a quick way to see the whole pipeline work end-to-end.

The surface is rescaled by /20 relative to its textbook form: at its
natural scale it is too stiff for `yamatsumi`'s default step size / spring
constant to optimise stably (verified empirically while designing this
demo). The known (rescaled) saddle energy between the two endpoints used
here is `-72.248940 / 20 = -3.612447`.

## Run it

All commands below are run from the **repository root** (so that
`python -m yamatsumi` finds the `yamatsumi` package without needing `pip install`
first; installing the package with `pip install -e .` also works fine and
lets you drop the `python -m` in favour of the `yamatsumi` console script).

```bash
python examples/muller_brown_demo/make_endpoints.py

MOCK_VASP="$(pwd)/tests/fixtures/mock_vasp.py"   # must be an absolute path --
                                                   # see note below

python -m yamatsumi --phase all \
  --init examples/muller_brown_demo/POSCAR_init \
  --final examples/muller_brown_demo/POSCAR_final \
  --copy-dir examples/muller_brown_demo/CopyDir \
  --work-dir examples/muller_brown_demo/NEB_run \
  --vasp-cmd "python3 $MOCK_VASP --pes mullerbrown --rescale 20.0" \
  --num-images 7 --optimizer fire --spring-const 5.0 --step-size 0.05 \
  --force-tol 0.05 --max-iter 200 --no-show
```

This takes well under two minutes (dozens of NEB iterations, each just a
few analytic-function evaluations rather than a real DFT calculation).

> **Why an absolute path:** `yamatsumi` launches `--vasp-cmd` with its working
> directory set to the image directory (`NEB_run/00`, `NEB_run/01`, ...) --
> the same way it launches real VASP, since VASP itself always writes
> `OUTCAR`/etc. into the current directory. A relative path to the mock
> VASP script (or to any custom wrapper script of your own) would resolve
> against that image directory instead of wherever you ran `yamatsumi` from,
> and fail with "No such file or directory".

## What to look at afterwards

- `examples/muller_brown_demo/NEB_run/energy_profile.png` -- the energy
  profile along the reaction path, with the climbing image marked.
- `examples/muller_brown_demo/NEB_run/yamatsumi_result.txt` -- forward/backward
  barriers and the phonon TS check. The climbing-image energy should land
  close to `-3.612447`.
- `examples/muller_brown_demo/NEB_run/convergence_history.csv` --
  per-iteration max/RMS NEB force.

To try MPI-parallel image dispatch instead (if you have a working MPI
runtime available), prefix the same command with `mpirun -n 7`.
