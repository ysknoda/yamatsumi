# Stage (a) real-VASP validation: Si7 vacancy hop

`run_validation.py` checks yamatsumi end to end against real VASP on a
single vacancy hop in the 8-site conventional diamond Si cell. The cell is
too small for a physically meaningful barrier. It is used because its
inversion symmetry predicts the answers exactly: degenerate endpoints, a
transition state at the bond midpoint, and exactly one imaginary mode.

```bash
nohup caffeinate -i python3 examples/si7_vacancy_vasp/run_validation.py \
    --env-script ~/vasp/vasp661_env.sh > si7_validation.log 2>&1 &
```

Everything is written under `run/` next to the script (gitignored; it holds
the POTCAR copied from your PAW library). Stages are checkpointed, so
rerunning the command continues an interrupted run, and `--evaluate-only`
re-scores an existing one. The acceptance criteria, the INCAR choices and
their sources, and the results are in
[docs/real_vasp_validation.md](../../docs/real_vasp_validation.md).
