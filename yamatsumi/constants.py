"""Default parameter values for the CI-NEB CLI and algorithm."""

DEFAULT_NUM_IMAGES    = 5
DEFAULT_PATH_INTERPOLATION = "linear"
DEFAULT_COPY_DIR      = "CopyDir"
DEFAULT_WORK_DIR      = "NEB_run"
DEFAULT_VASP_CMD      = "vasp_std"
DEFAULT_INIT_POSCAR   = "POSCAR_init"
DEFAULT_FINAL_POSCAR  = "POSCAR_final"
DEFAULT_MAX_ITER      = 100
DEFAULT_FORCE_TOL     = 0.05        # eV/Ang
DEFAULT_SPRING_CONST  = 5.0         # eV/Ang
DEFAULT_STEP_SIZE     = 0.05        # Ang
DEFAULT_CI_START_ITER = 1
DEFAULT_OPTIMIZER     = "sd"
DEFAULT_FIRE_DT0      = 0.1
DEFAULT_FIRE_DT_MAX   = 1.0
DEFAULT_FIRE_N_MIN    = 5
DEFAULT_FIRE_F_INC    = 1.1
DEFAULT_FIRE_F_DEC    = 0.5
DEFAULT_FIRE_ALPHA0   = 0.1
DEFAULT_FIRE_F_ALPHA  = 0.99
DEFAULT_LOG_INTERVAL  = 1           # full energy profile every N iterations
DEFAULT_PLOT_OUT      = "energy_profile.png"
DEFAULT_PLOT_FMT      = "png"
DEFAULT_LOCAL_WORKERS = 1           # thread-pool width when not running under mpirun

KB_EV_PER_K              = 8.617333262e-5  # Boltzmann constant, eV/K
DEFAULT_TEMPERATURE_K    = 300.0
DEFAULT_ATTEMPT_FREQ_HZ  = 1.0e13           # typical lattice-vibration order of magnitude
DEFAULT_RATE_ESTIMATE    = "none"
