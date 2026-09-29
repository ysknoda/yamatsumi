"""Tests for yamatsumi.cli."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from yamatsumi import constants
from yamatsumi.cli import build_parser, main, validate_args
from yamatsumi.errors import NEBError


def test_defaults_match_constants():
    args = build_parser().parse_args([])
    assert args.phase == "all"
    assert args.init == constants.DEFAULT_INIT_POSCAR
    assert args.final == constants.DEFAULT_FINAL_POSCAR
    assert args.num_images == constants.DEFAULT_NUM_IMAGES
    assert args.path_interpolation == constants.DEFAULT_PATH_INTERPOLATION
    assert args.spring_const == constants.DEFAULT_SPRING_CONST
    assert args.force_tol == constants.DEFAULT_FORCE_TOL
    assert args.step_size == constants.DEFAULT_STEP_SIZE
    assert args.max_iter == constants.DEFAULT_MAX_ITER
    assert args.ci_start_iter == constants.DEFAULT_CI_START_ITER
    assert args.optimizer == constants.DEFAULT_OPTIMIZER
    assert args.copy_dir == constants.DEFAULT_COPY_DIR
    assert args.work_dir == constants.DEFAULT_WORK_DIR
    assert args.vasp_cmd == constants.DEFAULT_VASP_CMD
    assert args.vasp_timeout is None
    assert args.rate_estimate == constants.DEFAULT_RATE_ESTIMATE
    assert args.temperature == constants.DEFAULT_TEMPERATURE_K
    assert args.attempt_freq == constants.DEFAULT_ATTEMPT_FREQ_HZ
    assert args.local_workers == constants.DEFAULT_LOCAL_WORKERS
    assert args.sanitize_mpi_env is False
    assert args.plot_out == constants.DEFAULT_PLOT_OUT
    assert args.plot_fmt == constants.DEFAULT_PLOT_FMT
    assert args.resume is False
    assert args.log_interval == constants.DEFAULT_LOG_INTERVAL
    assert args.dry_run is False
    assert args.verbose is False
    assert args.allow_vasp_relaxation is False


def test_optimizer_choices_restricted():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--optimizer", "not-a-real-optimizer"])


def test_phase_choices_restricted():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--phase", "not-a-real-phase"])


def test_plot_fmt_choices_restricted():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--plot-fmt", "bmp"])


def test_main_dispatches_init_and_neb_for_phase_all():
    with patch("sys.argv", ["yamatsumi", "--phase", "all"]), \
         patch("yamatsumi.cli.phase_init") as mock_init, \
         patch("yamatsumi.cli.phase_neb") as mock_neb, \
         patch("yamatsumi.cli.phase_plot") as mock_plot:
        main()
    mock_init.assert_called_once()
    mock_neb.assert_called_once()
    mock_plot.assert_not_called()


def test_main_dispatches_only_plot_for_phase_plot():
    with patch("sys.argv", ["yamatsumi", "--phase", "plot"]), \
         patch("yamatsumi.cli.phase_init") as mock_init, \
         patch("yamatsumi.cli.phase_neb") as mock_neb, \
         patch("yamatsumi.cli.phase_plot") as mock_plot:
        main()
    mock_init.assert_not_called()
    mock_neb.assert_not_called()
    mock_plot.assert_called_once()


def test_main_reports_neb_error_and_exits_nonzero():
    with patch("sys.argv", ["yamatsumi", "--phase", "neb"]), \
         patch("yamatsumi.cli.phase_neb", side_effect=NEBError("boom")), \
         pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# validate_args: numeric values argparse's type= accepts but that either
# crash outright or silently produce meaningless physics.
# ---------------------------------------------------------------------------

def test_validate_args_accepts_defaults():
    validate_args(build_parser().parse_args([]))  # must not raise


@pytest.mark.parametrize(
    "argv",
    [
        # Regression: iteration % 0 raised ZeroDivisionError on iteration 1.
        ["--log-interval", "0"],
        ["--log-interval", "-1"],
        # Regression: an empty iteration range left `energies` empty and
        # crashed the trailing report with IndexError.
        ["--max-iter", "0"],
        # No intermediate image to optimise at all.
        ["--num-images", "0"],
        ["--num-images", "-3"],
        # 0 never moves an image; negative steps uphill and diverges.
        ["--step-size", "0"],
        ["--step-size", "-0.05"],
        # A norm can never fall below a non-positive tolerance.
        ["--force-tol", "0"],
        ["--force-tol", "-0.01"],
        # Anti-restoring spring: images repel along the path.
        ["--spring-const", "-5.0"],
        # Every VASP invocation would time out instantly.
        ["--vasp-timeout", "0"],
        ["--vasp-timeout", "-10"],
        ["--ci-start-iter", "0"],
        ["--local-workers", "0"],
        # A negative tolerance would count every mode as imaginary.
        ["--zero-mode-tol", "-1"],
    ],
)
def test_validate_args_rejects_meaningless_numeric_values(argv):
    args = build_parser().parse_args(argv)
    with pytest.raises(NEBError):
        validate_args(args)


def test_validate_args_allows_zero_spring_constant():
    """k=0 is a legitimate 'no spring' NEB variant, unlike a negative k."""
    validate_args(build_parser().parse_args(["--spring-const", "0"]))  # must not raise


def test_validate_args_allows_unset_vasp_timeout():
    """The default (None) means an unbounded wait and must stay allowed."""
    args = build_parser().parse_args([])
    assert args.vasp_timeout is None
    validate_args(args)  # must not raise


def test_main_runs_validation_before_any_phase():
    """A bad numeric value must abort before a single phase is entered."""
    with patch("sys.argv", ["yamatsumi", "--phase", "all", "--log-interval", "0"]), \
         patch("yamatsumi.cli.phase_init") as mock_init, \
         patch("yamatsumi.cli.phase_neb") as mock_neb, \
         pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 1
    mock_init.assert_not_called()
    mock_neb.assert_not_called()
