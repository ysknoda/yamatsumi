"""
Energy-profile reporting and plotting.

`report_energy_profile` lives here (rather than in one of the `phases/`
modules) because it is used by both `phases.neb_phase` and
`phases.plot_phase` -- keeping it in `plotting.py` avoids either phase
module having to import a private helper from the other.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger(__name__)


def report_energy_profile(
    image_dirs : List[Path],
    energies   : List[float],
) -> None:
    """Log the final energy profile along the reaction path as a table."""
    e0 = energies[0]
    n  = len(image_dirs)
    logger.info("\n  Final energy profile along reaction path:")
    logger.info(f"  {'Image':>8}  {'E (eV)':>14}  {'dE (eV)':>14}  Role")
    logger.info(f"  {'-' * 58}")
    for i, (idir, e) in enumerate(zip(image_dirs, energies)):
        role = (
            "initial endpoint" if i == 0
            else "final endpoint"   if i == n - 1
            else "intermediate"
        )
        logger.info(f"  {idir.name:>8}  {e:>14.6f}  {e - e0:>+14.6f}  {role}")


def _spline_fallback(xs: List[float], ys: List[float], n_points: int = 300) -> Tuple[list, list]:
    """Natural cubic spline in pure Python, used only if scipy is unavailable."""
    n = len(xs)
    h = [xs[i + 1] - xs[i] for i in range(n - 1)]
    alpha = [0.0] * n
    for i in range(1, n - 1):
        alpha[i] = (3.0 / h[i] * (ys[i + 1] - ys[i]) - 3.0 / h[i - 1] * (ys[i] - ys[i - 1]))
    l = [1.0] + [0.0] * (n - 1); mu = [0.0] * n; z = [0.0] * n
    for i in range(1, n - 1):
        l[i]  = 2.0 * (xs[i + 1] - xs[i - 1]) - h[i - 1] * mu[i - 1]
        mu[i] = h[i] / l[i]
        z[i]  = (alpha[i] - h[i - 1] * z[i - 1]) / l[i]
    l[-1] = 1.0
    c = [0.0] * n; b = [0.0] * (n - 1); d = [0.0] * (n - 1)
    for j in range(n - 2, -1, -1):
        c[j] = z[j] - mu[j] * c[j + 1]
        b[j] = (ys[j + 1] - ys[j]) / h[j] - h[j] * (c[j + 1] + 2.0 * c[j]) / 3.0
        d[j] = (c[j + 1] - c[j]) / (3.0 * h[j])
    x_fine = [xs[0] + (xs[-1] - xs[0]) * t / (n_points - 1) for t in range(n_points)]
    y_fine = []; seg = 0
    for xv in x_fine:
        while seg < n - 2 and xv > xs[seg + 1]:
            seg += 1
        dx = xv - xs[seg]
        y_fine.append(ys[seg] + b[seg] * dx + c[seg] * dx ** 2 + d[seg] * dx ** 3)
    return x_fine, y_fine


def spline_curve(xs: List[float], ys: List[float], n_points: int = 300) -> Tuple[list, list]:
    """
    Natural cubic spline through (xs, ys), sampled at n_points. Prefers
    scipy.interpolate.CubicSpline (bc_type="natural") when scipy is
    installed; falls back to an equivalent hand-rolled implementation
    otherwise, so plotting keeps working with numpy alone.
    """
    try:
        from scipy.interpolate import CubicSpline
    except ImportError:
        return _spline_fallback(xs, ys, n_points)

    cs = CubicSpline(xs, ys, bc_type="natural")
    x_fine = [xs[0] + (xs[-1] - xs[0]) * t / (n_points - 1) for t in range(n_points)]
    y_fine = [float(v) for v in cs(x_fine)]
    return x_fine, y_fine


def plot_energy_profile(
    image_dirs  : List[Path],
    energies    : List[float],
    ci_index    : int,
    phonon      : dict,
    barrier_fwd : float,
    barrier_rev : float,
    output_path : Path,
    fmt         : str  = "png",
    no_show     : bool = False,
) -> None:
    """Plot and save the NEB energy profile."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning(
            "matplotlib is not installed -- energy profile plot skipped.\n"
            "Install it with:  pip install matplotlib"
        )
        return

    n     = len(image_dirs)
    e0    = energies[0]
    x_raw = [i / (n - 1) for i in range(n)]
    y_raw = [e - e0 for e in energies]
    x_ci  = x_raw[ci_index]
    y_ci  = y_raw[ci_index]

    x_fine, y_fine = spline_curve(x_raw, y_raw, n_points=300)

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#ffffff")
    ax.plot(x_raw,  y_raw,  color="#cccccc", linewidth=1.0, zorder=1)
    ax.plot(x_fine, y_fine, color="#2166ac", linewidth=2.2,
            solid_capstyle="round", zorder=2, label="CI-NEB path (spline)")
    ax.scatter(x_raw, y_raw, s=60, color="#2166ac", zorder=4,
               edgecolors="white", linewidths=0.8, label="Image energy")

    ts_label = (
        "Transition state (CI)"
        if phonon.get("is_transition_state") else "Climbing image (CI)"
    )
    ax.scatter([x_ci], [y_ci], s=160, marker="*", color="#d6604d", zorder=5,
               edgecolors="#b2182b", linewidths=0.8, label=ts_label)

    ax.annotate("Initial", xy=(x_raw[0],  y_raw[0]),  xytext=(0.02, 0.05),
                textcoords="axes fraction", fontsize=9, color="#555555", ha="left")
    ax.annotate("Final",   xy=(x_raw[-1], y_raw[-1]), xytext=(0.98, 0.05),
                textcoords="axes fraction", fontsize=9, color="#555555", ha="right")
    for i, (xv, yv) in enumerate(zip(x_raw, y_raw)):
        ax.annotate(str(i), xy=(xv, yv), xytext=(0, 7),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=7, color="#333333")

    e_ts_rel    = y_ci
    e_final_rel = y_raw[-1]
    ax.annotate("", xy=(x_ci, e_ts_rel), xytext=(x_ci, 0.0),
                arrowprops=dict(arrowstyle="<->", color="#d6604d",
                                lw=1.4, linestyle="dashed"))
    ax.text(x_ci + 0.015, e_ts_rel / 2, f"Ea = {barrier_fwd:.3f} eV",
            color="#d6604d", fontsize=8.5, va="center")
    ax.annotate("", xy=(x_ci, e_ts_rel), xytext=(x_ci, e_final_rel),
                arrowprops=dict(arrowstyle="<->", color="#4dac26",
                                lw=1.4, linestyle="dashed"))
    ax.text(x_ci - 0.015, (e_ts_rel + e_final_rel) / 2,
            f"Ea- = {barrier_rev:.3f} eV",
            color="#4dac26", fontsize=8.5, va="center", ha="right")

    ts_ok    = phonon.get("is_transition_state", False)
    n_imag   = phonon.get("n_imaginary", "N/A")
    ts_check = (
        "TS verified (1 imaginary freq.)"
        if ts_ok else f"TS not verified ({n_imag} imaginary freq.)"
    )
    info_text = "\n".join([
        f"CI image : {ci_index} ({image_dirs[ci_index].name})",
        f"Ea (fwd) : {barrier_fwd:.4f} eV",
        f"Ea (rev) : {barrier_rev:.4f} eV",
        ts_check,
    ])
    ax.text(0.02, 0.97, info_text, transform=ax.transAxes,
            fontsize=8.5, va="top", ha="left", family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#eef4fb",
                      edgecolor="#aac4de", alpha=0.9))

    ax.axhline(0.0, color="#aaaaaa", linewidth=0.8, linestyle="--", zorder=0)
    ax.set_xlabel("Reaction coordinate  (0 = initial,  1 = final)", fontsize=11)
    ax.set_ylabel("Relative energy,  dE  [eV]", fontsize=11)
    ax.set_title("CI-NEB Energy Profile", fontsize=13, fontweight="bold", pad=10)
    ax.set_xlim(-0.04, 1.04)
    ax.margins(y=0.15)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.85)
    ax.grid(True, linestyle=":", linewidth=0.5, color="#dddddd", zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    fig.savefig(str(output_path), format=fmt, dpi=150, bbox_inches="tight")
    logger.info(f"  Energy profile plot saved: {output_path}")
    if not no_show:
        try:
            matplotlib.use("TkAgg"); plt.show()
        except Exception as exc:
            logger.debug(f"  Could not open an interactive plot window: {exc}")
    plt.close(fig)
