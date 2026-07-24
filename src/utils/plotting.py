"""
Plotting utilities for the geothermal pile PINN (Task 2 challenge).

Centralizes chart generation that was previously duplicated across
`scripts/train.py`, `scripts/compare_variants.py`, and
`scripts/grid_search.py`. Those scripts should import from here rather
than defining their own copies -- this module has no side effects on
import (nothing runs or plots until you call a function), so it's safe
to import from anywhere.
"""

from pathlib import Path
from typing import Dict, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import TwoSlopeNorm

from src.models.ensemble import PINNEnsemble
from src.models.pinn import GeothermalPINN

DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent.parent.parent / "data" / "processed"
)


def reshape_ref_grid(X_ref: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Recover the (r*, t*) meshgrid and shape behind a flattened X_ref."""
    r_vals = np.unique(X_ref[:, 0])
    t_vals = np.unique(X_ref[:, 1])
    n_r, n_t = len(r_vals), len(t_vals)
    r_grid, t_grid = np.meshgrid(r_vals, t_vals, indexing="xy")
    return r_grid, t_grid, n_r, n_t


def u_star_norm(*grids: np.ndarray) -> TwoSlopeNorm:
    """
    Diverging norm for u*, centered exactly at zero.

    Excess pore pressure should only build up (u* >= 0) while heating in
    this scenario; any negative region is a numerical artifact, not a
    physical feature. Centering the colormap at zero makes that region
    visually obvious instead of blending into the rest of the scale.
    """
    vmin = min(float(g.min()) for g in grids)
    vmax = max(float(g.max()) for g in grids)
    vmin = min(vmin, -1e-6)
    vmax = max(vmax, 1e-6)
    return TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)


def plot_field_pair(
    r_grid: np.ndarray,
    t_grid: np.ndarray,
    T_field: np.ndarray,
    u_field: np.ndarray,
    title: str,
    out_path: Path,
) -> Path:
    """
    Plot one model's (or the FDM reference's) T*/u* fields side by side.

    r* is shown on a log scale: almost all of the interesting structure
    sits within r* < 0.2 (close to the pile), which a linear axis
    compresses into a sliver. u* uses a diverging colormap centered at
    u*=0 with the zero level marked on the colorbar, since negative
    excess pore pressure has no physical meaning here and should be
    easy to spot at a glance.

    Args:
        r_grid: Meshgrid of r* values, shape (n_t, n_r).
        t_grid: Meshgrid of t* values, shape (n_t, n_r).
        T_field: Temperature field, shape (n_t, n_r).
        u_field: Pore-pressure field, shape (n_t, n_r).
        title: Figure title.
        out_path: Where to save the PNG.

    Returns:
        The path the figure was saved to.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    mesh_T = axes[0].pcolormesh(r_grid, t_grid, T_field, shading="auto", cmap="viridis")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("r* (log scale)")
    axes[0].set_ylabel("t*")
    axes[0].set_title("T*")
    fig.colorbar(mesh_T, ax=axes[0], label="T*")

    norm = u_star_norm(u_field)
    mesh_u = axes[1].pcolormesh(
        r_grid, t_grid, u_field, shading="auto", cmap="RdBu_r", norm=norm
    )
    axes[1].set_xscale("log")
    axes[1].set_xlabel("r* (log scale)")
    axes[1].set_title("u* (blue = negative, non-physical here)")
    cbar = fig.colorbar(mesh_u, ax=axes[1], label="u*")
    cbar.ax.axhline(0.0, color="black", linewidth=1.2)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")
    return out_path


def plot_reference_pair(
    X_ref: np.ndarray,
    y_ref: np.ndarray,
    case_num: int,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    """Plot the FDM reference T*/u* fields as the 'correct answer' pair."""
    r_grid, t_grid, n_r, n_t = reshape_ref_grid(X_ref)
    T_field = y_ref[:, 0].reshape(n_t, n_r)
    u_field = y_ref[:, 1].reshape(n_t, n_r)
    out_path = output_dir / f"case{case_num}_reference_pair.png"
    return plot_field_pair(
        r_grid, t_grid, T_field, u_field, f"Case {case_num}: FDM reference", out_path
    )


def plot_model_pair(
    model: GeothermalPINN,
    X_ref: np.ndarray,
    case_num: int,
    name: str,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    """Plot a trained model's predicted T*/u* fields as one pair figure."""
    r_grid, t_grid, n_r, n_t = reshape_ref_grid(X_ref)
    r_ref = torch.tensor(X_ref[:, 0:1], dtype=torch.float32)
    t_ref = torch.tensor(X_ref[:, 1:2], dtype=torch.float32)
    T_pred, u_pred = model.predict(r_ref, t_ref)
    T_field = T_pred.reshape(n_t, n_r)
    u_field = u_pred.reshape(n_t, n_r)
    out_path = output_dir / f"case{case_num}_{name}_pair.png"
    return plot_field_pair(
        r_grid, t_grid, T_field, u_field, f"Case {case_num}: {name}", out_path
    )


def plot_error_heatmap(
    model: GeothermalPINN,
    X_ref: np.ndarray,
    y_ref: np.ndarray,
    case_num: int,
    name: str,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    """
    Plot the absolute error |PINN - FDM| for T*/u* as heatmaps.

    Unlike `plot_field_pair` (which shows the fields themselves), this
    shows WHERE predictions deviate from the FDM reference -- useful
    for telling apart "the fit is bad everywhere" from "the fit is fine
    near the pile (where u* actually matters physically) but noisy in
    the flat far field (where u*=0 anyway)", a distinction a single
    relative-L2 number can't make.

    Args:
        model: Trained GeothermalPINN.
        X_ref: Reference inputs, columns [r*, t*].
        y_ref: Reference targets, columns [T*, u*].
        case_num: FDM case number, used in the title/filename.
        name: Label for this model/variant, used in the title/filename.
        output_dir: Directory to save the PNG in.

    Returns:
        The path the figure was saved to.
    """
    r_grid, t_grid, n_r, n_t = reshape_ref_grid(X_ref)
    r_ref = torch.tensor(X_ref[:, 0:1], dtype=torch.float32)
    t_ref = torch.tensor(X_ref[:, 1:2], dtype=torch.float32)
    T_pred, u_pred = model.predict(r_ref, t_ref)

    T_err = np.abs(T_pred.reshape(n_t, n_r) - y_ref[:, 0].reshape(n_t, n_r))
    u_err = np.abs(u_pred.reshape(n_t, n_r) - y_ref[:, 1].reshape(n_t, n_r))

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    mesh_T = axes[0].pcolormesh(r_grid, t_grid, T_err, shading="auto", cmap="Reds")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("r* (log scale)")
    axes[0].set_ylabel("t*")
    axes[0].set_title("|T* error|")
    fig.colorbar(mesh_T, ax=axes[0], label="|T_pred - T_true|")

    mesh_u = axes[1].pcolormesh(r_grid, t_grid, u_err, shading="auto", cmap="Reds")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("r* (log scale)")
    axes[1].set_title("|u* error|")
    fig.colorbar(mesh_u, ax=axes[1], label="|u_pred - u_true|")

    fig.suptitle(f"Case {case_num}: {name} -- absolute error vs FDM")
    fig.tight_layout()

    out_path = output_dir / f"case{case_num}_{name}_error.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")
    return out_path


def plot_profiles(
    model: GeothermalPINN,
    X_ref: np.ndarray,
    y_ref: np.ndarray,
    case_num: int,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    """Plot PINN vs FDM temperature/pressure profiles at a few time steps."""
    output_dir.mkdir(parents=True, exist_ok=True)

    t_values = np.unique(X_ref[:, 1])
    snapshot_times = [t_values[0], t_values[len(t_values) // 2], t_values[-1]]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for t_star in snapshot_times:
        mask = np.isclose(X_ref[:, 1], t_star)
        r_slice = X_ref[mask, 0]
        order = np.argsort(r_slice)
        r_slice = r_slice[order]

        r_tensor = torch.tensor(r_slice.reshape(-1, 1), dtype=torch.float32)
        t_tensor = torch.full_like(r_tensor, float(t_star))
        T_pred, u_pred = model.predict(r_tensor, t_tensor)

        axes[0].plot(r_slice, y_ref[mask, 0][order], "o", ms=3, alpha=0.4)
        axes[0].plot(r_slice, T_pred.ravel(), "-", label=f"t*={t_star:.3f}")

        axes[1].plot(r_slice, y_ref[mask, 1][order], "o", ms=3, alpha=0.4)
        axes[1].plot(r_slice, u_pred.ravel(), "-", label=f"t*={t_star:.3f}")

    axes[0].set_xlabel("r*")
    axes[0].set_ylabel("T*")
    axes[0].set_title("Temperature (dots=FDM, lines=PINN)")
    axes[0].legend()

    axes[1].set_xlabel("r*")
    axes[1].set_ylabel("u*")
    axes[1].set_title("Pore pressure (dots=FDM, lines=PINN)")
    axes[1].legend()

    fig.suptitle(f"Forward PINN vs FDM -- case {case_num}")
    fig.tight_layout()

    out_path = output_dir / f"forward_case{case_num}_profiles.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")
    return out_path


def plot_variant_heatmaps(
    variants: Dict[str, GeothermalPINN],
    X_ref: np.ndarray,
    y_ref: np.ndarray,
    case_num: int,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    """Plot FDM reference + every variant's T*/u* field in one grid."""
    output_dir.mkdir(parents=True, exist_ok=True)

    r_grid, t_grid, n_r, n_t = reshape_ref_grid(X_ref)
    T_fdm = y_ref[:, 0].reshape(n_t, n_r)
    u_fdm = y_ref[:, 1].reshape(n_t, n_r)

    r_ref = torch.tensor(X_ref[:, 0:1], dtype=torch.float32)
    t_ref = torch.tensor(X_ref[:, 1:2], dtype=torch.float32)

    names = list(variants.keys())
    n_cols = 1 + len(names)

    fig, axes = plt.subplots(
        2, n_cols, figsize=(3.3 * n_cols, 7.5), sharex=True, sharey=True
    )

    for row, (label, cmap, fdm_grid) in enumerate(
        [("T*", "viridis", T_fdm), ("u*", "plasma", u_fdm)]
    ):
        pinn_grids = []
        for name in names:
            T_pred, u_pred = variants[name].predict(r_ref, t_ref)
            grid = (T_pred if row == 0 else u_pred).reshape(n_t, n_r)
            pinn_grids.append(grid)

        vmin = min(fdm_grid.min(), *(g.min() for g in pinn_grids))
        vmax = max(fdm_grid.max(), *(g.max() for g in pinn_grids))

        panels = [("FDM (reference)", fdm_grid)] + list(zip(names, pinn_grids))
        mesh = None
        for col, (title, grid) in enumerate(panels):
            ax = axes[row, col]
            mesh = ax.pcolormesh(
                r_grid, t_grid, grid, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax
            )
            ax.set_title(f"{label} -- {title}", fontsize=10)
            if row == 1:
                ax.set_xlabel("r*")
            if col == 0:
                ax.set_ylabel("t*")
        fig.colorbar(mesh, ax=axes[row, :].tolist(), shrink=0.8, label=label)

    fig.suptitle(f"Case {case_num}: FDM reference vs PINN variants")

    out_path = output_dir / f"case{case_num}_variant_heatmaps.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")
    return out_path


def plot_metric_comparison(
    metrics: Dict[str, Dict[str, float]],
    case_num: int,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    """Plot a bar chart of relative L2 error per variant."""
    output_dir.mkdir(parents=True, exist_ok=True)

    names = list(metrics.keys())
    T_errs = [metrics[n]["T_rel_l2"] for n in names]
    u_errs = [metrics[n]["u_rel_l2"] for n in names]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bars_T = ax.bar(x - width / 2, T_errs, width, label="T* rel. L2 error")
    bars_u = ax.bar(x + width / 2, u_errs, width, label="u* rel. L2 error")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Relative L2 error")
    ax.set_title(f"Case {case_num}: error comparison across variants")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.bar_label(bars_T, fmt="%.2f", fontsize=8)
    ax.bar_label(bars_u, fmt="%.2f", fontsize=8)

    fig.tight_layout()
    out_path = output_dir / f"case{case_num}_metric_comparison.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")
    return out_path


def plot_loss_curves(
    variants: Dict[str, GeothermalPINN],
    case_num: int,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    """Plot the training loss convergence curve for each variant."""
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, model in variants.items():
        ax.plot(model.loss_history, label=name, alpha=0.8, linewidth=1)
    ax.set_yscale("log")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Total training loss (log scale)")
    ax.set_title(f"Case {case_num}: training convergence")
    ax.legend()
    ax.grid(alpha=0.3, which="both")

    fig.tight_layout()
    out_path = output_dir / f"case{case_num}_loss_curves.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")
    return out_path


def plot_grid_heatmap(
    results: Sequence,
    case_num: int,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    metric_names: Tuple[str, str] = ("T_rel_l2", "u_rel_l2"),
) -> Path:
    """
    Plot a T*/u* error metric over the (w_u, adam_epochs) grid, with one
    column of panels per distinct lbfgs_epochs value tried.

    Args:
        results: Sequence of grid_search.GridPoint (or any object with
            w_u, adam_epochs, lbfgs_epochs, and the two attributes named
            in `metric_names`).
        case_num: FDM case number, used in the title/filename.
        output_dir: Directory to save the PNG in.
        metric_names: Which (T, u) attributes on each result to plot.
            Defaults to relative L2 error; pass ("T_nrmse", "u_nrmse")
            for the max-normalized RMSE instead, which is less
            misleading for u* since it stays close to zero almost
            everywhere except right next to the pile.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    T_metric, u_metric = metric_names

    w_u_values = sorted({p.w_u for p in results})
    adam_values = sorted({p.adam_epochs for p in results})
    lbfgs_values = sorted({p.lbfgs_epochs for p in results})

    fig, axes = plt.subplots(
        2, len(lbfgs_values), figsize=(4.5 * len(lbfgs_values), 8), squeeze=False
    )

    for col, lbfgs_epochs in enumerate(lbfgs_values):
        subset = {
            (p.w_u, p.adam_epochs): p for p in results if p.lbfgs_epochs == lbfgs_epochs
        }
        T_grid = np.array(
            [
                [getattr(subset[(w, a)], T_metric) for w in w_u_values]
                for a in adam_values
            ]
        )
        u_grid = np.array(
            [
                [getattr(subset[(w, a)], u_metric) for w in w_u_values]
                for a in adam_values
            ]
        )

        for row, (label, grid, cmap) in enumerate(
            [(T_metric, T_grid, "viridis"), (u_metric, u_grid, "magma")]
        ):
            ax = axes[row, col]
            mesh = ax.imshow(grid, aspect="auto", cmap=cmap, origin="lower")
            ax.set_xticks(range(len(w_u_values)))
            ax.set_xticklabels([f"{w:g}" for w in w_u_values])
            ax.set_yticks(range(len(adam_values)))
            ax.set_yticklabels([str(a) for a in adam_values])
            ax.set_xlabel("w_u")
            if col == 0:
                ax.set_ylabel("adam_epochs")
            ax.set_title(f"{label} (lbfgs_epochs={lbfgs_epochs})", fontsize=10)
            for r, _ in enumerate(adam_values):
                for c, _ in enumerate(w_u_values):
                    ax.text(
                        c,
                        r,
                        f"{grid[r, c]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="white",
                    )
            fig.colorbar(mesh, ax=ax, shrink=0.85)

    fig.suptitle(f"Case {case_num}: grid search error surface ({T_metric}/{u_metric})")
    fig.tight_layout()
    suffix = "" if metric_names == ("T_rel_l2", "u_rel_l2") else f"_{u_metric}"
    out_path = output_dir / f"case{case_num}_grid_search_heatmap{suffix}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")
    return out_path


def plot_radar_chart(
    df: pd.DataFrame,
    metrics: Sequence[str],
    case_num: int,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    """
    Plot a radar/spider chart comparing configurations across metrics.

    Each metric is min-max normalized and inverted (1.0 = best value
    seen, 0.0 = worst), so a larger filled area always means a better
    overall configuration, regardless of whether a given metric is
    "smaller is better" (true for all the error metrics used here).

    Args:
        df: DataFrame with one row per configuration, including 'w_u',
            'adam_epochs', and 'lbfgs_epochs' columns plus one column
            per entry in `metrics`.
        metrics: Column names to plot as radar axes.
        case_num: FDM case number, used in the title/filename.
        output_dir: Directory to save the PNG in.

    Returns:
        The path the figure was saved to.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    df_norm = df.copy()
    for m in metrics:
        min_v, max_v = df[m].min(), df[m].max()
        df_norm[m] = 1.0 - (df[m] - min_v) / (max_v - min_v + 1e-9)

    df_norm["config"] = (
        "w="
        + df["w_u"].astype(str)
        + ", A="
        + df["adam_epochs"].astype(str)
        + ", L="
        + df["lbfgs_epochs"].astype(str)
    )

    num_vars = len(metrics)
    angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for _, row in df_norm.iterrows():
        values = row[list(metrics)].values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=1.5, linestyle="solid", label=row["config"])
        ax.fill(angles, values, alpha=0.05)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(list(metrics), size=11)
    ax.set_rlim(0, 1)
    ax.set_title(f"Case {case_num}: config comparison (larger area = better)", y=1.1)
    ax.legend(bbox_to_anchor=(1.2, 1.1), loc="upper left", fontsize=9)
    fig.tight_layout()

    out_path = output_dir / f"case{case_num}_radar_chart.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")
    return out_path


def plot_uncertainty_band(
    ensemble: PINNEnsemble,
    X_ref: np.ndarray,
    y_ref: np.ndarray,
    case_num: int,
    name: str,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    n_sigma: float = 2.0,
) -> Path:
    """
    Plot ensemble mean +/- n_sigma*std vs FDM reference, at a few times.

    Args:
        ensemble: Trained PINNEnsemble.
        X_ref: Reference inputs, columns [r*, t*].
        y_ref: Reference targets, columns [T*, u*].
        case_num: FDM case number, used in the title/filename.
        name: Label for this ensemble, used in the title/filename.
        output_dir: Directory to save the PNG in.
        n_sigma: Width of the shaded uncertainty band, in std devs.

    Returns:
        The path the figure was saved to.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    t_values = np.unique(X_ref[:, 1])
    snapshot_times = [t_values[0], t_values[len(t_values) // 2], t_values[-1]]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for t_star in snapshot_times:
        mask = np.isclose(X_ref[:, 1], t_star)
        r_slice = X_ref[mask, 0]
        order = np.argsort(r_slice)
        r_slice = r_slice[order]

        r_tensor = torch.tensor(r_slice.reshape(-1, 1), dtype=torch.float32)
        t_tensor = torch.full_like(r_tensor, float(t_star))
        T_mean, T_std, u_mean, u_std = ensemble.predict(r_tensor, t_tensor)

        (line_T,) = axes[0].plot(r_slice, T_mean.ravel(), "-", label=f"t*={t_star:.3f}")
        axes[0].fill_between(
            r_slice,
            (T_mean - n_sigma * T_std).ravel(),
            (T_mean + n_sigma * T_std).ravel(),
            color=line_T.get_color(),
            alpha=0.2,
        )
        axes[0].plot(
            r_slice,
            y_ref[mask, 0][order],
            "o",
            ms=3,
            alpha=0.4,
            color=line_T.get_color(),
        )

        (line_u,) = axes[1].plot(r_slice, u_mean.ravel(), "-", label=f"t*={t_star:.3f}")
        axes[1].fill_between(
            r_slice,
            (u_mean - n_sigma * u_std).ravel(),
            (u_mean + n_sigma * u_std).ravel(),
            color=line_u.get_color(),
            alpha=0.2,
        )
        axes[1].plot(
            r_slice,
            y_ref[mask, 1][order],
            "o",
            ms=3,
            alpha=0.4,
            color=line_u.get_color(),
        )

    axes[0].set_xlabel("r*")
    axes[0].set_ylabel("T*")
    axes[0].set_title(f"Temperature (dots=FDM, line=mean, band=+/-{n_sigma:g} sigma)")
    axes[0].legend()

    axes[1].set_xlabel("r*")
    axes[1].set_ylabel("u*")
    axes[1].set_title(f"Pore pressure (dots=FDM, line=mean, band=+/-{n_sigma:g} sigma)")
    axes[1].legend()

    fig.suptitle(f"Case {case_num}: {name} uncertainty quantification")
    fig.tight_layout()

    out_path = output_dir / f"case{case_num}_{name}_uncertainty.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")
    return out_path
