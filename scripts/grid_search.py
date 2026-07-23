"""
Grid search over PINN training hyperparameters (Task 2 challenge).

Trains one PINN per (w_u, adam_epochs, lbfgs_epochs) combination, all
sharing the same seeded collocation points and the same seeded initial
network weights, and reports T*/u* error against the FDM reference for
each combination. This is meant to be run locally rather than in an
assistant turn -- a full default grid takes on the order of 30-40
minutes on CPU (see `run_grid_search`'s docstring for the estimate, and
shrink the grid via its arguments for a quicker pass).

Reproducibility: `GeothermalSampler.sample(..., seed=...)` seeds both
torch's RNG and scipy's LatinHypercube engine (the latter used to be
uncontrolled by `torch.manual_seed`, so LHS domain points differed
between separate runs even with "the same seed"). With a fixed `seed`
here, rerunning this script reproduces the same collocation points and
network initialization every time.

Every trained model's weights are saved to
`data/processed/checkpoints/`, so later analysis (e.g. an error
heatmap for one specific combination) doesn't require retraining.

Usage:
    python -m scripts.grid_search               # full (w_u, epochs, lbfgs) grid
    python -m scripts.grid_search --architecture  # architecture-only sweep, see
                                                    # run_architecture_search()
"""

import csv
import itertools
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Sequence

import torch

from src.models.mlp import DEFAULT_ARCHITECTURE, build_mlp
from src.models.pinn import GeothermalPINN
from src.utils.config import DEFAULT_KS
from src.utils.experiment import (
    DEFAULT_CASE_NUM,
    DEFAULT_NUM_BC,
    DEFAULT_NUM_DOM,
    DEFAULT_NUM_IC,
    DEFAULT_SEED,
    prepare_case,
)
from src.utils.plotting import (
    plot_error_heatmap,
    plot_grid_heatmap,
    plot_model_pair,
    plot_reference_pair,
)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

Architecture = Sequence[int]


@dataclass
class GridPoint:
    """One hyperparameter combination and its resulting error metrics."""

    w_u: float
    adam_epochs: int
    lbfgs_epochs: int
    architecture: str
    T_mse: float = 0.0
    T_rel_l2: float = 0.0
    T_nrmse: float = 0.0
    u_mse: float = 0.0
    u_rel_l2: float = 0.0
    u_nrmse: float = 0.0
    T_pde_residual_rms: float = 0.0
    u_pde_residual_rms: float = 0.0


def _architecture_label(architecture: Architecture) -> str:
    """Human-readable, filename-safe label for an architecture, e.g. '64-64-64'."""
    return "-".join(str(w) for w in architecture)


def _checkpoint_path(case_num: int, point_id: str) -> Path:
    return CHECKPOINT_DIR / f"case{case_num}_{point_id}.pt"


def _build_model(seed: int, device: str, architecture: Architecture) -> GeothermalPINN:
    """Build a fresh PINN with a fixed seed, so every run starts identical."""
    torch.manual_seed(seed)
    net_u = build_mlp(in_dim=2, out_dim=2, hidden_width=list(architecture))
    return GeothermalPINN(net_u, device=device)


def _train_one(
    w_u: float,
    adam_epochs: int,
    lbfgs_epochs: int,
    architecture: Architecture,
    domain_points: torch.Tensor,
    ic_points: torch.Tensor,
    bc_pile_points: torch.Tensor,
    bc_far_points: torch.Tensor,
    C1: float,
    C2: float,
    C3: float,
    seed: int,
    device: str,
    log_interval: int,
) -> GeothermalPINN:
    """Train a single PINN for one hyperparameter combination."""
    model = _build_model(seed, device, architecture)
    model.train_net(
        domain_points=domain_points,
        ic_points=ic_points,
        bc_pile_points=bc_pile_points,
        bc_far_points=bc_far_points,
        C1=C1,
        C2=C2,
        C3=C3,
        epochs=adam_epochs,
        optimizer=torch.optim.Adam,
        log_interval=log_interval,
        w_T=1.0,
        w_u=w_u,
    )
    if lbfgs_epochs > 0:
        model.train_net(
            domain_points=domain_points,
            ic_points=ic_points,
            bc_pile_points=bc_pile_points,
            bc_far_points=bc_far_points,
            C1=C1,
            C2=C2,
            C3=C3,
            epochs=lbfgs_epochs,
            optimizer=torch.optim.LBFGS,
            log_interval=log_interval,
            w_T=1.0,
            w_u=w_u,
        )
    return model


def run_grid_search(
    case_num: int = DEFAULT_CASE_NUM,
    Ks: float = DEFAULT_KS,
    num_dom: int = DEFAULT_NUM_DOM,
    num_ic: int = DEFAULT_NUM_IC,
    num_bc: int = DEFAULT_NUM_BC,
    w_u_values: Sequence[float] = (1.0, 30.0, 100.0),
    adam_epochs_values: Sequence[int] = (3000, 6000),
    lbfgs_epochs_values: Sequence[int] = (0, 30),
    architecture: Architecture = DEFAULT_ARCHITECTURE,
    seed: int = DEFAULT_SEED,
    device: str = "cpu",
    log_interval: int = 500,
) -> List[GridPoint]:
    """
    Train one PINN per combination of (w_u, adam_epochs, lbfgs_epochs).

    Architecture is fixed for this sweep (see `run_architecture_search`
    for sweeping architecture instead, once the best w_u/epochs are
    known -- crossing all four dimensions here would multiply runtime
    by len(architectures) on top of the already-long default grid).

    Every combination shares the same seeded collocation points and the
    same seeded initial network weights (both re-derived from `seed`
    right before use), so differences between combinations come only
    from the hyperparameters being swept, not from sampling noise.

    Estimated runtime on CPU: roughly 35s per 1000 Adam epochs, plus a
    few seconds per 10 L-BFGS iterations. The default grid (3 w_u x 2
    adam_epochs x 2 lbfgs_epochs = 12 points, up to 6000 Adam epochs
    each) takes on the order of 30-40 minutes total. Pass smaller
    sequences to any of `w_u_values` / `adam_epochs_values` /
    `lbfgs_epochs_values` for a quicker pass.

    Args:
        case_num: FDM case (1-5).
        Ks: Soil compressibility (Pa).
        num_dom: Interior collocation points.
        num_ic: Initial condition points.
        num_bc: Boundary points (per boundary).
        w_u_values: Pore-pressure loss weights to try.
        adam_epochs_values: Adam iteration counts to try.
        lbfgs_epochs_values: L-BFGS fine-tuning iteration counts to try
            (0 = no fine-tuning stage).
        architecture: Fixed hidden-layer widths for every combination.
        seed: Seed controlling both collocation sampling and network
            initialization, for reproducible runs.
        device: Device to train on.
        log_interval: Training log print interval (raise this to cut
            console output over a long grid search).

    Returns:
        List of GridPoint results, one per hyperparameter combination.
    """
    X_ref, y_ref, points, (C1, C2, C3), (r_ref, t_ref) = prepare_case(
        case_num, Ks, num_dom, num_ic, num_bc, seed
    )
    domain_points, ic_points, bc_pile_points, bc_far_points = points
    T_ref = y_ref[:, 0:1]
    u_ref = y_ref[:, 1:2]

    combos = list(
        itertools.product(w_u_values, adam_epochs_values, lbfgs_epochs_values)
    )
    results: List[GridPoint] = []

    for i, (w_u, adam_epochs, lbfgs_epochs) in enumerate(combos, start=1):
        point_id = f"wu{w_u:g}_a{adam_epochs}_l{lbfgs_epochs}"
        print(f"\n=== [{i}/{len(combos)}] {point_id} ===")
        model = _train_one(
            w_u=w_u,
            adam_epochs=adam_epochs,
            lbfgs_epochs=lbfgs_epochs,
            architecture=architecture,
            domain_points=domain_points,
            ic_points=ic_points,
            bc_pile_points=bc_pile_points,
            bc_far_points=bc_far_points,
            C1=C1,
            C2=C2,
            C3=C3,
            seed=seed,
            device=device,
            log_interval=log_interval,
        )
        m = model.evaluate(r_ref, t_ref, T_ref, u_ref)
        point = GridPoint(
            w_u=w_u,
            adam_epochs=adam_epochs,
            lbfgs_epochs=lbfgs_epochs,
            architecture=_architecture_label(architecture),
            T_mse=m["T_mse"],
            T_rel_l2=m["T_rel_l2"],
            T_nrmse=m["T_nrmse"],
            u_mse=m["u_mse"],
            u_rel_l2=m["u_rel_l2"],
            u_nrmse=m["u_nrmse"],
            T_pde_residual_rms=m["T_pde_residual_rms"],
            u_pde_residual_rms=m["u_pde_residual_rms"],
        )
        results.append(point)
        print(
            f"  -> T_rel_l2={point.T_rel_l2:.4f} (NRMSE={point.T_nrmse:.3f})   "
            f"u_rel_l2={point.u_rel_l2:.4f} (NRMSE={point.u_nrmse:.3f})   "
            f"PDE resid T/u={point.T_pde_residual_rms:.3e}/{point.u_pde_residual_rms:.3e}"
        )

        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(model.net_u.state_dict(), _checkpoint_path(case_num, point_id))

    save_results_csv(results, case_num, "grid_search")

    best = min(results, key=lambda p: p.T_nrmse + p.u_nrmse)
    print(f"\nBest combo by (T_nrmse + u_nrmse): {best}")

    plot_grid_heatmap(results, case_num)
    plot_grid_heatmap(results, case_num, metric_names=("T_nrmse", "u_nrmse"))
    plot_reference_pair(X_ref, y_ref, case_num)

    best_id = f"wu{best.w_u:g}_a{best.adam_epochs}_l{best.lbfgs_epochs}"
    best_model = _build_model(seed, device, architecture)
    best_model.net_u.load_state_dict(torch.load(_checkpoint_path(case_num, best_id)))
    plot_model_pair(best_model, X_ref, case_num, f"best_{best_id}")
    plot_error_heatmap(best_model, X_ref, y_ref, case_num, f"best_{best_id}")

    return results


def run_architecture_search(
    case_num: int = DEFAULT_CASE_NUM,
    Ks: float = DEFAULT_KS,
    num_dom: int = DEFAULT_NUM_DOM,
    num_ic: int = DEFAULT_NUM_IC,
    num_bc: int = DEFAULT_NUM_BC,
    w_u: float = 30.0,
    adam_epochs: int = 6000,
    lbfgs_epochs: int = 30,
    architectures: Sequence[Architecture] = (
        DEFAULT_ARCHITECTURE,  # current default: 6 layers x 64 (uniform)
        (32,) * 10,  # deeper, narrower
        (128,) * 4,  # shallower, wider
        (128, 64, 32, 32, 64, 128),  # hourglass
        (32, 64, 128, 128, 64, 32),  # inverse hourglass
    ),
    seed: int = DEFAULT_SEED,
    device: str = "cpu",
    log_interval: int = 500,
) -> List[GridPoint]:
    """
    Sweep MLP architecture at a fixed, already-good (w_u, epochs) point.

    Rather than crossing architecture with the full (w_u, adam_epochs,
    lbfgs_epochs) grid -- which would multiply run_grid_search's already
    long runtime by len(architectures) -- this fixes w_u/adam_epochs/
    lbfgs_epochs at the best combination `run_grid_search` found (its
    defaults below match that) and only varies the network shape. This
    directly answers "would a different neuron arrangement help", at a
    cost proportional to len(architectures) rather than the full cross
    product.

    Estimated runtime on CPU with the default 5 architectures and
    adam_epochs=6000: roughly 5 x (6000 epochs' worth of Adam + 30
    L-BFGS iterations), i.e. on the order of 20-25 minutes total.

    Args:
        case_num: FDM case (1-5).
        Ks: Soil compressibility (Pa).
        num_dom: Interior collocation points.
        num_ic: Initial condition points.
        num_bc: Boundary points (per boundary).
        w_u: Pore-pressure loss weight, held fixed across architectures.
        adam_epochs: Adam iterations, held fixed across architectures.
        lbfgs_epochs: L-BFGS fine-tuning iterations, held fixed (0 to
            skip fine-tuning).
        architectures: Hidden-layer width sequences to try. Each entry
            is a tuple of per-layer widths -- its length is the number
            of hidden layers.
        seed: Seed controlling both collocation sampling and network
            initialization, for reproducible runs.
        device: Device to train on.
        log_interval: Training log print interval.

    Returns:
        List of GridPoint results, one per architecture.
    """
    X_ref, y_ref, points, (C1, C2, C3), (r_ref, t_ref) = prepare_case(
        case_num, Ks, num_dom, num_ic, num_bc, seed
    )
    domain_points, ic_points, bc_pile_points, bc_far_points = points
    T_ref = y_ref[:, 0:1]
    u_ref = y_ref[:, 1:2]

    results: List[GridPoint] = []

    for i, architecture in enumerate(architectures, start=1):
        label = _architecture_label(architecture)
        point_id = f"arch_{label}"
        print(f"\n=== [{i}/{len(architectures)}] architecture={label} ===")
        model = _train_one(
            w_u=w_u,
            adam_epochs=adam_epochs,
            lbfgs_epochs=lbfgs_epochs,
            architecture=architecture,
            domain_points=domain_points,
            ic_points=ic_points,
            bc_pile_points=bc_pile_points,
            bc_far_points=bc_far_points,
            C1=C1,
            C2=C2,
            C3=C3,
            seed=seed,
            device=device,
            log_interval=log_interval,
        )
        m = model.evaluate(r_ref, t_ref, T_ref, u_ref)
        point = GridPoint(
            w_u=w_u,
            adam_epochs=adam_epochs,
            lbfgs_epochs=lbfgs_epochs,
            architecture=label,
            T_mse=m["T_mse"],
            T_rel_l2=m["T_rel_l2"],
            T_nrmse=m["T_nrmse"],
            u_mse=m["u_mse"],
            u_rel_l2=m["u_rel_l2"],
            u_nrmse=m["u_nrmse"],
            T_pde_residual_rms=m["T_pde_residual_rms"],
            u_pde_residual_rms=m["u_pde_residual_rms"],
        )
        results.append(point)
        print(
            f"  -> T_rel_l2={point.T_rel_l2:.4f} (NRMSE={point.T_nrmse:.3f})   "
            f"u_rel_l2={point.u_rel_l2:.4f} (NRMSE={point.u_nrmse:.3f})   "
            f"PDE resid T/u={point.T_pde_residual_rms:.3e}/{point.u_pde_residual_rms:.3e}"
        )

        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(model.net_u.state_dict(), _checkpoint_path(case_num, point_id))

        plot_model_pair(model, X_ref, case_num, point_id)
        plot_error_heatmap(model, X_ref, y_ref, case_num, point_id)

    save_results_csv(results, case_num, "architecture_search")

    best = min(results, key=lambda p: p.T_nrmse + p.u_nrmse)
    print(f"\nBest architecture by (T_nrmse + u_nrmse): {best}")

    plot_reference_pair(X_ref, y_ref, case_num)

    return results


def save_results_csv(results: List[GridPoint], case_num: int, tag: str) -> Path:
    """Save all grid points to a CSV for later analysis."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"case{case_num}_{tag}_results.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for point in results:
            writer.writerow(asdict(point))
    print(f"Saved results to {out_path}")
    return out_path


if __name__ == "__main__":
    # Shrink these sequences for a faster first pass, e.g.:
    #   run_grid_search(w_u_values=(1.0, 30.0), adam_epochs_values=(3000,))
    run_grid_search()

    # Once you know the best (w_u, adam_epochs, lbfgs_epochs), uncomment
    # this to test whether the network's shape matters too:
    # run_architecture_search()
