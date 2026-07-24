"""
Uncertainty quantification via a deep ensemble of PINNs (Task 2 challenge).

Trains N independently-seeded PINNs with the same architecture and
hyperparameters (defaults match the best combination
`scripts/grid_search.py` found: w_u=30, adam_epochs=6000,
lbfgs_epochs=30), then combines their predictions into a mean and
standard deviation via `PINNEnsemble`. Reports point-prediction error
on the mean plus calibration: what fraction of the true values
actually fall inside the predicted uncertainty band (coverage near
0.68/0.95 at 1/2 sigma would indicate a trustworthy uncertainty
estimate, not just decoration on top of a point prediction).

Usage:
    python -m scripts.ensemble
"""

from typing import Dict, List, Sequence

import torch

from src.models.ensemble import PINNEnsemble
from src.models.mlp import DEFAULT_ARCHITECTURE, build_mlp
from src.models.pinn import GeothermalPINN
from src.utils.experiment import (
    DEFAULT_CASE_NUM,
    DEFAULT_NUM_BC,
    DEFAULT_NUM_DOM,
    DEFAULT_NUM_IC,
    DEFAULT_SEED,
    prepare_case,
)
from src.utils.plotting import plot_uncertainty_band


def _train_member(
    seed: int,
    architecture: Sequence[int],
    domain_points: torch.Tensor,
    ic_points: torch.Tensor,
    bc_pile_points: torch.Tensor,
    bc_far_points: torch.Tensor,
    C1: float,
    C2: float,
    C3: float,
    w_u: float,
    adam_epochs: int,
    lbfgs_epochs: int,
    device: str,
    log_interval: int,
) -> GeothermalPINN:
    """Train one ensemble member from a fresh, seeded initialization."""
    torch.manual_seed(seed)
    net_u = build_mlp(in_dim=2, out_dim=2, hidden_width=list(architecture))
    model = GeothermalPINN(net_u, device=device)

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


def run_ensemble(
    case_num: int = DEFAULT_CASE_NUM,
    n_models: int = 5,
    w_u: float = 30.0,
    adam_epochs: int = 6000,
    lbfgs_epochs: int = 30,
    architecture: Sequence[int] = DEFAULT_ARCHITECTURE,
    num_dom: int = DEFAULT_NUM_DOM,
    num_ic: int = DEFAULT_NUM_IC,
    num_bc: int = DEFAULT_NUM_BC,
    base_seed: int = DEFAULT_SEED,
    device: str = "cpu",
    log_interval: int = 1000,
) -> Dict[str, float]:
    """
    Train an ensemble and report both accuracy and calibration.

    Every ensemble member trains on the SAME collocation points (seeded
    from `base_seed`) -- only each member's initial weights differ (one
    distinct seed per member) -- so the resulting spread in predictions
    reflects genuine optimization/initialization variability, not
    sampling noise.

    Estimated runtime on CPU: roughly n_models x (adam_epochs/1000 x
    35s + a few seconds per 10 L-BFGS iterations). Defaults (5 members,
    6000 Adam epochs each) take on the order of 15-20 minutes total.

    Args:
        case_num: FDM case (1-5).
        n_models: Number of ensemble members to train.
        w_u: Pore-pressure loss weight (see grid_search.py for how the
            default was chosen).
        adam_epochs: Adam iterations per member.
        lbfgs_epochs: L-BFGS fine-tuning iterations per member (0 to skip).
        architecture: Hidden-layer widths, shared by every member.
        num_dom: Interior collocation points.
        num_ic: Initial condition points.
        num_bc: Boundary points (per boundary).
        base_seed: Collocation points are sampled once using this seed;
            member i is initialized with seed `base_seed + i + 1`.
        device: Device to train on.
        log_interval: Training log print interval.

    Returns:
        The ensemble's evaluate() metrics dict (accuracy + coverage).
    """
    X_ref, y_ref, points, (C1, C2, C3), (r_ref, t_ref) = prepare_case(
        case_num, num_dom=num_dom, num_ic=num_ic, num_bc=num_bc, seed=base_seed
    )
    domain_points, ic_points, bc_pile_points, bc_far_points = points
    T_ref = y_ref[:, 0:1]
    u_ref = y_ref[:, 1:2]

    models: List[GeothermalPINN] = []
    for i in range(n_models):
        member_seed = base_seed + i + 1
        print(f"\n=== Ensemble member {i + 1}/{n_models} (seed={member_seed}) ===")
        model = _train_member(
            seed=member_seed,
            architecture=architecture,
            domain_points=domain_points,
            ic_points=ic_points,
            bc_pile_points=bc_pile_points,
            bc_far_points=bc_far_points,
            C1=C1,
            C2=C2,
            C3=C3,
            w_u=w_u,
            adam_epochs=adam_epochs,
            lbfgs_epochs=lbfgs_epochs,
            device=device,
            log_interval=log_interval,
        )
        models.append(model)

    ensemble = PINNEnsemble(models)
    metrics = ensemble.evaluate(r_ref, t_ref, T_ref, u_ref)

    print("\n--- Ensemble evaluation against FDM reference ---")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
    print(
        "\n(coverage_1sigma near 0.68 and coverage_2sigma near 0.95 would "
        "indicate a well-calibrated uncertainty estimate)"
    )

    plot_uncertainty_band(ensemble, X_ref, y_ref, case_num, "ensemble")

    return metrics


if __name__ == "__main__":
    run_ensemble()
