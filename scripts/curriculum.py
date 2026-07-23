"""
Curriculum ("staged loss weighting") training for the forward PINN.

`compare_variants.py`'s "weighted" variant and `grid_search.py` both
use a single, fixed w_u for an entire run. This instead ramps w_u up
over a sequence of stages within ONE continuous training run --
starting close to a T*-only fit (low w_u, since the pressure PDE's
source term dT*/dtau depends on T* already being reasonable) and
gradually increasing the pore-pressure loss's weight.
`GeothermalPINN.train_net()` already supports being called repeatedly
on the same instance (continuing from the current weights, appending
to the same loss history) -- a curriculum is just a sequence of such
calls with an increasing w_u. No changes to GeothermalPINN itself were
needed for this.

The default stages sum to 6000 Adam epochs (1500+1500+3000) plus 30
L-BFGS iterations, matching `grid_search.py`'s best fixed-weight combo
(w_u=30, adam_epochs=6000, lbfgs_epochs=30) in total training budget,
so the two are a fair, apples-to-apples comparison: does *how* you
reach w_u=30 matter, not just reaching it.

Usage:
    python -m scripts.curriculum
"""

from typing import Dict, Sequence, Tuple

import torch

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
from src.utils.plotting import plot_loss_curves, plot_model_pair, plot_reference_pair

# (w_u, adam_epochs) per stage, applied in order.
DEFAULT_STAGES: Sequence[Tuple[float, int]] = (
    (1.0, 1500),
    (10.0, 1500),
    (30.0, 3000),
)
DEFAULT_LBFGS_EPOCHS = 30


def run_curriculum(
    case_num: int = DEFAULT_CASE_NUM,
    stages: Sequence[Tuple[float, int]] = DEFAULT_STAGES,
    lbfgs_epochs: int = DEFAULT_LBFGS_EPOCHS,
    architecture: Sequence[int] = DEFAULT_ARCHITECTURE,
    num_dom: int = DEFAULT_NUM_DOM,
    num_ic: int = DEFAULT_NUM_IC,
    num_bc: int = DEFAULT_NUM_BC,
    seed: int = DEFAULT_SEED,
    device: str = "cpu",
    log_interval: int = 500,
) -> Dict[str, float]:
    """
    Train one PINN through a curriculum of increasing w_u stages.

    Args:
        case_num: FDM case (1-5).
        stages: Sequence of (w_u, adam_epochs) stages applied in order,
            each continuing training from the previous stage's weights.
        lbfgs_epochs: L-BFGS fine-tuning iterations after the last Adam
            stage, at the final stage's w_u (0 to skip).
        architecture: Hidden-layer widths.
        num_dom: Interior collocation points.
        num_ic: Initial condition points.
        num_bc: Boundary points (per boundary).
        seed: Seed for collocation sampling and network initialization.
        device: Device to train on.
        log_interval: Training log print interval.

    Returns:
        The trained model's evaluate() metrics dict.
    """
    X_ref, y_ref, points, (C1, C2, C3), (r_ref, t_ref) = prepare_case(
        case_num, num_dom=num_dom, num_ic=num_ic, num_bc=num_bc, seed=seed
    )
    domain_points, ic_points, bc_pile_points, bc_far_points = points
    T_ref = y_ref[:, 0:1]
    u_ref = y_ref[:, 1:2]

    torch.manual_seed(seed)
    net_u = build_mlp(in_dim=2, out_dim=2, hidden_width=list(architecture))
    model = GeothermalPINN(net_u, device=device)

    last_w_u = 1.0
    for stage_num, (w_u, epochs) in enumerate(stages, start=1):
        print(
            f"\n=== Curriculum stage {stage_num}/{len(stages)}: "
            f"w_u={w_u:g}, {epochs} Adam epochs ==="
        )
        model.train_net(
            domain_points=domain_points,
            ic_points=ic_points,
            bc_pile_points=bc_pile_points,
            bc_far_points=bc_far_points,
            C1=C1,
            C2=C2,
            C3=C3,
            epochs=epochs,
            optimizer=torch.optim.Adam,
            log_interval=log_interval,
            w_T=1.0,
            w_u=w_u,
        )
        last_w_u = w_u

    if lbfgs_epochs > 0:
        print(
            f"\n=== Curriculum fine-tune: L-BFGS {lbfgs_epochs}it at w_u={last_w_u:g} ==="
        )
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
            w_u=last_w_u,
        )

    metrics = model.evaluate(r_ref, t_ref, T_ref, u_ref)
    print("\n--- Curriculum model evaluation against FDM reference ---")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
    print(
        "\nCompare against case{n}_grid_search_results.csv's row for "
        "w_u=30, adam_epochs=6000, lbfgs_epochs=30 -- same total budget, "
        "fixed weight throughout instead of staged.".format(n=case_num)
    )

    plot_reference_pair(X_ref, y_ref, case_num)
    plot_model_pair(model, X_ref, case_num, "curriculum")
    plot_loss_curves({"curriculum": model}, case_num)

    return metrics


if __name__ == "__main__":
    run_curriculum()
