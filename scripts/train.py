"""
Task 2 -- Forward PINN training and evaluation for Tf=50C, Ts=12C.

Trains a GeothermalPINN on the PDE residual plus initial/boundary
condition losses only (no labeled interior data, per the task spec),
then benchmarks the trained model against the matching FDM reference
case and saves comparison plots.
"""

from typing import Dict, Tuple

import torch

from src.models.mlp import build_mlp
from src.models.pinn import GeothermalPINN
from src.utils.config import CASE_PERMEABILITIES, DEFAULT_KS
from src.utils.physics_constants import PhysicsConstants
from src.utils.data_loader import prepare_training_data
from src.utils.plotting import (
    plot_error_heatmap,
    plot_model_pair,
    plot_profiles,
    plot_reference_pair,
)
from src.utils.sampler import GeothermalSampler


def train_forward_model(
    case_num: int = 3,
    Ks: float = DEFAULT_KS,
    num_dom: int = 4000,
    num_ic: int = 200,
    num_bc: int = 200,
    hidden_layers: int = 6,
    hidden_width: int = 64,
    epochs: int = 5000,
    device: str = "cpu",
) -> Tuple[GeothermalPINN, Dict[str, float]]:
    """
    Train and evaluate the forward PINN for Tf=50C, Ts=12C.

    Args:
        case_num: FDM case (1-5) whose permeability to use and whose
            reference solution to benchmark against.
        Ks: Soil compressibility (Pa).
        num_dom: Number of interior collocation points.
        num_ic: Number of initial condition points.
        num_bc: Number of boundary points (per boundary).
        hidden_layers: Hidden layers in the underlying MLP.
        hidden_width: Neurons per hidden layer.
        epochs: Number of Adam training iterations.
        device: Device to train on ('cpu', 'cuda', or 'mps').

    Returns:
        Tuple (model, metrics) with the trained GeothermalPINN and the
        error metrics from `evaluate()` against the FDM reference.
    """
    physics = PhysicsConstants()
    k = CASE_PERMEABILITIES[case_num]

    # t_c/u_c default to this instance's own values, matching the
    # nondimensionalization data_loader.py applies to the FDM reference.
    C1, C2, C3 = physics.calculate_physics_constants(k=k, Ks=Ks)

    # Reference FDM solution for this case, already nondimensionalized
    # with the same r*/t*/T*/u* scaling as the PINN (see data_loader.py).
    result = prepare_training_data(case_num)
    if isinstance(result, dict):
        raise RuntimeError(f"Failed to load case {case_num}: {result['error_message']}")
    X_ref, y_ref = result

    # Sample the collocation domain over exactly the (r*, t*) range the
    # FDM reference covers, so training and evaluation share one domain.
    r_min, r_max = float(X_ref[:, 0].min()), float(X_ref[:, 0].max())
    t_min, t_max = float(X_ref[:, 1].min()), float(X_ref[:, 1].max())
    sampler = GeothermalSampler(spans=[(r_min, r_max), (t_min, t_max)])
    domain_points, ic_points, bc_pile_points, bc_far_points = sampler.sample(
        num_dom=num_dom, num_ic=num_ic, num_bc=num_bc, seed=42
    )

    net_u = build_mlp(
        in_dim=2, out_dim=2, hidden_layers=hidden_layers, hidden_width=hidden_width
    )
    model = GeothermalPINN(net_u, device=device)

    model.train_net(
        domain_points=domain_points,
        ic_points=ic_points,
        bc_pile_points=bc_pile_points,
        bc_far_points=bc_far_points,
        C1=C1,
        C2=C2,
        C3=C3,
        epochs=epochs,
    )

    r_ref = torch.tensor(X_ref[:, 0:1], dtype=torch.float32)
    t_ref = torch.tensor(X_ref[:, 1:2], dtype=torch.float32)
    T_ref = y_ref[:, 0:1]
    u_ref = y_ref[:, 1:2]
    metrics = model.evaluate(r_ref, t_ref, T_ref, u_ref)

    print(f"\nEvaluation against FDM case {case_num} (k={k:.0e}):")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4e}")

    plot_reference_pair(X_ref, y_ref, case_num)
    plot_model_pair(model, X_ref, case_num, "forward")
    plot_error_heatmap(model, X_ref, y_ref, case_num, "forward")
    plot_profiles(model, X_ref, y_ref, case_num)

    return model, metrics


if __name__ == "__main__":
    train_forward_model()
