"""
Task 2 -- Forward PINN training and evaluation for Tf=50C, Ts=12C.

Trains a GeothermalPINN on the PDE residual plus initial/boundary
condition losses only (no labeled interior data, per the task spec),
then benchmarks the trained model against the matching FDM reference
case and saves comparison plots.
"""

from typing import Dict, Tuple

from src.models.mlp import DEFAULT_HIDDEN_LAYERS, DEFAULT_HIDDEN_WIDTH, build_mlp
from src.models.pinn import GeothermalPINN
from src.utils.config import CASE_PERMEABILITIES, DEFAULT_KS
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
    plot_model_pair,
    plot_profiles,
    plot_reference_pair,
)


def train_forward_model(
    case_num: int = DEFAULT_CASE_NUM,
    Ks: float = DEFAULT_KS,
    num_dom: int = DEFAULT_NUM_DOM,
    num_ic: int = DEFAULT_NUM_IC,
    num_bc: int = DEFAULT_NUM_BC,
    hidden_layers: int = DEFAULT_HIDDEN_LAYERS,
    hidden_width: int = DEFAULT_HIDDEN_WIDTH,
    epochs: int = 5000,
    seed: int = DEFAULT_SEED,
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
        seed: Seed for collocation sampling and network initialization.
        device: Device to train on ('cpu', 'cuda', or 'mps').

    Returns:
        Tuple (model, metrics) with the trained GeothermalPINN and the
        error metrics from `evaluate()` against the FDM reference.
    """
    X_ref, y_ref, points, (C1, C2, C3), (r_ref, t_ref) = prepare_case(
        case_num, Ks=Ks, num_dom=num_dom, num_ic=num_ic, num_bc=num_bc, seed=seed
    )
    domain_points, ic_points, bc_pile_points, bc_far_points = points

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

    T_ref = y_ref[:, 0:1]
    u_ref = y_ref[:, 1:2]
    metrics = model.evaluate(r_ref, t_ref, T_ref, u_ref)

    k = CASE_PERMEABILITIES[case_num]
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
