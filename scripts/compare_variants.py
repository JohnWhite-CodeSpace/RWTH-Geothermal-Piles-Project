"""
Task 2 challenge comparison -- FDM reference vs three PINN variants.

Trains three PINN configurations on the same case, the same collocation
points, and the same initial network weights, so differences in the
results come only from the training strategy:

  - baseline: Adam only, uniform T*/u* loss weights.
  - weighted: Adam only, higher weight on the pore-pressure loss terms
    to counter the T*/u* magnitude imbalance (T* ~ O(1), u* ~ O(1e-3)).
  - lbfgs: baseline Adam run, then fine-tuned with L-BFGS.
  - best: weighted Adam run, then fine-tuned with L-BFGS.

Produces a heatmap grid (FDM + all variants, T* and u*), a bar chart of
relative L2 error per variant, and a training loss convergence plot.
All chart-generation code lives in `src/utils/plotting.py` -- this
script only trains the variants and calls into it.
"""

from typing import Dict

import torch

from src.models.mlp import DEFAULT_HIDDEN_LAYERS, DEFAULT_HIDDEN_WIDTH, build_mlp
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
    plot_loss_curves,
    plot_metric_comparison,
    plot_model_pair,
    plot_reference_pair,
    plot_variant_heatmaps,
)

# Deliberately NOT centralized: this is a specific, intentionally
# aggressive weight used only in this file's own "what if we pick too
# high a w_u" narrative, not a project-wide default worth sharing.
W_U_HIGH = 100.0


def _build_model(device: str, seed: int = DEFAULT_SEED) -> GeothermalPINN:
    """Build a fresh PINN with a fixed seed, so variants start identical."""
    torch.manual_seed(seed)
    net_u = build_mlp(
        in_dim=2,
        out_dim=2,
        hidden_layers=DEFAULT_HIDDEN_LAYERS,
        hidden_width=DEFAULT_HIDDEN_WIDTH,
    )
    return GeothermalPINN(net_u, device=device)


def run_comparison(
    case_num: int = DEFAULT_CASE_NUM,
    Ks: float = DEFAULT_KS,
    num_dom: int = DEFAULT_NUM_DOM,
    num_ic: int = DEFAULT_NUM_IC,
    num_bc: int = DEFAULT_NUM_BC,
    adam_epochs: int = 3000,
    lbfgs_epochs: int = 200,
    seed: int = DEFAULT_SEED,
    device: str = "cpu",
) -> Dict[str, Dict[str, float]]:
    """
    Train baseline / weighted / lbfgs / weighted+lbfgs variants and compare.

    Args:
        case_num: FDM case (1-5).
        Ks: Soil compressibility (Pa).
        num_dom: Interior collocation points.
        num_ic: Initial condition points.
        num_bc: Boundary points (per boundary).
        adam_epochs: Adam iterations for every variant's first stage.
        lbfgs_epochs: L-BFGS iterations for the fine-tuning stage.
        device: Device to train on.

    Returns:
        Dict mapping variant name -> its evaluate() metrics dict.
    """
    # Same collocation points for every variant, so differences in the
    # results come only from the training strategy, not sampling noise.
    X_ref, y_ref, points, (C1, C2, C3), (r_ref, t_ref) = prepare_case(
        case_num, Ks=Ks, num_dom=num_dom, num_ic=num_ic, num_bc=num_bc, seed=seed
    )
    domain_points, ic_points, bc_pile_points, bc_far_points = points
    T_ref = y_ref[:, 0:1]
    u_ref = y_ref[:, 1:2]

    def _train_adam(model: GeothermalPINN, w_T: float = 1.0, w_u: float = 1.0) -> None:
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
            w_T=w_T,
            w_u=w_u,
        )

    def _finetune_lbfgs(
        model: GeothermalPINN, w_T: float = 1.0, w_u: float = 1.0
    ) -> None:
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
            w_T=w_T,
            w_u=w_u,
        )

    variants: Dict[str, GeothermalPINN] = {}

    print("\n=== baseline: Adam, uniform weights ===")
    model = _build_model(device, seed)
    _train_adam(model)
    variants["baseline"] = model

    print(f"\n=== weighted: Adam, w_u={W_U_HIGH:.0f} ===")
    model = _build_model(device, seed)
    _train_adam(model, w_u=W_U_HIGH)
    variants["weighted"] = model

    print("\n=== lbfgs: baseline Adam + L-BFGS fine-tune ===")
    model = _build_model(device, seed)
    _train_adam(model)
    _finetune_lbfgs(model)
    variants["lbfgs"] = model

    print(f"\n=== best: weighted Adam (w_u={W_U_HIGH:.0f}) + L-BFGS fine-tune ===")
    model = _build_model(device, seed)
    _train_adam(model, w_u=W_U_HIGH)
    _finetune_lbfgs(model, w_u=W_U_HIGH)
    variants["best"] = model

    metrics: Dict[str, Dict[str, float]] = {}
    print("\n--- Final evaluation against FDM reference ---")
    for name, variant_model in variants.items():
        m = variant_model.evaluate(r_ref, t_ref, T_ref, u_ref)
        metrics[name] = m
        print(
            f"{name:10s}  T_rel_l2={m['T_rel_l2']:.4f}   u_rel_l2={m['u_rel_l2']:.4f}"
        )

    plot_variant_heatmaps(variants, X_ref, y_ref, case_num)
    plot_metric_comparison(metrics, case_num)
    plot_loss_curves(variants, case_num)

    return metrics


def run_weighting_ablation(
    case_num: int = DEFAULT_CASE_NUM,
    Ks: float = DEFAULT_KS,
    num_dom: int = DEFAULT_NUM_DOM,
    num_ic: int = DEFAULT_NUM_IC,
    num_bc: int = DEFAULT_NUM_BC,
    seed: int = DEFAULT_SEED,
    device: str = "cpu",
) -> Dict[str, Dict[str, float]]:
    """
    Test three ways to fix the "weighted" variant's poor T* accuracy.

    "weighted" (w_u=100, 3000 Adam epochs) traded T* accuracy for u*
    accuracy but hadn't fully converged. Each step below changes exactly
    one thing relative to that run, so the effect of each change is
    isolated rather than confounded:

      - more_epochs: same w_u=100, but 6000 Adam epochs instead of 3000,
        to see whether it just needed more time to settle.
      - lower_weight: w_u=30 instead of 100, still 3000 epochs, to see
        whether 100 was simply too aggressive.
      - short_lbfgs: the original weighted run, fine-tuned with only 30
        L-BFGS iterations instead of 200, to see whether the earlier
        200-iteration fine-tune was overshooting rather than helping.

    Args:
        case_num: FDM case (1-5).
        Ks: Soil compressibility (Pa).
        num_dom: Interior collocation points.
        num_ic: Initial condition points.
        num_bc: Boundary points (per boundary).
        seed: Seed for collocation sampling and network initialization.
        device: Device to train on.

    Returns:
        Dict mapping variant name -> its evaluate() metrics dict.
    """
    X_ref, y_ref, points, (C1, C2, C3), (r_ref, t_ref) = prepare_case(
        case_num, Ks=Ks, num_dom=num_dom, num_ic=num_ic, num_bc=num_bc, seed=seed
    )
    domain_points, ic_points, bc_pile_points, bc_far_points = points
    T_ref = y_ref[:, 0:1]
    u_ref = y_ref[:, 1:2]

    def _train(model, epochs, optimizer, w_u):
        model.train_net(
            domain_points=domain_points,
            ic_points=ic_points,
            bc_pile_points=bc_pile_points,
            bc_far_points=bc_far_points,
            C1=C1,
            C2=C2,
            C3=C3,
            epochs=epochs,
            optimizer=optimizer,
            w_T=1.0,
            w_u=w_u,
        )

    variants: Dict[str, GeothermalPINN] = {}

    print(f"\n=== weighted (reference): Adam 3000ep, w_u={W_U_HIGH:.0f} ===")
    model = _build_model(device, seed)
    _train(model, 3000, torch.optim.Adam, W_U_HIGH)
    variants["weighted"] = model

    print(f"\n=== more_epochs: Adam 6000ep, w_u={W_U_HIGH:.0f} ===")
    model = _build_model(device, seed)
    _train(model, 6000, torch.optim.Adam, W_U_HIGH)
    variants["more_epochs"] = model

    print("\n=== lower_weight: Adam 3000ep, w_u=30 ===")
    model = _build_model(device, seed)
    _train(model, 3000, torch.optim.Adam, 30.0)
    variants["lower_weight"] = model

    print(f"\n=== short_lbfgs: Adam 3000ep (w_u={W_U_HIGH:.0f}) + L-BFGS 30it ===")
    model = _build_model(device, seed)
    _train(model, 3000, torch.optim.Adam, W_U_HIGH)
    _train(model, 30, torch.optim.LBFGS, W_U_HIGH)
    variants["short_lbfgs"] = model

    metrics: Dict[str, Dict[str, float]] = {}
    print("\n--- Final evaluation against FDM reference ---")
    for name, variant_model in variants.items():
        m = variant_model.evaluate(r_ref, t_ref, T_ref, u_ref)
        metrics[name] = m
        print(
            f"{name:14s}  T_rel_l2={m['T_rel_l2']:.4f}   u_rel_l2={m['u_rel_l2']:.4f}"
        )

    plot_reference_pair(X_ref, y_ref, case_num)
    for name, variant_model in variants.items():
        plot_model_pair(variant_model, X_ref, case_num, name)
    plot_metric_comparison(metrics, case_num)
    plot_loss_curves(variants, case_num)

    return metrics


if __name__ == "__main__":
    run_comparison()
