"""
Shared setup for Task 2 experiment scripts.

Bundles the steps every training experiment (grid search, ensemble,
curriculum) repeats: compute C1/C2/C3 for a case, load its FDM
reference, and sample the shared collocation points.
"""

from typing import Tuple

import torch

from src.utils.config import CASE_PERMEABILITIES, DEFAULT_KS
from src.utils.data_loader import prepare_training_data
from src.utils.physics_constants import PhysicsConstants
from src.utils.sampler import GeothermalSampler


def prepare_case(
    case_num: int,
    Ks: float = DEFAULT_KS,
    num_dom: int = 4000,
    num_ic: int = 200,
    num_bc: int = 200,
    seed: int = 42,
) -> Tuple:
    """
    Load the FDM reference and sample the shared collocation points for a case.

    Args:
        case_num: FDM case (1-5).
        Ks: Soil compressibility (Pa).
        num_dom: Interior collocation points.
        num_ic: Initial condition points.
        num_bc: Boundary points (per boundary).
        seed: Seed for reproducible collocation sampling (see
            `GeothermalSampler.sample`).

    Returns:
        Tuple (X_ref, y_ref, points, coeffs, eval_points):
            X_ref, y_ref: FDM reference (r*, t*) -> (T*, u*) arrays.
            points: (domain_points, ic_points, bc_pile_points, bc_far_points).
            coeffs: (C1, C2, C3).
            eval_points: (r_ref, t_ref) tensors for evaluate()/predict().
    """
    physics = PhysicsConstants()
    k = CASE_PERMEABILITIES[case_num]
    C1, C2, C3 = physics.calculate_physics_constants(k=k, Ks=Ks)

    result = prepare_training_data(case_num)
    if isinstance(result, dict):
        raise RuntimeError(f"Failed to load case {case_num}: {result['error_message']}")
    X_ref, y_ref = result

    r_min, r_max = float(X_ref[:, 0].min()), float(X_ref[:, 0].max())
    t_min, t_max = float(X_ref[:, 1].min()), float(X_ref[:, 1].max())
    sampler = GeothermalSampler(spans=[(r_min, r_max), (t_min, t_max)])

    # seed=... makes the LHS domain points reproducible too (scipy's
    # LatinHypercube keeps its own RNG that torch.manual_seed does not
    # reach) -- see src/utils/sampler.py.
    points = sampler.sample(num_dom=num_dom, num_ic=num_ic, num_bc=num_bc, seed=seed)

    r_ref = torch.tensor(X_ref[:, 0:1], dtype=torch.float32)
    t_ref = torch.tensor(X_ref[:, 1:2], dtype=torch.float32)

    return X_ref, y_ref, points, (C1, C2, C3), (r_ref, t_ref)
