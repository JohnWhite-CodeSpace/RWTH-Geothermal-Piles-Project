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

# Single source of truth for the project's default case/experiment
# setup. Every script should import these rather than repeating the
# numbers -- otherwise two scripts can silently drift apart (e.g. one
# using num_dom=4000 and another 5000) and their results stop being a
# fair comparison without anyone noticing.
DEFAULT_CASE_NUM = 3
DEFAULT_SEED = 42
DEFAULT_NUM_DOM = 4000
DEFAULT_NUM_IC = 200
DEFAULT_NUM_BC = 200


def prepare_case(
    case_num: int = DEFAULT_CASE_NUM,
    Ks: float = DEFAULT_KS,
    num_dom: int = DEFAULT_NUM_DOM,
    num_ic: int = DEFAULT_NUM_IC,
    num_bc: int = DEFAULT_NUM_BC,
    seed: int = DEFAULT_SEED,
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
