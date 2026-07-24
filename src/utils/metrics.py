"""
Canonical error metrics for comparing PINN predictions against FDM data.

Always evaluates in the same nondimensional (T*, u*) space, regardless of
what internal scale a given model trains with, so results from different
architectures (e.g. a single shared-output network vs. decoupled
temperature/pressure networks with their own physically-derived u_ref) are
directly comparable. Feed this module physical-unit fields (degrees C,
Pa); the nondimensionalization is applied once, here, so every reported
number in the project comes from the same definition.
"""

from typing import Dict

import numpy as np

from src.utils.config import PhysicsConstants

PHYS_CONST = PhysicsConstants()


def to_nondimensional_T(T_phys: np.ndarray) -> np.ndarray:
    """Convert physical temperature (C) to T* = (T - T_s) / delta_T."""
    return (T_phys - PHYS_CONST.T_s) / PHYS_CONST.delta_T


def to_nondimensional_u(u_phys: np.ndarray) -> np.ndarray:
    """Convert physical excess pore pressure (Pa) to u* = u / u_c."""
    return u_phys / PHYS_CONST.u_c


def error_metrics(pred: np.ndarray, true: np.ndarray) -> Dict[str, float]:
    """
    Compute MSE, relative L2 norm, and NRMSE between two arrays.

    NRMSE (RMSE normalized by the max magnitude of `true`) is reported
    alongside relative L2 because relative L2 inflates for fields that are
    close to zero almost everywhere (e.g. u*), while NRMSE stays
    well-behaved in that regime.

    Args:
        pred: Predicted values.
        true: Reference values, same shape as `pred`.

    Returns:
        Dict with keys "mse", "rel_l2", "nrmse".
    """
    diff = pred - true
    mse = float(np.mean(diff**2))
    rel_l2 = float(np.linalg.norm(diff) / (np.linalg.norm(true) + 1e-12))
    nrmse = float(np.sqrt(mse) / (np.max(np.abs(true)) + 1e-12))
    return {"mse": mse, "rel_l2": rel_l2, "nrmse": nrmse}


def physical_error_metrics(
    T_pred_phys: np.ndarray,
    T_ref_phys: np.ndarray,
    u_pred_phys: np.ndarray,
    u_ref_phys: np.ndarray,
) -> Dict[str, float]:
    """
    Canonical comparison entry point for any model's physical-unit output.

    Use this for every model we want to compare head-to-head (single
    shared-network PINN, decoupled T/u PINN, ...) instead of each script
    computing its own relative-L2 formula -- that's what previously made
    our T* numbers and a colleague's T numbers look different even though
    the underlying prediction quality wasn't: relative L2 on physical T
    (which includes the T_s offset) is not the same quantity as relative
    L2 on nondimensional T*.

    Args:
        T_pred_phys: Predicted temperature, degrees C.
        T_ref_phys: Reference temperature, degrees C.
        u_pred_phys: Predicted excess pore pressure, Pa.
        u_ref_phys: Reference excess pore pressure, Pa.

    Returns:
        Dict with "T_mse", "T_rel_l2", "T_nrmse", "u_mse", "u_rel_l2", "u_nrmse".
    """
    T_metrics = error_metrics(
        to_nondimensional_T(T_pred_phys), to_nondimensional_T(T_ref_phys)
    )
    u_metrics = error_metrics(
        to_nondimensional_u(u_pred_phys), to_nondimensional_u(u_ref_phys)
    )
    result = {f"T_{k}": v for k, v in T_metrics.items()}
    result.update({f"u_{k}": v for k, v in u_metrics.items()})
    return result