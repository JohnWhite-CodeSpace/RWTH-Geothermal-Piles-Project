"""
Data loader for FDM reference solutions (case1-case5).

Loads temperature and excess pore-water pressure fields produced by the
Finite Difference Method (FDM) solver and prepares them as nondimensional
(r*, t*) -> (T*, u*) pairs, using the same nondimensionalization as the
PINN model: r* = r/R_s, t* = t/t_c, T* = (T - T_s)/delta_T, u* = u/u_c.

Usage:
    from src.utils.data_loader import load_single_case, prepare_training_data

    temp_df, pressure_df = load_single_case(case_num=1)
    X, y = prepare_training_data(case_num=1)
"""

import traceback
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from src.utils.physics_constants import PhysicsConstants

SECONDS_PER_DAY = 24.0 * 60.0 * 60.0

PHYS_CONST = PhysicsConstants()


def load_single_case(
    case_num: int,
) -> tuple[pd.DataFrame, pd.DataFrame] | dict[str, str | int]:
    """
    Load temperature and excess pore pressure data for a selected case.

    Parameters
    ----------
    case_num : int
        Case number (1-5).

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Temperature and pressure DataFrames.

    dict
        Error information if loading fails.
    """
    try:
        if not isinstance(case_num, int):
            raise TypeError("case_num must be an integer")
        if not 1 <= case_num <= 5:
            raise ValueError("case_num must be between 1 and 5")

        root = Path(__file__).resolve().parent.parent.parent
        temp_path = (
            root / "data" / "raw" / f"case{case_num}" / "temperature_distribution.csv"
        )
        pressure_path = (
            root / "data" / "raw" / f"case{case_num}" / "excess_pore_pressure.csv"
        )

        temp_df = pd.read_csv(temp_path, index_col=0)
        pressure_df = pd.read_csv(pressure_path, index_col=0)
        return temp_df, pressure_df

    except Exception as ex:
        return {
            "return_code": -1,
            "error_message": repr(ex),
            "traceback": traceback.format_exc(),
        }


def load_all_cases() -> dict[int, dict[str, pd.DataFrame]] | dict[str, str | int]:
    """
    Load temperature and excess pore pressure data for all available cases.

    Returns
    -------
    dict
        Dictionary containing data for cases 1-5.

    dict
        Error information if loading fails.
    """
    try:
        all_cases = {}
        for i in range(1, 6):
            result = load_single_case(i)
            if isinstance(result, dict):
                return result
            temp_df, pressure_df = result
            all_cases[i] = {"Temp_data": temp_df, "Pressure_data": pressure_df}
        return all_cases
    except Exception as ex:
        return {
            "return_code": -1,
            "error_message": repr(ex),
            "traceback": traceback.format_exc(),
        }


def prepare_training_data(
    case_num: int,
) -> Tuple[np.ndarray, np.ndarray] | dict[str, str | int]:
    """
    Build a nondimensional (r*, t*) -> (T*, u*) dataset from FDM data.

    Applies the same nondimensionalization used by the PINN model
    (r* = r/R_s, t* = t/t_c, T* = (T-T_s)/delta_T, u* = u/u_c), using
    the module-level `PHYS_CONST` instance, so the returned arrays live
    in the same coordinate system as the network's inputs/outputs.
    Whatever computes C1/C2/C3 for a case must use this same instance's
    `t_c`/`u_c` (see `PhysicsConstants.calculate_physics_constants`) to
    stay consistent.

    Parameters
    ----------
    case_num : int
        Case number (1-5).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        X: (N, 2) array with columns [r*, t*].
        y: (N, 2) array with columns [T*, u*], matching the output
        order of `GeothermalPINN.net_u_forward` (T then u).

    dict
        Error information if loading fails.
    """
    result = load_single_case(case_num)
    if isinstance(result, dict):
        return result

    try:
        temp_df, pressure_df = result

        r = temp_df.columns.astype(float).to_numpy()
        t_days = temp_df.index.astype(float).to_numpy()
        T = temp_df.to_numpy(dtype=float)
        u = pressure_df.to_numpy(dtype=float)

        r_mesh, t_mesh = np.meshgrid(r, t_days, indexing="xy")

        r_star = r_mesh / PHYS_CONST.R_s
        t_star = (t_mesh * SECONDS_PER_DAY) / PHYS_CONST.t_c
        T_star = (T - PHYS_CONST.T_s) / PHYS_CONST.delta_T
        u_star = u / PHYS_CONST.u_c

        X = np.column_stack([r_star.ravel(), t_star.ravel()])
        y = np.column_stack([T_star.ravel(), u_star.ravel()])

        return X, y

    except Exception as ex:
        return {
            "return_code": -1,
            "error_message": repr(ex),
            "traceback": traceback.format_exc(),
        }


if __name__ == "__main__":
    print("Loading FDM reference data...")

    single_case_result = load_single_case(case_num=1)
    if isinstance(single_case_result, dict):
        print("Error loading case1:")
        print(single_case_result["error_message"])
    else:
        temp_df, pressure_df = single_case_result
        print("\nCase 1:")
        print(f"  Temperature shape: {temp_df.shape}")
        print(f"  Pressure shape: {pressure_df.shape}")
        print(f"  Time steps: {len(temp_df.index)}")
        print(f"  Spatial points: {len(temp_df.columns)}")

    print("\nLoading all cases...")
    all_cases = load_all_cases()
    if isinstance(all_cases, dict) and "return_code" in all_cases:
        print("Error loading all cases:")
        print(all_cases["error_message"])
    else:
        print(f"Total cases loaded: {len(all_cases)}")

    print("\nPreparing training data for case1...")
    training_data_result = prepare_training_data(case_num=1)
    if isinstance(training_data_result, dict):
        print("Error preparing training data:")
        print(training_data_result["error_message"])
    else:
        X, y = training_data_result
        print("Training data:")
        print(f"  Inputs (r*, t*): {X.shape}")
        print(f"  Targets (T*, u*): {y.shape}")
        print("\nFirst five samples:")
        print("X:")
        print(X[:5000])
        print("\ny:")
        print(y[:5000])

    print("\nTemperature")
    print(f"min: {y[:, 0].min()}")
    print(f"max: {y[:, 0].max()}")

    print("\nPressure")
    print(f"min: {y[:, 1].min()}")
    print(f"max: {y[:, 1].max()}")
