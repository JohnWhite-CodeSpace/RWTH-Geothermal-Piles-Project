"""
Data loader for FDM reference solutions (case1-case5).

Usage:
    from example_data_loader import load_case, load_all_cases
    
    # Load single case
    u, T, r, t = load_case(case_num=1)
    
    # Load all cases
    all_data = load_all_cases()
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict
import traceback

def load_single_case(case_num: int)-> tuple[pd.DataFrame, pd.DataFrame] | dict[str, str | int]:
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

        ROOT = Path(__file__).resolve().parent.parent.parent
        temp_path = ROOT / "data" / "raw" / f"case{case_num}" / "temperature_distribution.csv"
        pressure_path = ROOT / "data" / "raw" / f"case{case_num}" / "excess_pore_pressure.csv"

        if not isinstance(case_num, int):
            raise TypeError("case_num must be an integer")
        if not 1 <= case_num <= 5:
            raise ValueError("case_num must be between 1 and 5")

        temp_df = pd.read_csv(temp_path, index_col=0)
        pressure_df = pd.read_csv(pressure_path, index_col=0)
        return temp_df, pressure_df

    except Exception as ex:
        return {"return_code": -1, "error_message": repr(ex), "taceback": traceback.format_exc()}

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
        return {"return_code": -1, "error_message": repr(ex), "traceback": traceback.format_exc()}

def prepare_training_data(case_num: int,normalize: bool = True) -> Tuple[np.ndarray, np.ndarray] | dict[str, str | int]:
    """
    Creates training dataset:
        (r, t) -> (pressure, temperature)

    Returns
    -------
    X : ndarray (N,2)
        columns = [radius, time]

    y : ndarray (N,2)
        columns = [pressure, temperature]
    """

    result = load_single_case(case_num)
    try:
        if isinstance(result, dict):
            return result

        temp_df, pressure_df = result
        r = temp_df.columns.astype(float).to_numpy()
        t = temp_df.index.astype(float).to_numpy()

        # ---------- values ----------
        T = temp_df.to_numpy(dtype=float)
        u = pressure_df.to_numpy(dtype=float)

        # ---------- mesh ----------
        r_mesh, t_mesh = np.meshgrid(r, t, indexing="xy")
        X = np.column_stack([r_mesh.ravel(),t_mesh.ravel()])

        y = np.column_stack([u.ravel(),T.ravel()])

        if normalize:
            # radius
            X[:, 0] = (X[:, 0] - X[:, 0].min()) / (X[:, 0].max() - X[:, 0].min())

            # time
            X[:, 1] = (X[:, 1] - X[:, 1].min()) / (X[:, 1].max() - X[:, 1].min())

            # pressure
            y[:, 0] = (y[:, 0] - y[:, 0].min()) / (y[:, 0].max() - y[:, 0].min() + 1e-8)

            # temperature
            y[:, 1] = (y[:, 1] - y[:, 1].min()) / (y[:, 1].max() - y[:, 1].min() + 1e-8)

        return X, y
    except Exception as ex:
        return {"return_code": -1, "error_message": repr(ex), "traceback": traceback.format_exc()}

if __name__ == "__main__":
    print("Loading FDM reference data...")

    # Load single case
    result = load_single_case(case_num=1)

    if isinstance(result, dict):
        print("Error loading case1:")
        print(result["error_message"])
    else:
        temp_df, pressure_df = result

        print("\nCase 1:")
        print(f"  Temperature shape: {temp_df.shape}")
        print(f"  Pressure shape: {pressure_df.shape}")
        print(f"  Time steps: {len(temp_df.index)}")
        print(f"  Spatial points: {len(temp_df.columns)}")

    # Load all cases
    print("\nLoading all cases...")

    all_cases = load_all_cases()

    if isinstance(all_cases, dict) and "return_code" in all_cases:
        print("Error loading all cases:")
        print(all_cases["error_message"])
    else:
        print(f"Total cases loaded: {len(all_cases)}")

    # Prepare training data
    print("\nPreparing training data for case1...")

    result = prepare_training_data(case_num=1, normalize=True)

    if isinstance(result, dict):
        print("Error preparing training data:")
        print(result["error_message"])
    else:
        X, y = result

        print("Training data:")
        print(f"  Inputs (radius, time): {X.shape}")
        print(f"  Targets (pressure, temperature): {y.shape}")

        print("\nFirst fifty samples:")
        print("X:")
        print(X[:50])
        print("\ny:")
        print(y[:50])
