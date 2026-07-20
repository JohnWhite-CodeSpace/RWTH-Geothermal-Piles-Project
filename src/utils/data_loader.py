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


def load_case(case_num: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load FDM reference data for a single case.
    
    Args:
        case_num: Case number (1-5)
    
    Returns:
        u: Excess pore pressure (n_space, n_time)
        T: Temperature distribution (n_space, n_time)
        r: Spatial coordinates (n_space,)
        t: Time coordinates (n_time,)
    """
    case_dir = Path(f"data/raw/case{case_num}")
    
    if not case_dir.exists():
        raise FileNotFoundError(f"Case directory not found: {case_dir}")
    
    # Load CSV files
    pressure_file = case_dir / "excess_pore_pressure.csv"
    temperature_file = case_dir / "temperature_distribution.csv"
    
    if not pressure_file.exists():
        raise FileNotFoundError(f"Pressure file not found: {pressure_file}")
    if not temperature_file.exists():
        raise FileNotFoundError(f"Temperature file not found: {temperature_file}")
    
    # Read data
    u_df = pd.read_csv(pressure_file)
    T_df = pd.read_csv(temperature_file)
    
    # Extract values
    u = u_df.values
    T = T_df.values
    
    # Extract coordinates from column/row names
    # Assumes: columns are time values, rows are spatial values (indexed 0...)
    # Adjust based on your actual CSV structure!
    
    # If first column is 'r' coordinate:
    if 'r' in u_df.columns:
        r = u_df['r'].values
        u = u_df.drop(columns=['r']).values
        T = T_df.drop(columns=['r']).values if 'r' in T_df.columns else T_df.values
    else:
        # Generate spatial coordinates (assuming 0.5m to 30m, equally spaced)
        n_space = u.shape[0]
        r = np.linspace(0.5, 30, n_space)
    
    # If first row is time values:
    if u.shape[1] > 0:
        try:
            t = np.array(u_df.columns[1:], dtype=float)  # Skip first column if it's 'r'
        except:
            # Generate time coordinates
            n_time = u.shape[1]
            t = np.linspace(0, 1e6, n_time)  # Adjust to your time range
    else:
        t = np.array([])
    
    return u, T, r, t


def load_all_cases() -> Dict:
    """
    Load all FDM reference cases (1-5).
    
    Returns:
        Dictionary with keys 'case1'-'case5', each containing:
        {'u': pressure, 'T': temperature, 'r': radius, 't': time}
    """
    all_data = {}
    
    for case_num in range(1, 6):
        try:
            u, T, r, t = load_case(case_num)
            all_data[f'case{case_num}'] = {
                'u': u,
                'T': T,
                'r': r,
                't': t,
                'k': [1e-8, 1e-9, 1e-10, 1e-11, 1e-12][case_num-1]  # Permeability
            }
            print(f"✓ case{case_num} loaded: u.shape={u.shape}, T.shape={T.shape}")
        except Exception as e:
            print(f"✗ case{case_num} failed: {e}")
    
    return all_data


def prepare_training_data(case_num: int, normalize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare training data from FDM reference solution.
    
    Creates dataset of (r, t) → (u, T) pairs for PINN training.
    
    Args:
        case_num: Case number (1-5)
        normalize: Whether to normalize to [0, 1]
    
    Returns:
        X: Input coordinates (n_samples, 2) with columns [r, t]
        y: Target values (n_samples, 2) with columns [u, T]
    """
    u, T, r, t = load_case(case_num)
    
    # Create meshgrid
    r_mesh, t_mesh = np.meshgrid(r, t, indexing='ij')
    
    # Flatten to create training pairs
    X = np.column_stack([r_mesh.flatten(), t_mesh.flatten()])
    y = np.column_stack([u.flatten(), T.flatten()])
    
    # Normalize
    if normalize:
        # Normalize r to [0, 1]
        X[:, 0] = (X[:, 0] - X[:, 0].min()) / (X[:, 0].max() - X[:, 0].min())
        # Normalize t to [0, 1]
        X[:, 1] = (X[:, 1] - X[:, 1].min()) / (X[:, 1].max() - X[:, 1].min())
        
        # Normalize targets
        y[:, 0] = (y[:, 0] - y[:, 0].min()) / (y[:, 0].max() - y[:, 0].min() + 1e-8)
        y[:, 1] = (y[:, 1] - y[:, 1].min()) / (y[:, 1].max() - y[:, 1].min() + 1e-8)
    
    return X, y


if __name__ == "__main__":
    # Example usage
    print("Loading FDM reference data...")
    
    # Load single case
    try:
        u, T, r, t = load_case(case_num=1)
        print(f"\ncase1:")
        print(f"  Pressure shape: {u.shape}")
        print(f"  Temperature shape: {T.shape}")
        print(f"  Spatial points: {len(r)}")
        print(f"  Time steps: {len(t)}")
    except Exception as e:
        print(f"Error loading case1: {e}")
    
    # Load all cases
    print("\n\nLoading all cases...")
    all_cases = load_all_cases()
    print(f"\nTotal cases loaded: {len(all_cases)}")
    
    # Prepare training data
    print("\n\nPreparing training data for case1...")
    try:
        X, y = prepare_training_data(case_num=1, normalize=True)
        print(f"Training data shapes:")
        print(f"  Inputs (r, t): {X.shape}")
        print(f"  Targets (u, T): {y.shape}")
    except Exception as e:
        print(f"Error preparing training data: {e}")
