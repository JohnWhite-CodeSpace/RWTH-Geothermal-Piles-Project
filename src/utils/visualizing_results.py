"""
Radar-chart comparison of grid search configurations.

Reads `data/processed/case{N}_grid_search_results.csv` (produced by
`scripts/grid_search.py`) and plots a radar chart comparing every
hyperparameter combination across all four error metrics at once. The
actual plotting logic lives in `src.utils.plotting.plot_radar_chart`
(shared with the rest of the project's charts); this module is just
the entry point for this one CSV -> chart use case.

Usage:
    python -m src.utils.visualizing_results [case_num]
"""

import sys
from pathlib import Path

import pandas as pd

from src.utils.plotting import plot_radar_chart

BASE_DIR = Path(__file__).resolve().parent.parent.parent
METRICS = ["T_mse", "T_rel_l2", "u_mse", "u_rel_l2"]


def plot_grid_search_radar(case_num: int) -> Path:
    """Load a case's grid search CSV and plot its radar chart comparison."""
    csv_path = (
        BASE_DIR / "data" / "processed" / f"case{case_num}_grid_search_results.csv"
    )
    df = pd.read_csv(csv_path)
    return plot_radar_chart(df, METRICS, case_num)


if __name__ == "__main__":
    case = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    plot_grid_search_radar(case)
