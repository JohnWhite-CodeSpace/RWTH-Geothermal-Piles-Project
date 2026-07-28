import sys
import os
import numpy as np
from typing import Tuple

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Ensure double precision globally to prevent underflow in small-scale terms (C3)
torch.set_default_dtype(torch.float64)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.config import PhysicsConstants
from src.utils.sampler import GeothermalSampler
from src.utils.data_loader import load_single_case
from models.pinn2 import GeothermalPINN, GeothermalInversePINN


def build_network(input_dim: int = 2, output_dim: int = 1) -> nn.Module:
    """
    Construct the underlying Multi-Layer Perceptron (MLP) architecture.
    """
    return nn.Sequential(
        nn.Linear(input_dim, 40),
        nn.Tanh(),
        nn.Linear(40, 40),
        nn.Tanh(),
        nn.Linear(40, 40),
        nn.Tanh(),
        nn.Linear(40, 40),
        nn.Tanh(),
        nn.Linear(40, output_dim),
    )


def run_inverse_curriculum_pipeline() -> None:
    """
    Execute the Inverse PINN Curriculum Learning pipeline across all 5 cases (1 to 5).
    Uses proper warm-starting for log_k with float64 precision and fresh network weight initialization per case.
    """
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    
    # 1. Physics & Domain Setup
    t_c_val = 1e7
    u_c_val_reference = 8205.9 
    physics = PhysicsConstants()

    domain_spans = [(0.0167, 1.0), (0.0, 1.0)]
    print(">>> Generating collocation points for Inverse Curriculum...")
    sampler = GeothermalSampler(spans=domain_spans)
    dom_pts, ic_pts, bc_pile_points, bc_far_points = sampler.sample(
        num_dom=5000, num_ic=500, num_bc=500, methods=("lhs", "grid", "grid"), device=device
    )
    dom_pts = dom_pts.to(torch.float64)
    ic_pts = ic_pts.to(torch.float64)
    bc_pile_points = bc_pile_points.to(torch.float64)
    bc_far_points = bc_far_points.to(torch.float64)

    # 2. Pre-train Temperature Network robustly USING TEMPERATURE DATA
    print("\n>>> Loading Temperature Sensor Data for pre-training...")
    # Temperature is pure conduction (independent of k), so Case 1 data is sufficient for all cases.
    temp_df, _ = load_single_case(1)
    
    r_true_T = temp_df.columns.astype(float).to_numpy()
    t_days_T = temp_df.index.astype(float).to_numpy()
    T_true_raw = temp_df.to_numpy(dtype=float)
    
    # Normalize Temperature to [0, 1] to match the PDE boundary conditions
    T_min, T_max = np.min(T_true_raw), np.max(T_true_raw)
    T_star_true = (T_true_raw - T_min) / (T_max - T_min) if T_max > T_min else T_true_raw
    
    r_mesh_T, t_mesh_T = np.meshgrid(r_true_T, t_days_T * 86400.0, indexing="xy")
    r_star_T = (r_mesh_T.flatten() / 30.0).reshape(-1, 1)
    t_star_T = (t_mesh_T.flatten() / t_c_val).reshape(-1, 1)
    T_star_true_flat = T_star_true.flatten().reshape(-1, 1)

    np.random.seed(42)
    idx_T = np.random.choice(len(r_star_T), 1500, replace=False) # Pick 1500 sensor points for Temp
    sensor_pts_T = torch.tensor(np.hstack((r_star_T[idx_T], t_star_T[idx_T])), dtype=torch.float64, device=device)
    sensor_T = torch.tensor(T_star_true_flat[idx_T], dtype=torch.float64, device=device)

    print(">>> Pre-training Temperature Network...")
    net_T = build_network(input_dim=2, output_dim=1).to(device)
    net_u_temp = build_network(input_dim=2, output_dim=1).to(device)
    forward_model = GeothermalPINN(net_T=net_T, net_u=net_u_temp, device=device)
    
    dummy_C1, _, _, _ = physics.calculate_physics_constants(k=1e-10, Ks=2.0e6, t_c=t_c_val)
    
    # Train with 15000 epochs, adaptive sampling, and sensor data!
    forward_model.train_net(
        domain_points=dom_pts, ic_points=ic_pts, bc_pile_points=bc_pile_points, bc_far_points=bc_far_points,
        C1=dummy_C1, C2=1.0, C3=1.0, epochs=6000, train_T=True, train_u=False,
        adaptive_method='rar', adaptive_interval=1500, num_adaptive_points=100,
        sensor_points_T=sensor_pts_T, sensor_T_true=sensor_T
    )

    # 3. Define the Permeability Curriculum Table (Cases 1 to 5)
    cases_permeability = [
        (1, 1.00e-8),
        (2, 1.00e-9),
        (3, 1.00e-10),
        (4, 1.00e-11),
        (5, 1.00e-12)
    ]

    current_log_k_guess = -8.0 
    all_histories = {}

    for case_num, true_k in cases_permeability:
        print("\n" + "="*60)
        print(f" INVERSE CURRICULUM: Processing Case {case_num} (True k = {true_k:.1e})")
        print("="*60)

        temp_df, pressure_df = load_single_case(case_num)

        r_true = pressure_df.columns.astype(float).to_numpy()
        t_days = pressure_df.index.astype(float).to_numpy()
        u_true = pressure_df.to_numpy(dtype=float)

        r_mesh, t_mesh = np.meshgrid(r_true, t_days * 86400.0, indexing="xy")
        r_star = (r_mesh.flatten() / 30.0).reshape(-1, 1)
        t_star = (t_mesh.flatten() / t_c_val).reshape(-1, 1)
        u_star_true = (u_true.flatten() / u_c_val_reference).reshape(-1, 1)

        np.random.seed(42)
        idx = np.random.choice(len(r_star), 1000, replace=False)
        sensor_pts_u = torch.tensor(np.hstack((r_star[idx], t_star[idx])), dtype=torch.float64, device=device)
        sensor_u = torch.tensor(u_star_true[idx], dtype=torch.float64, device=device)

        # Re-initialize net_u for each case to prevent error accumulation from previous steps
        net_u_case = build_network(input_dim=2, output_dim=1).to(device)

        inverse_model = GeothermalInversePINN(
            net_T=net_T, 
            net_u=net_u_case, 
            initial_log_k=current_log_k_guess, 
            device=device
        )

        current_burn_in = 2000 if case_num == 1 else 1000
        current_epochs = 6000  
        history_epochs, history_loss, history_k = inverse_model.train_inverse(
            domain_points=dom_pts, ic_points=ic_pts, bc_pile_points=bc_pile_points, bc_far_points=bc_far_points,
            sensor_points=sensor_pts_u, sensor_u_true=sensor_u, physics_config=physics,
            t_c_val=t_c_val, 
            epochs=current_epochs, 
            log_interval=500,
            burn_in_epochs=current_burn_in, 
            lr_k=1e-4
        )
        
        # Plot individual case convergence
        plt.figure(figsize=(8, 5))
        plt.plot(history_epochs, history_k, label='Estimated k (PINN)', color='green', linewidth=2)
        plt.axhline(y=true_k, color='red', linestyle='--', linewidth=2, label=f'True k (Case {case_num}: {true_k:.1e})')
        plt.yscale('log')
        plt.xlabel('Epochs', fontsize=12)
        plt.ylabel('Permeability k (m/s)', fontsize=12)
        plt.title(f'Case {case_num}: Permeability Convergence (Estimated vs True)', fontsize=14)
        plt.grid(True, which="both", ls="--")
        plt.legend(fontsize=11)
        plt.tight_layout()
        
        plot_filename = f"inverse_case_{case_num}_k_convergence.png"
        plt.savefig(plot_filename, dpi=150)
        plt.close()
        print(f">>> Saved case {case_num} convergence plot: {plot_filename}")

        all_histories[case_num] = (history_epochs, history_k, true_k)

        # Warm-Start update: Keep the optimized log_k for the next case
        current_log_k_guess = inverse_model.log_k.item()
        print(f">>> Case {case_num} completed. Estimated k: {10**current_log_k_guess:.3e} m/s (Passed to next step)")

    # Plot final summary comparing all curriculum cases
    plt.figure(figsize=(10, 6))
    for c_num, (h_ep, h_k, t_val) in all_histories.items():
        plt.plot(h_ep, h_k, label=f'Case {c_num} (True: {t_val:.1e})', linewidth=2)
    
    plt.yscale('log')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Estimated Permeability k (m/s)', fontsize=12)
    plt.title('Curriculum Learning Summary: Estimated k vs Epochs Across All Cases', fontsize=14)
    plt.grid(True, which="both", ls="--")
    plt.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    
    summary_filename = "inverse_curriculum_all_cases_summary.png"
    plt.savefig(summary_filename, dpi=150)
    plt.show()
    print(f">>> Saved overall summary plot: {summary_filename}")


if __name__ == "__main__":
    run_inverse_curriculum_pipeline()