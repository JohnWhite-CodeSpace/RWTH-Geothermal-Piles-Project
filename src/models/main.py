import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from src.utils.config import PhysicsConstants
from src.utils.Sampler import GeothermalSampler
from pinn import GeothermalPINN


def build_network(input_dim: int = 2, output_dim: int = 2) -> nn.Module:
    """
    Construct the underlying Multi-Layer Perceptron (MLP) architecture.

    Args:
        input_dim: Number of input features (r*, t*). Defaults to 2.
        output_dim: Number of output features (T*, u*). Defaults to 2.

    Returns:
        A PyTorch Sequential model mapping input coordinates to outputs.
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


def run_geothermal_pipeline() -> None:
    """
    Execute the complete sampling, training, and evaluation pipeline.

    This function coordinates the execution flow:
    1. Generates collocation points via GeothermalSampler.
    2. Builds the neural network and initializes GeothermalPINN.
    3. Trains the PINN on the governing PDEs, ICs, and BCs.
    4. Evaluates the trained predictions against the FDM reference case.
    """
    # Device and execution parameters
    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs = 5000
    case_num = 1

    # Nondimensional physical governing coefficients
    t_c_val = 1e7
    u_c_val = 8e5
    
    physics = PhysicsConstants()
    C1, C2, C3 = physics.calculate_physics_constants(
        k=1.00e-8,
        Ks=2.0e6,
        t_c=t_c_val,
        u_c=u_c_val
    )

    # Define domain spatial and temporal boundaries [(r_min, r_max), (t_min, t_max)]
    domain_spans = [(0.0167, 1.0), (0.0, 1.0)]

    # 1. Generate collocation points
    print(">>> Generating collocation points...")
    sampler = GeothermalSampler(spans=domain_spans)
    dom_pts, ic_pts, bc_pile_pts, bc_far_pts = sampler.sample(
        num_dom=5000,
        num_ic=500,
        num_bc=500,
        methods=("lhs", "grid", "grid"),
        device=device,
    )

    # 2. Construct network and instantiate PINN model
    print(">>> Initializing GeothermalPINN model...")
    net_u = build_network(input_dim=2, output_dim=2)
    model = GeothermalPINN(net_u=net_u, device=device)

    # 3. Train the neural network
    print(f">>> Training model for {epochs} epochs on {device.upper()}...")
    model.train_net(
        domain_points=dom_pts,
        ic_points=ic_pts,
        bc_pile_points=bc_pile_pts,
        bc_far_points=bc_far_pts,
        C1=C1,
        C2=C2,
        C3=C3,
        epochs=epochs,
        optimizer=optim.Adam,
        log_interval=200,
    )

    # 4. Evaluate and plot results against FDM reference data
    print(f">>> Evaluating against FDM Case {case_num}...")
    model.evaluate(case_num=case_num, t_c=t_c_val, u_c=u_c_val)


if __name__ == "__main__":
    run_geothermal_pipeline()