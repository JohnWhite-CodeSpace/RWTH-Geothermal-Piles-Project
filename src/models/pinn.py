from typing import Optional, Tuple, Type

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class GeothermalPINN(nn.Module):
    """Physics-informed network for coupled temperature/pore-pressure fields.

    The underlying network predicts the dimensionless temperature T* and
    excess pore-water pressure u* as functions of dimensionless radius r*
    and time t*.
    """

    def __init__(
        self,
        net_T: nn.Module,
        net_u: nn.Module,
        loss_fn: Optional[nn.Module] = None,
        device: str = "cpu",
    ):
        """
        Initialize the PINN with decoupled networks for temperature and pressure.

        Args:
            net_T: Underlying network mapping (r, t) -> T.
            net_u: Underlying network mapping (r, t) -> u.
            loss_fn: Loss function used for the PDE/IC/BC residuals.
                Defaults to `nn.MSELoss()`.
            device: Device to run the model on.
        """
        super().__init__()

        self.device = torch.device(device)
        self.net_T = net_T.to(self.device)
        self.net_u = net_u.to(self.device)
        self.adaptive_points = None  # Reserved for future adaptive sampling

        self.loss_fn = loss_fn if loss_fn is not None else nn.MSELoss()

    def net_forward(
        self, r: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the decoupled underlying networks.

        Args:
            r: Dimensionless radius, shape (N, 1).
            t: Dimensionless time, shape (N, 1).

        Returns:
            Tuple (T, u) of dimensionless temperature and pore pressure.
        """
        inputs = torch.cat((r, t), dim=1)
        T = self.net_T(inputs)
        u = self.net_u(inputs)
        return T, u
    def train_net(
        self,
        domain_points: torch.Tensor,
        ic_points: torch.Tensor,
        bc_pile_points: torch.Tensor,
        bc_far_points: torch.Tensor,
        C1: float,
        C2: float,
        C3: float,
        epochs: int,
        optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam,
        log_interval: int = 500,
    ) -> None:
        """
        Train the networks sequentially: Stage 1 (Temperature) -> Freeze -> Stage 2 (Pressure).
        Each stage uses Adam for initial convergence, followed by L-BFGS for strict physics enforcement.

        Args:
            domain_points: Interior collocation points, columns [r, t].
            ic_points: Initial condition points, columns [r, t].
            bc_pile_points: Pile boundary points, columns [r, t].
            bc_far_points: Far-field boundary points, columns [r, t].
            C1: Nondimensional heat equation coefficient.
            C2: Nondimensional thermal-coupling coefficient.
            C3: Nondimensional pore-pressure diffusion coefficient.
            epochs: Number of training iterations for Adam.
            optimizer: (Legacy argument, safely ignored here as optimizers are hardcoded for 2-stage).
            log_interval: Number of iterations between log prints.
        """
        def prepare_pts(pts):
            r = pts[:, 0:1].to(self.device).requires_grad_(True)
            t = pts[:, 1:2].to(self.device).requires_grad_(True)
            return r, t

        r_dom, t_dom = prepare_pts(domain_points)
        r_ic, t_ic = prepare_pts(ic_points)
        r_bc_p, t_bc_p = prepare_pts(bc_pile_points)
        r_bc_f, t_bc_f = prepare_pts(bc_far_points)

        self.C1 = C1
        self.C2 = C2
        self.C3 = C3

        # =======================================================
        # STAGE 1A: Train Temperature Network (Adam)
        # =======================================================
        print("\n" + "="*50)
        print(" STAGE 1A: Training Temperature Network (Adam)")
        print("="*50)
        
        self.net_T.train()
        opt_T_adam = torch.optim.Adam(self.net_T.parameters(), lr=1e-3)
        
        for ep in range(epochs):
            opt_T_adam.zero_grad()
            
            T_dom = self.net_T(torch.cat((r_dom, t_dom), dim=1))
            T_r = torch.autograd.grad(T_dom, r_dom, torch.ones_like(T_dom), create_graph=True)[0]
            T_rr = torch.autograd.grad(T_r, r_dom, torch.ones_like(T_r), create_graph=True)[0]
            T_t = torch.autograd.grad(T_dom, t_dom, torch.ones_like(T_dom), create_graph=True)[0]
            
            res_T = T_t - C1 * ((1.0 / r_dom) * T_r + T_rr)
            loss_pde_T = self.loss_fn(res_T, torch.zeros_like(res_T))
            
            T_ic = self.net_T(torch.cat((r_ic, t_ic), dim=1))
            loss_ic_T = self.loss_fn(T_ic, torch.zeros_like(T_ic))
            
            T_bc_p = self.net_T(torch.cat((r_bc_p, t_bc_p), dim=1))
            loss_bc_p_T = self.loss_fn(T_bc_p, torch.ones_like(T_bc_p))
            
            T_bc_f = self.net_T(torch.cat((r_bc_f, t_bc_f), dim=1))
            loss_bc_f_T = self.loss_fn(T_bc_f, torch.zeros_like(T_bc_f))
            
            total_loss_T = loss_pde_T + 100.0 * (loss_ic_T + loss_bc_p_T + loss_bc_f_T)
            total_loss_T.backward()
            opt_T_adam.step()
            
            if ep % log_interval == 0:
                print(f"[Stage 1A - Adam] Iter {ep:4d} | Total Loss T: {total_loss_T.item():.3e}")

        # =======================================================
        # STAGE 1B: Fine-tune Temperature Network (L-BFGS)
        # =======================================================
        print("\n>>> STAGE 1B: Fine-tuning Temperature with L-BFGS...")
        opt_T_lbfgs = torch.optim.LBFGS(
            self.net_T.parameters(), 
            lr=0.5, 
            max_iter=5000, 
            history_size=100,
            tolerance_grad=1e-9,      
            tolerance_change=1e-11,   
            line_search_fn="strong_wolfe"
        )
        
        def closure_T():
            opt_T_lbfgs.zero_grad()
            T_dom = self.net_T(torch.cat((r_dom, t_dom), dim=1))
            T_r = torch.autograd.grad(T_dom, r_dom, torch.ones_like(T_dom), create_graph=True)[0]
            T_rr = torch.autograd.grad(T_r, r_dom, torch.ones_like(T_r), create_graph=True)[0]
            T_t = torch.autograd.grad(T_dom, t_dom, torch.ones_like(T_dom), create_graph=True)[0]
            res_T = T_t - C1 * ((1.0 / r_dom) * T_r + T_rr)
            loss_pde_T = self.loss_fn(res_T, torch.zeros_like(res_T))
            
            T_ic = self.net_T(torch.cat((r_ic, t_ic), dim=1))
            loss_ic_T = self.loss_fn(T_ic, torch.zeros_like(T_ic))
            
            T_bc_p = self.net_T(torch.cat((r_bc_p, t_bc_p), dim=1))
            loss_bc_p_T = self.loss_fn(T_bc_p, torch.ones_like(T_bc_p))
            
            T_bc_f = self.net_T(torch.cat((r_bc_f, t_bc_f), dim=1))
            loss_bc_f_T = self.loss_fn(T_bc_f, torch.zeros_like(T_bc_f))
            
            total_loss_T = loss_pde_T + 10.0 * (loss_ic_T + loss_bc_p_T + loss_bc_f_T)
            total_loss_T.backward()
            return total_loss_T
            
        opt_T_lbfgs.step(closure_T)
        print(f"[Stage 1B - LBFGS] Final Loss T: {closure_T().item():.3e}")

        # --- Freeze Temperature Network ---
        print("\n>>> Freezing Temperature Network...")
        for param in self.net_T.parameters():
            param.requires_grad = False
        self.net_T.eval()

        # =======================================================
        # STAGE 2A: Train Pore Pressure Network (Adam)
        # =======================================================
        print("\n" + "="*50)
        print(" STAGE 2A: Training Pore Pressure Network (Adam)")
        print("="*50)
        
        self.net_u.train()
        opt_u_adam = torch.optim.Adam(self.net_u.parameters(), lr=1e-3)

        for ep in range(epochs):
            opt_u_adam.zero_grad()
            
            # Extract fixed temperature gradient (source term)
            T_dom_fixed = self.net_T(torch.cat((r_dom, t_dom), dim=1))
            T_t_fixed = torch.autograd.grad(T_dom_fixed, t_dom, torch.ones_like(T_dom_fixed), create_graph=True)[0].detach()
            
            u_dom = self.net_u(torch.cat((r_dom, t_dom), dim=1))
            u_r = torch.autograd.grad(u_dom, r_dom, torch.ones_like(u_dom), create_graph=True)[0]
            u_rr = torch.autograd.grad(u_r, r_dom, torch.ones_like(u_r), create_graph=True)[0]
            u_t = torch.autograd.grad(u_dom, t_dom, torch.ones_like(u_dom), create_graph=True)[0]
            
            res_u = u_t - C2 * T_t_fixed - C3 * ((1.0 / r_dom) * u_r + u_rr)
            loss_pde_u = self.loss_fn(res_u, torch.zeros_like(res_u))
            
            u_ic = self.net_u(torch.cat((r_ic, t_ic), dim=1))
            loss_ic_u = self.loss_fn(u_ic, torch.zeros_like(u_ic))
            
            u_bc_p = self.net_u(torch.cat((r_bc_p, t_bc_p), dim=1))
            u_r_bc_p = torch.autograd.grad(u_bc_p, r_bc_p, torch.ones_like(u_bc_p), create_graph=True)[0]
            loss_bc_p_u = self.loss_fn(u_r_bc_p, torch.zeros_like(u_r_bc_p))
            
            u_bc_f = self.net_u(torch.cat((r_bc_f, t_bc_f), dim=1))
            loss_bc_f_u = self.loss_fn(u_bc_f, torch.zeros_like(u_bc_f))
            
            total_loss_u = loss_pde_u + 10.0 * (loss_ic_u + loss_bc_p_u + loss_bc_f_u)
            total_loss_u.backward()
            opt_u_adam.step()
            
            if ep % log_interval == 0:
                print(f"[Stage 2A - Adam] Iter {ep:4d} | Total Loss u: {total_loss_u.item():.3e}")

        # =======================================================
        # STAGE 2B: Fine-tune Pore Pressure Network (L-BFGS)
        # =======================================================
        print("\n>>> STAGE 2B: Fine-tuning Pore Pressure with L-BFGS (The crucial step!)...")
        opt_u_lbfgs = torch.optim.LBFGS(
            self.net_u.parameters(), 
            lr=0.5, 
            max_iter=5000, 
            history_size=100,
            tolerance_grad=1e-9,      
            tolerance_change=1e-11,
            line_search_fn="strong_wolfe"
        )
        
        def closure_u():
            opt_u_lbfgs.zero_grad()
            T_dom_fixed = self.net_T(torch.cat((r_dom, t_dom), dim=1))
            T_t_fixed = torch.autograd.grad(T_dom_fixed, t_dom, torch.ones_like(T_dom_fixed), create_graph=True)[0].detach()
            
            u_dom = self.net_u(torch.cat((r_dom, t_dom), dim=1))
            u_r = torch.autograd.grad(u_dom, r_dom, torch.ones_like(u_dom), create_graph=True)[0]
            u_rr = torch.autograd.grad(u_r, r_dom, torch.ones_like(u_r), create_graph=True)[0]
            u_t = torch.autograd.grad(u_dom, t_dom, torch.ones_like(u_dom), create_graph=True)[0]
            
            res_u = u_t - C2 * T_t_fixed - C3 * ((1.0 / r_dom) * u_r + u_rr)
            loss_pde_u = self.loss_fn(res_u, torch.zeros_like(res_u))
            
            u_ic = self.net_u(torch.cat((r_ic, t_ic), dim=1))
            loss_ic_u = self.loss_fn(u_ic, torch.zeros_like(u_ic))
            
            u_bc_p = self.net_u(torch.cat((r_bc_p, t_bc_p), dim=1))
            u_r_bc_p = torch.autograd.grad(u_bc_p, r_bc_p, torch.ones_like(u_bc_p), create_graph=True)[0]
            loss_bc_p_u = self.loss_fn(u_r_bc_p, torch.zeros_like(u_r_bc_p))
            
            u_bc_f = self.net_u(torch.cat((r_bc_f, t_bc_f), dim=1))
            loss_bc_f_u = self.loss_fn(u_bc_f, torch.zeros_like(u_bc_f))
            
            total_loss_u = loss_pde_u + 10.0 * (loss_ic_u + loss_bc_p_u + loss_bc_f_u)
            total_loss_u.backward()
            return total_loss_u
            
        opt_u_lbfgs.step(closure_u)
        print(f"[Stage 2B - LBFGS] Final Loss u: {closure_u().item():.3e}")
        
    def predict(self, r: torch.Tensor, t: torch.Tensor) -> Tuple:
        """
        Predict temperature and pore pressure at the given points.

        Args:
            r: Dimensionless radius.
            t: Dimensionless time.

        Returns:
            Tuple (T, u) of NumPy arrays with predicted values.
        """
        self.net_T.eval()
        self.net_u.eval()
        with torch.no_grad():
            r = r.to(self.device)
            t = t.to(self.device)
            T_pred, u_pred = self.net_forward(r, t)
            return T_pred.cpu().numpy(), u_pred.cpu().numpy()

    def evaluate(
        self,
        case_num: int,
        t_c: float = 1e7,
        u_c: float = 8e5,
        r_max: float = 30.0,
        T_initial: float = 12.0,
        delta_T: float = 38.0,
    ) -> None:
        """
        Evaluate predictions against FDM reference data and plot the results.

        Args:
            case_num: The FDM case number (1-5) to load from raw data.
            t_c: Characteristic time for dimensionalization.
            u_c: Characteristic pressure for dimensionalization.
            r_max: Maximum radius used for dimensionalization (e.g., 30m).
            T_initial: Initial temperature of the soil in degrees Celsius.
            delta_T: Temperature difference applied at the pile (e.g., 50 - 12 = 38).
        """

        try:
            from src.utils.data_loader import load_single_case
        except ImportError:
            print("Error: 'data_loader.py' module not found. Please check the import path.")
            return

        # Load the true FDM data using the data loader
        print(f"Loading reference FDM data for Case {case_num}...")
        result = load_single_case(case_num)

        if isinstance(result, dict):
            print(f"Failed to load data: {result['error_message']}")
            return

        temp_df, pressure_df = result

        # Extract spatial and temporal coordinates from the DataFrames
        r_true = temp_df.columns.astype(float).to_numpy()
        t_days = temp_df.index.astype(float).to_numpy()
        T_true = temp_df.to_numpy(dtype=float)
        u_true = pressure_df.to_numpy(dtype=float)

        # Convert time from days to seconds to match the PINN formulation
        seconds_per_day = 24.0 * 60.0 * 60.0
        t_sec = t_days * seconds_per_day

        # Create a 2D meshgrid for the evaluation points
        r_mesh, t_mesh = np.meshgrid(r_true, t_sec, indexing="xy")

        # Nondimensionalize inputs for the network
        r_star = r_mesh.flatten() / r_max
        t_star = t_mesh.flatten() / t_c

        # Convert to PyTorch tensors and move to the designated device
        r_tensor = torch.tensor(r_star, dtype=torch.float32).unsqueeze(1).to(self.device)
        t_tensor = torch.tensor(t_star, dtype=torch.float32).unsqueeze(1).to(self.device)

        # Predict dimensionless fields using the trained model
        T_pred_star, u_pred_star = self.predict(r_tensor, t_tensor)

        # Dimensionalize the predictions back to physical units
        T_pred = (T_pred_star.reshape(T_true.shape) * delta_T) + T_initial
        u_pred = u_pred_star.reshape(u_true.shape) * u_c

        # Calculate the L2 relative errors
        error_T = np.linalg.norm(T_true - T_pred) / np.linalg.norm(T_true)
        error_u = np.linalg.norm(u_true - u_pred) / (np.linalg.norm(u_true) + 1e-8)

        print(f"\n--- Evaluation Results for Case {case_num} ---")
        print(f"Temperature L2 Relative Error   : {error_T * 100:.3f}%")
        print(f"Pore Pressure L2 Relative Error : {error_u * 100:.3f}%")

        # Convert time from seconds to days for plotting
        t_days_mesh = t_mesh / seconds_per_day
        
        # Calculate absolute errors
        error_T_map = np.abs(T_true - T_pred)
        error_u_map = np.abs(u_true - u_pred)

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        # Temperature FDM (Reference)
        im0 = axes[0, 0].pcolormesh(r_mesh, t_days_mesh, T_true, shading="auto", cmap="hot")
        axes[0, 0].set_title("Temperature: FDM (Reference)")
        axes[0, 0].set_xlabel("Radius (m)")
        axes[0, 0].set_ylabel("Time (Days)")
        fig.colorbar(im0, ax=axes[0, 0], label="Temperature (°C)")

        # Temperature PINN (Prediction)
        im1 = axes[0, 1].pcolormesh(r_mesh, t_days_mesh, T_pred, shading="auto", cmap="hot")
        axes[0, 1].set_title("Temperature: PINN (Prediction)")
        axes[0, 1].set_xlabel("Radius (m)")
        axes[0, 1].set_ylabel("Time (Days)")
        fig.colorbar(im1, ax=axes[0, 1], label="Temperature (°C)")

        # Temperature Absolute Error
        im2 = axes[0, 2].pcolormesh(r_mesh, t_days_mesh, error_T_map, shading="auto", cmap="Reds")
        axes[0, 2].set_title(f"Temp Error (Rel L2: {error_T * 100:.2f}%)")
        axes[0, 2].set_xlabel("Radius (m)")
        axes[0, 2].set_ylabel("Time (Days)")
        fig.colorbar(im2, ax=axes[0, 2], label="Error (°C)")

        # Pressure FDM (Reference)
        im3 = axes[1, 0].pcolormesh(r_mesh, t_days_mesh, u_true, shading="auto", cmap="jet")
        axes[1, 0].set_title("Pore Pressure: FDM (Reference)")
        axes[1, 0].set_xlabel("Radius (m)")
        axes[1, 0].set_ylabel("Time (Days)")
        fig.colorbar(im3, ax=axes[1, 0], label="Excess Pore Pressure (Pa)")

        # Pressure PINN (Prediction)
        im4 = axes[1, 1].pcolormesh(r_mesh, t_days_mesh, u_pred, shading="auto", cmap="jet")
        axes[1, 1].set_title("Pore Pressure: PINN (Prediction)")
        axes[1, 1].set_xlabel("Radius (m)")
        axes[1, 1].set_ylabel("Time (Days)")
        fig.colorbar(im4, ax=axes[1, 1], label="Excess Pore Pressure (Pa)")

        # Pressure Absolute Error
        im5 = axes[1, 2].pcolormesh(r_mesh, t_days_mesh, error_u_map, shading="auto", cmap="Reds")
        axes[1, 2].set_title(f"Pressure Error (Rel L2: {error_u * 100:.2f}%)")
        axes[1, 2].set_xlabel("Radius (m)")
        axes[1, 2].set_ylabel("Time (Days)")
        fig.colorbar(im5, ax=axes[1, 2], label="Error (Pa)")

        plt.tight_layout()
        plt.show()
