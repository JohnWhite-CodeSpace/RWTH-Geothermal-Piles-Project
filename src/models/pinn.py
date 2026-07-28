from typing import Dict, Optional, Tuple, Type
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def error_metrics(pred: np.ndarray, true: np.ndarray) -> Tuple[float, float, float]:
    """Compute MSE, relative L2 norm error, and NRMSE between pred and true."""
    mse = float(np.mean((pred - true) ** 2))
    rel_l2 = float(np.linalg.norm(pred - true) / np.linalg.norm(true))
    denom = max(float(np.max(np.abs(true))), 1e-12)
    nrmse = float(np.sqrt(mse) / denom)
    return mse, rel_l2, nrmse


class GeothermalPINN(nn.Module):
    """Physics-informed network for coupled temperature/pore-pressure fields (Forward Problem)."""

    def __init__(
        self,
        net_T: nn.Module,
        net_u: nn.Module,
        loss_fn: Optional[nn.Module] = None,
        device: str = "cpu",
    ):
        super().__init__()
        self.device = torch.device(device)
        self.net_T = net_T.to(self.device)
        self.net_u = net_u.to(self.device)
        self.adaptive_points = None
        self.loss_fn = loss_fn if loss_fn is not None else nn.MSELoss()

    def net_u_forward(
        self, r: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs = torch.cat((r, t), dim=1)
        T = self.net_T(inputs)
        u = self.net_u(inputs)
        return T, u

    def adaptive_sampling(
        self,
        num_points: int,
        base_grid_points: torch.Tensor,
        method: str,
        stage: str = 'u',
        C1: float = 0.0,
        C2: float = 0.0,
        C3: float = 0.0
    ) -> torch.Tensor:
        base_grid_r = base_grid_points[:, 0:1].to(self.device).requires_grad_(True)
        base_grid_t = base_grid_points[:, 1:2].to(self.device).requires_grad_(True)

        self.eval()

        if stage == 'T':
            val = self.net_T(torch.cat((base_grid_r, base_grid_t), dim=1))
        else:
            val = self.net_u(torch.cat((base_grid_r, base_grid_t), dim=1))

        if method == 'gar':
            val_r = torch.autograd.grad(val, base_grid_r, torch.ones_like(val), create_graph=True, retain_graph=True)[0]
            val_t = torch.autograd.grad(val, base_grid_t, torch.ones_like(val), create_graph=True, retain_graph=True)[0]

            _, idx_val_r = torch.topk(val_r.squeeze().abs(), num_points)
            _, idx_val_t = torch.topk(val_t.squeeze().abs(), num_points)

            idx = torch.cat((idx_val_r, idx_val_t), dim=0)
            idx = torch.unique(idx)

        elif method == 'rar':
            val_r = torch.autograd.grad(val, base_grid_r, torch.ones_like(val), create_graph=True, retain_graph=True)[0]
            val_rr = torch.autograd.grad(val_r, base_grid_r, torch.ones_like(val_r), create_graph=True, retain_graph=True)[0]
            val_t = torch.autograd.grad(val, base_grid_t, torch.ones_like(val), create_graph=True, retain_graph=True)[0]

            if stage == 'T':
                pde_residual = val_t - C1 * ((1.0 / base_grid_r) * val_r + val_rr)
            else:
                T_fixed = self.net_T(torch.cat((base_grid_r, base_grid_t), dim=1))
                T_t_fixed = torch.autograd.grad(T_fixed, base_grid_t, torch.ones_like(T_fixed), create_graph=True, retain_graph=True)[0].detach()
                pde_residual = val_t - C2 * T_t_fixed - C3 * ((1.0 / base_grid_r) * val_r + val_rr)

            _, idx = torch.topk(pde_residual.squeeze().abs(), num_points)
        else:
            raise ValueError("Adaptive method must be either 'gar' or 'rar'.")

        self.train()
        self.adaptive_points = base_grid_points[idx, :]
        return self.adaptive_points.to(self.device)

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
        log_interval: int = 500,
        base_grid_points: Optional[torch.Tensor] = None,
        adaptive_method: str = "rar",
        adaptive_interval: int = 1000,
        num_adaptive_points: int = 100,
        train_T: bool = True,
        train_u: bool = True,
        sensor_points_T: Optional[torch.Tensor] = None,
        sensor_T_true: Optional[torch.Tensor] = None
    ) -> None:
        def prepare_pts(pts):
            r = pts[:, 0:1].to(self.device).requires_grad_(True)
            t = pts[:, 1:2].to(self.device).requires_grad_(True)
            return r, t

        r_dom, t_dom = prepare_pts(domain_points)
        r_ic, t_ic = prepare_pts(ic_points)
        r_bc_p, t_bc_p = prepare_pts(bc_pile_points)
        r_bc_f, t_bc_f = prepare_pts(bc_far_points)

        if train_T:
            print("\n" + "="*50)
            print(f" STAGE 1A: Training Temperature Network (Adam) - {adaptive_method.upper()}")
            if sensor_points_T is not None:
                print(" >>> Data-Driven Forward Mode: USING TEMPERATURE SENSOR DATA!")
            print("="*50)

            self.net_T.train()
            opt_T_adam = torch.optim.Adam(self.net_T.parameters(), lr=1e-3)
            scheduler_T = torch.optim.lr_scheduler.StepLR(opt_T_adam, step_size=4000, gamma=0.5)

            for ep in range(epochs):
                if base_grid_points is not None and ep > 0 and ep % adaptive_interval == 0:
                    new_pts = self.adaptive_sampling(
                        num_points=num_adaptive_points,
                        base_grid_points=base_grid_points,
                        method=adaptive_method,
                        stage='T',
                        C1=C1
                    )
                    r_new = new_pts[:, 0:1].clone().detach().requires_grad_(True)
                    t_new = new_pts[:, 1:2].clone().detach().requires_grad_(True)
                    r_dom = torch.cat((r_dom, r_new), dim=0)
                    t_dom = torch.cat((t_dom, t_new), dim=0)

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

                total_loss_T = loss_pde_T + 100.0 * (loss_ic_T + loss_bc_p_T) + 10.0 * loss_bc_f_T
                
                # --- اعمال دیتای سنسور دما ---
                if sensor_points_T is not None and sensor_T_true is not None:
                    T_sensor_pred = self.net_T(sensor_points_T)
                    loss_data_T = self.loss_fn(T_sensor_pred, sensor_T_true)
                    total_loss_T = total_loss_T + 100.0 * loss_data_T

                total_loss_T.backward()
                opt_T_adam.step()
                scheduler_T.step()

                if ep % log_interval == 0:
                    current_lr = opt_T_adam.param_groups[0]['lr']
                    loss_data_val = loss_data_T.item() if (sensor_points_T is not None) else 0.0
                    print(f"[Adam T] Iter {ep:5d} | Total Loss: {total_loss_T.item():.3e} | Data Loss: {loss_data_val:.3e} | LR: {current_lr:.2e}")

            print("\n>>> Freezing Temperature Network...")
            for param in self.net_T.parameters():
                param.requires_grad = False
            self.net_T.eval()

class GeothermalInversePINN(nn.Module):
    """Physics-informed network for solving the INVERSE geothermal problem with HARD BOUNDARIES and L-BFGS."""

    def __init__(
        self,
        net_T: nn.Module,
        net_u: nn.Module,
        initial_log_k: float = -10.0,
        loss_fn: Optional[nn.Module] = None,
        device: str = "cpu",
    ):
        super().__init__()
        self.device = torch.device(device)
        self.net_T = net_T.to(self.device)
        self.net_u = net_u.to(self.device)
        
        for param in self.net_T.parameters():
            param.requires_grad = False
        self.net_T.eval()

        self.log_k = nn.Parameter(
            torch.tensor([initial_log_k], dtype=torch.float64, device=self.device)
        )
        self.loss_fn = loss_fn if loss_fn is not None else nn.MSELoss()

    def get_current_k(self) -> float:
        return (10 ** self.log_k).item()

    def predict_u(self, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        HARD INITIAL CONDITION: Multiplies the raw NN output by 't'.
        This guarantees that at t=0, u is exactly 0.0, fulfilling the physical IC unconditionally.
        """
        raw_u = self.net_u(torch.cat((r, t), dim=1))
        return raw_u * t

    def train_inverse(
        self,
        domain_points: torch.Tensor,
        ic_points: torch.Tensor,
        bc_pile_points: torch.Tensor,
        bc_far_points: torch.Tensor,
        sensor_points: torch.Tensor,
        sensor_u_true: torch.Tensor,
        physics_config: Type,
        t_c_val: float,
        epochs: int = 10000,
        lbfgs_epochs: int = 500,  
        log_interval: int = 500,
        burn_in_epochs: int = 3000,
        lr_k: float = 1e-2
    ) -> Tuple[list, list, list]:
        
        def prepare_pts(pts):
            r = pts[:, 0:1].to(self.device).requires_grad_(True)
            t = pts[:, 1:2].to(self.device).requires_grad_(True)
            return r, t

        r_dom, t_dom = prepare_pts(domain_points)
        
        sort_idx = torch.argsort(t_dom.squeeze())
        r_dom = r_dom[sort_idx].clone().detach().requires_grad_(True)
        t_dom = t_dom[sort_idx].clone().detach().requires_grad_(True)
        
        # ic_points is loaded but we no longer compute IC loss thanks to Hard IC!
        r_bc_p, t_bc_p = prepare_pts(bc_pile_points)
        r_bc_f, t_bc_f = prepare_pts(bc_far_points)
        
        r_sensor = sensor_points[:, 0:1].to(self.device)
        t_sensor = sensor_points[:, 1:2].to(self.device)
        u_sensor_true = sensor_u_true.to(self.device)

        print("\n" + "="*70)
        print(f" PHASE 1: INVERSE MODEL - ADAM (Burn-in: {burn_in_epochs}, Hard IC applied)")
        print(f" Initial Permeability Guess (k): {self.get_current_k():.2e} m/s")
        print("="*70)

        self.net_u.train()

        opt_u = torch.optim.Adam(self.net_u.parameters(), lr=1e-3)
        opt_k = torch.optim.Adam([self.log_k], lr=lr_k)

        best_loss_data = float('inf')
        best_k = self.get_current_k()

        scheduler_u = torch.optim.lr_scheduler.StepLR(opt_u, step_size=3000, gamma=0.5)

        T_dom_fixed = self.net_T(torch.cat((r_dom, t_dom), dim=1))
        T_t_fixed = torch.autograd.grad(T_dom_fixed, t_dom, torch.ones_like(T_dom_fixed), create_graph=True)[0].detach()

        history_epochs = []
        history_loss = []
        history_k = []

        w_pde = 1.0
        w_bc = 1.0
        w_data = 100.0 

        # ---------------------------------------------------------
        # 1. ADAM OPTIMIZATION LOOP
        # ---------------------------------------------------------
        for ep in range(epochs):
            current_k = 10 ** self.log_k
            C1, C2, C3_tensor, u_c_val = physics_config.calculate_physics_constants_tensor(
                k_tensor=current_k, Ks=2.0e6, t_c=t_c_val
            )

            u_dom = self.predict_u(r_dom, t_dom)
            u_r = torch.autograd.grad(u_dom, r_dom, torch.ones_like(u_dom), create_graph=True)[0]
            u_rr = torch.autograd.grad(u_r, r_dom, torch.ones_like(u_r), create_graph=True)[0]
            u_t = torch.autograd.grad(u_dom, t_dom, torch.ones_like(u_dom), create_graph=True)[0]

            res_u = u_t - C2 * T_t_fixed - C3_tensor * ((1.0 / r_dom) * u_r + u_rr)
            loss_pde = torch.mean(res_u ** 2)

            # NOTE: loss_ic is REMOVED! Hard IC automatically guarantees zero initial condition.

            u_bc_p = self.predict_u(r_bc_p, t_bc_p)
            u_r_bc_p = torch.autograd.grad(u_bc_p, r_bc_p, torch.ones_like(u_bc_p), create_graph=True)[0]
            loss_bc_p = self.loss_fn(u_r_bc_p, torch.zeros_like(u_r_bc_p))

            u_bc_f = self.predict_u(r_bc_f, t_bc_f)
            loss_bc_f = self.loss_fn(u_bc_f, torch.zeros_like(u_bc_f))

            loss_bc_total = loss_bc_p + loss_bc_f

            u_sensor_pred = self.predict_u(r_sensor, t_sensor)
            loss_data = self.loss_fn(u_sensor_pred, u_sensor_true)

            opt_u.zero_grad()
            opt_k.zero_grad()
            
            total_loss = w_pde * loss_pde + w_bc * loss_bc_total + w_data * loss_data
            total_loss.backward()
            
            opt_u.step() 
            scheduler_u.step()
            
            if ep >= burn_in_epochs:
                opt_k.step()

            if loss_data.item() < best_loss_data:
                best_loss_data = loss_data.item()
                best_k = self.get_current_k()

            if ep % log_interval == 0:
                history_epochs.append(ep)
                history_loss.append(total_loss.item())
                history_k.append(self.get_current_k())
                print(f"[Adam Iter {ep:5d}] Loss: {total_loss.item():.3e} | k: {self.get_current_k():.3e}")

        # ---------------------------------------------------------
        # 2. L-BFGS OPTIMIZATION LOOP (FINE-TUNING)
        # ---------------------------------------------------------
        print("\n" + "="*70)
        print(f" PHASE 2: INVERSE MODEL - L-BFGS FINE-TUNING")
        print("="*70)

        opt_lbfgs = torch.optim.LBFGS(
            list(self.net_u.parameters()) + [self.log_k],
            lr=0.1,
            max_iter=20,
            max_eval=25,
            tolerance_grad=1e-7,
            tolerance_change=1e-9,
            history_size=50,
            line_search_fn="strong_wolfe"
        )

        def closure():
            opt_lbfgs.zero_grad()
            current_k_val = 10 ** self.log_k
            C1_l, C2_l, C3_tensor_l, _ = physics_config.calculate_physics_constants_tensor(
                k_tensor=current_k_val, Ks=2.0e6, t_c=t_c_val
            )

            u_dom_l = self.predict_u(r_dom, t_dom)
            u_r_l = torch.autograd.grad(u_dom_l, r_dom, torch.ones_like(u_dom_l), create_graph=True)[0]
            u_rr_l = torch.autograd.grad(u_r_l, r_dom, torch.ones_like(u_r_l), create_graph=True)[0]
            u_t_l = torch.autograd.grad(u_dom_l, t_dom, torch.ones_like(u_dom_l), create_graph=True)[0]

            res_u_l = u_t_l - C2_l * T_t_fixed - C3_tensor_l * ((1.0 / r_dom) * u_r_l + u_rr_l)
            loss_pde_l = torch.mean(res_u_l ** 2)

            u_bc_p_l = self.predict_u(r_bc_p, t_bc_p)
            u_r_bc_p_l = torch.autograd.grad(u_bc_p_l, r_bc_p, torch.ones_like(u_bc_p_l), create_graph=True)[0]
            loss_bc_p_l = self.loss_fn(u_r_bc_p_l, torch.zeros_like(u_r_bc_p_l))

            u_bc_f_l = self.predict_u(r_bc_f, t_bc_f)
            loss_bc_f_l = self.loss_fn(u_bc_f_l, torch.zeros_like(u_bc_f_l))

            loss_bc_total_l = loss_bc_p_l + loss_bc_f_l

            u_sensor_pred_l = self.predict_u(r_sensor, t_sensor)
            loss_data_l = self.loss_fn(u_sensor_pred_l, u_sensor_true)

            total_loss_l = w_pde * loss_pde_l + w_bc * loss_bc_total_l + w_data * loss_data_l
            total_loss_l.backward()
            
            return total_loss_l

        for ep in range(epochs, epochs + lbfgs_epochs):
            loss_val = opt_lbfgs.step(closure)
            
            if ep % max(1, log_interval // 5) == 0 or ep == epochs + lbfgs_epochs - 1:
                current_loss = loss_val.item()
                history_epochs.append(ep)
                history_loss.append(current_loss)
                history_k.append(self.get_current_k())
                
                if current_loss < best_loss_data:
                    best_loss_data = current_loss
                    best_k = self.get_current_k()

                print(f"[L-BFGS Iter {ep:5d}] Loss: {current_loss:.3e} | k: {self.get_current_k():.3e}")

        print("\n>>> INVERSE TRAINING COMPLETE")
        print(f">>> Best Estimated k: {best_k:.3e} m/s")

        return history_epochs, history_loss, history_k