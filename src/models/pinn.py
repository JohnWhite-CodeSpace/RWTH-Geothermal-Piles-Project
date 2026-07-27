from typing import Dict, Optional, Tuple, Type

import numpy as np
import torch
import torch.nn as nn


def error_metrics(pred: np.ndarray, true: np.ndarray) -> Tuple[float, float, float]:
    """
    Compute MSE, relative L2 norm error, and NRMSE between pred and true.

    Shared by `GeothermalPINN.evaluate` and `PINNEnsemble.evaluate`
    (src/models/ensemble.py) so both report metrics the same way.
    """
    mse = float(np.mean((pred - true) ** 2))
    rel_l2 = float(np.linalg.norm(pred - true) / np.linalg.norm(true))
    denom = max(float(np.max(np.abs(true))), 1e-12)
    nrmse = float(np.sqrt(mse) / denom)
    return mse, rel_l2, nrmse


class GeothermalPINN(nn.Module):
    """Physics-informed network for coupled temperature/pore-pressure fields."""

    def __init__(
        self,
        net_u: nn.Module,
        loss_fn: Optional[nn.Module] = None,
        device: str = "cpu",
    ):
<<<<<<< Updated upstream
        """
        Initialize the PINN.

        Args:
            net_u: Underlying network mapping (r, t) -> (T, u).
            loss_fn: Loss function used for the PDE/IC/BC residuals.
                Defaults to `nn.MSELoss()`.
            device: Device to run the model on.
        """
=======
>>>>>>> Stashed changes
        super().__init__()
        self.device = torch.device(device)
        self.net_u = net_u.to(self.device)
        self.adaptive_points = None

        self.loss_fn = loss_fn if loss_fn is not None else nn.MSELoss()

    def net_u_forward(
        self, r: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
<<<<<<< Updated upstream
        """
        Forward pass through the underlying network.

        Args:
            r: Dimensionless radius, shape (N, 1).
            t: Dimensionless time, shape (N, 1).

        Returns:
            Tuple (T, u) of dimensionless temperature and pore pressure.
        """
=======
>>>>>>> Stashed changes
        inputs = torch.cat((r, t), dim=1)
        outputs = self.net_u(inputs)
        T = outputs[:, 0:1]
        u = outputs[:, 1:2]
        return T, u

<<<<<<< Updated upstream
    def pde_residual(self, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the residuals of the dimensionless governing equations.

        Args:
            r: Dimensionless radius with `requires_grad=True`, shape (N, 1).
            t: Dimensionless time with `requires_grad=True`, shape (N, 1).

        Returns:
            Residuals of the heat and pore-pressure equations, shape (N, 2).
        """
        T, u = self.net_u_forward(r, t)

        # Gradients for temperature
        T_r = torch.autograd.grad(T, r, torch.ones_like(T), create_graph=True)[0]
        T_rr = torch.autograd.grad(T_r, r, torch.ones_like(T_r), create_graph=True)[0]
        T_t = torch.autograd.grad(T, t, torch.ones_like(T), create_graph=True)[0]

        # Gradients for pore-water pressure
        u_r = torch.autograd.grad(u, r, torch.ones_like(u), create_graph=True)[0]
        u_rr = torch.autograd.grad(u_r, r, torch.ones_like(u_r), create_graph=True)[0]
        u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]

        # Dimensionless governing equations
        pde_residual_T = T_t - self.C1 * ((1.0 / r) * T_r + T_rr)
        pde_residual_u = u_t - self.C2 * T_t - self.C3 * ((1.0 / r) * u_r + u_rr)
        return torch.cat((pde_residual_T, pde_residual_u), dim=1)

    def loss_pde(self, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute the PDE residual loss."""
        pde_residual = self.pde_residual(r, t)
        res_T = pde_residual[:, 0:1]
        res_u = pde_residual[:, 1:2]
        loss_T = self.loss_fn(res_T, torch.zeros_like(res_T))
        loss_u = self.loss_fn(res_u, torch.zeros_like(res_u))
        return self.w_T * loss_T + self.w_u * loss_u

    def loss_ic(self, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute the initial condition loss (T*=0, u*=0 at t=0)."""
        T, u = self.net_u_forward(r, t)
        ic_loss_T = self.loss_fn(T, torch.zeros_like(T))
        ic_loss_u = self.loss_fn(u, torch.zeros_like(u))
        return self.w_T * ic_loss_T + self.w_u * ic_loss_u

    def loss_bc_pile(self, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute the pile boundary loss (r*=0.0167): T*=1, du/dr=0."""
        T, u = self.net_u_forward(r, t)

        bc_loss_T = self.loss_fn(T, torch.ones_like(T))

        u_r = torch.autograd.grad(u, r, torch.ones_like(u), create_graph=True)[0]
        bc_loss_u = self.loss_fn(u_r, torch.zeros_like(u_r))

        return self.w_T * bc_loss_T + self.w_u * bc_loss_u

    def loss_bc_far(self, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute the far-field boundary loss (r*=1.0): T*=0, u*=0."""
        T, u = self.net_u_forward(r, t)

        bc_loss_T = self.loss_fn(T, torch.zeros_like(T))
        bc_loss_u = self.loss_fn(u, torch.zeros_like(u))

        return self.w_T * bc_loss_T + self.w_u * bc_loss_u

    def loss_u_nonneg(self, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Penalize negative excess pore-water pressure predictions.

        Excess pore pressure only builds up while heating in this
        scenario (Fuentes et al. 2016) -- u* should be >= 0 everywhere.
        The governing PDE and IC/BC losses don't enforce that on their
        own; a negative prediction elsewhere in the domain is a
        numerical artifact (see the u* heatmaps in
        data/processed/case3_weighted_pair.png for an example), not a
        feature the physics implies. `torch.relu(-u)` is 0 wherever
        u >= 0 and grows where u < 0, so squaring it via `self.loss_fn`
        gives a smooth soft-inequality-constraint penalty.
        """
        _, u = self.net_u_forward(r, t)
        violation = torch.relu(-u)
        return self.loss_fn(violation, torch.zeros_like(violation))

    def physics_residual_check(
        self, r: torch.Tensor, t: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute the PDE residual norm on held-out points.

        Evidence that the network learned the underlying physics rather
        than only fitting the IC/BC targets: `r`/`t` should be points
        the network never saw as *training* collocation points (e.g.
        the FDM reference grid, as opposed to the LHS-sampled
        `domain_points` passed to `train_net`). A residual still close
        to zero there is independent confirmation of physical
        understanding, not just curve-fitting to boundary/initial data.

        Args:
            r: Dimensionless radius, held-out points, shape (N, 1).
            t: Dimensionless time, held-out points, shape (N, 1).

        Returns:
            Dictionary with the RMS PDE residual for the temperature
            and pore-pressure equations.
        """
        r = r.clone().detach().to(self.device).requires_grad_(True)
        t = t.clone().detach().to(self.device).requires_grad_(True)
        residual = self.pde_residual(r, t)

        res_T = residual[:, 0:1].detach().cpu().numpy()
        res_u = residual[:, 1:2].detach().cpu().numpy()

        return {
            "T_pde_residual_rms": float(np.sqrt(np.mean(res_T**2))),
            "u_pde_residual_rms": float(np.sqrt(np.mean(res_u**2))),
        }

    def pinn_loss(self) -> torch.Tensor:
        """Compute and backpropagate the total training loss for one step."""
        self.optimizer.zero_grad()

        r_dom = self.domain_points[:, 0:1].to(self.device).requires_grad_(True)
        t_dom = self.domain_points[:, 1:2].to(self.device).requires_grad_(True)

        r_ic = self.ic_points[:, 0:1].to(self.device).requires_grad_(True)
        t_ic = self.ic_points[:, 1:2].to(self.device).requires_grad_(True)

        r_bc_pile = self.bc_pile_points[:, 0:1].to(self.device).requires_grad_(True)
        t_bc_pile = self.bc_pile_points[:, 1:2].to(self.device).requires_grad_(True)

        r_bc_far = self.bc_far_points[:, 0:1].to(self.device).requires_grad_(True)
        t_bc_far = self.bc_far_points[:, 1:2].to(self.device).requires_grad_(True)

        pde_loss = self.loss_pde(r_dom, t_dom)
        ic_loss = self.loss_ic(r_ic, t_ic)
        bc_pile_loss = self.loss_bc_pile(r_bc_pile, t_bc_pile)
        bc_far_loss = self.loss_bc_far(r_bc_far, t_bc_far)
        physical_loss = self.loss_u_nonneg(r_dom, t_dom)

        total_loss = (
            pde_loss
            + ic_loss
            + bc_pile_loss
            + bc_far_loss
            + self.w_physical * physical_loss
        )
        total_loss.backward()

        if self.iter % self.log_interval == 0:
            print(
                f"[{self.optimizer_name}]: Iteration {self.iter}, "
                f"Total loss: {total_loss.item():.3e}, "
                f"PDE Loss: {pde_loss.item():.3e}, "
                f"IC Loss: {ic_loss.item():.3e}, "
                f"Pile BC Loss: {bc_pile_loss.item():.3e}, "
                f"Far BC Loss: {bc_far_loss.item():.3e}, "
                f"u>=0 Loss: {physical_loss.item():.3e}"
            )
        self.iter += 1
        self.loss_history.append(total_loss.item())
        return total_loss
=======
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
>>>>>>> Stashed changes

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
<<<<<<< Updated upstream
        log_interval: int = 100,
        w_T: float = 1.0,
        w_u: float = 1.0,
        w_physical: float = 1.0,
    ) -> None:
        """
        Train the network on the PDE, initial condition, and boundary losses.

        Calling this more than once on the same instance (e.g. Adam
        first, then LBFGS) continues training from the current weights
        and appends to the existing loss history, giving a fine-tuning
        stage rather than starting over.

        Args:
            domain_points: Interior collocation points, columns [r, t].
            ic_points: Initial condition points, columns [r, t].
            bc_pile_points: Pile boundary points, columns [r, t].
            bc_far_points: Far-field boundary points, columns [r, t].
            C1: Nondimensional heat equation coefficient.
            C2: Nondimensional thermal-coupling coefficient.
            C3: Nondimensional pore-pressure diffusion coefficient.
            epochs: Number of training iterations.
            optimizer: Optimizer class to use (Adam or LBFGS).
            log_interval: Number of iterations between log prints.
            w_T: Weight applied to temperature loss terms.
            w_u: Weight applied to pore-pressure loss terms. u* is ~2-3
                orders of magnitude smaller than T* (see Table 1), so an
                equal-weight MSE lets the temperature loss dominate the
                gradient; raise w_u to counteract that imbalance.
            w_physical: Weight applied to the u* >= 0 soft-constraint
                penalty (`loss_u_nonneg`). Set to 0 to disable it.
        """
        self.domain_points = domain_points
        self.ic_points = ic_points
        self.bc_pile_points = bc_pile_points
        self.bc_far_points = bc_far_points
=======
        log_interval: int = 500,
        base_grid_points: Optional[torch.Tensor] = None,
        adaptive_method: str = "rar",
        adaptive_interval: int = 1000,
        num_adaptive_points: int = 100,
        train_T: bool = True,
        train_u: bool = True,
    ) -> None:
        def prepare_pts(pts):
            r = pts[:, 0:1].to(self.device).requires_grad_(True)
            t = pts[:, 1:2].to(self.device).requires_grad_(True)
            return r, t

        r_dom, t_dom = prepare_pts(domain_points)
        r_ic, t_ic = prepare_pts(ic_points)
        r_bc_p, t_bc_p = prepare_pts(bc_pile_points)
        r_bc_f, t_bc_f = prepare_pts(bc_far_points)
>>>>>>> Stashed changes

        self.C1 = C1
        self.C2 = C2
        self.C3 = C3
        self.w_T = w_T
        self.w_u = w_u
        self.w_physical = w_physical

<<<<<<< Updated upstream
        self.net_u.train()
        self.iter = 0
        self.log_interval = log_interval
        if not hasattr(self, "loss_history"):
            self.loss_history = []

        if optimizer is torch.optim.LBFGS:
            self.optimizer_name = "LBFGS"
            self.optimizer = optimizer(
                self.net_u.parameters(),
                lr=1,
                max_iter=epochs,
                max_eval=None,
                history_size=100,
                tolerance_grad=1e-8,
                tolerance_change=1e-8,
                line_search_fn="strong_wolfe",
            )
            self.optimizer.step(self.pinn_loss)
        else:
            self.optimizer_name = "Adam"
            self.optimizer = optimizer(self.net_u.parameters(), lr=1e-3)
            for _ in range(epochs):
                self.pinn_loss()
                self.optimizer.step()

    def predict(self, r: torch.Tensor, t: torch.Tensor) -> Tuple:
        """
        Predict temperature and pore pressure at the given points.

        Args:
            r: Dimensionless radius.
            t: Dimensionless time.

        Returns:
            Tuple (T, u) of NumPy arrays with predicted values.
        """
=======
        if train_T:
            print("\n" + "="*50)
            print(f" STAGE 1A: Training Temperature Network (Adam) - {adaptive_method.upper()}")
            print("="*50)

            self.net_T.train()
            opt_T_adam = torch.optim.Adam(self.net_T.parameters(), lr=1e-3)

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
                    print(f"  --> [Adaptive {adaptive_method.upper()}] Added {r_new.shape[0]} points. Total Domain: {r_dom.shape[0]}")

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
                total_loss_T.backward()
                opt_T_adam.step()

                if ep % log_interval == 0:
                    print(f"[Stage 1A - Adam] Iter {ep:4d} | Total Loss T: {total_loss_T.item():.3e}")

            print("\n>>> STAGE 1B: Fine-tuning Temperature with L-BFGS...")
            opt_T_lbfgs = torch.optim.LBFGS(
                self.net_T.parameters(), lr=0.5, max_iter=5000, history_size=100,
                tolerance_grad=1e-9, tolerance_change=1e-11, line_search_fn="strong_wolfe"
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

                total_loss_T = loss_pde_T + 100.0 * (loss_ic_T + loss_bc_p_T) + 10.0 * loss_bc_f_T
                total_loss_T.backward()
                return total_loss_T

            opt_T_lbfgs.step(closure_T)
            print(f"[Stage 1B - LBFGS] Final Loss T: {closure_T().item():.3e}")

            print("\n>>> Freezing Temperature Network...")
            for param in self.net_T.parameters():
                param.requires_grad = False
            self.net_T.eval()

        if train_u:
            r_dom, t_dom = prepare_pts(domain_points)

            print("\n" + "="*50)
            print(f" STAGE 2A: Training Pore Pressure Network (Adam) - {adaptive_method.upper()}")
            print("="*50)

            self.net_u.train()
            opt_u_adam = torch.optim.Adam(self.net_u.parameters(), lr=1e-3)

            for ep in range(epochs):
                if base_grid_points is not None and ep > 0 and ep % adaptive_interval == 0:
                    new_pts = self.adaptive_sampling(
                        num_points=num_adaptive_points,
                        base_grid_points=base_grid_points,
                        method=adaptive_method,
                        stage='u',
                        C2=C2,
                        C3=C3
                    )
                    r_new = new_pts[:, 0:1].clone().detach().requires_grad_(True)
                    t_new = new_pts[:, 1:2].clone().detach().requires_grad_(True)
                    r_dom = torch.cat((r_dom, r_new), dim=0)
                    t_dom = torch.cat((t_dom, t_new), dim=0)
                    print(f"  --> [Adaptive {adaptive_method.upper()}] Added {r_new.shape[0]} points. Total Domain: {r_dom.shape[0]}")

                opt_u_adam.zero_grad()

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

                total_loss_u = loss_pde_u + 100.0 * (loss_ic_u + loss_bc_p_u) + 10.0 * loss_bc_f_u
                total_loss_u.backward()
                opt_u_adam.step()

                if ep % log_interval == 0:
                    print(f"[Stage 2A - Adam] Iter {ep:4d} | Total Loss u: {total_loss_u.item():.3e}")

            print("\n>>> STAGE 2B: Fine-tuning Pore Pressure with L-BFGS (The crucial step!)...")
            opt_u_lbfgs = torch.optim.LBFGS(
                self.net_u.parameters(), lr=0.5, max_iter=5000, history_size=100,
                tolerance_grad=1e-9, tolerance_change=1e-11, line_search_fn="strong_wolfe"
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

                total_loss_u = loss_pde_u + 100.0 * (loss_ic_u + loss_bc_p_u) + 10.0 * loss_bc_f_u
                total_loss_u.backward()
                return total_loss_u

            opt_u_lbfgs.step(closure_u)
            print(f"[Stage 2B - LBFGS] Final Loss u: {closure_u().item():.3e}")

    def predict(self, r: torch.Tensor, t: torch.Tensor) -> Tuple:
        self.net_T.eval()
>>>>>>> Stashed changes
        self.net_u.eval()
        with torch.no_grad():
            r = r.to(self.device)
            t = t.to(self.device)
            T_pred, u_pred = self.net_u_forward(r, t)
            return T_pred.cpu().numpy(), u_pred.cpu().numpy()

    def evaluate(
        self,
<<<<<<< Updated upstream
        r: torch.Tensor,
        t: torch.Tensor,
        T_true: np.ndarray,
        u_true: np.ndarray,
    ) -> Dict[str, float]:
        """
        Evaluate predictions against reference (e.g. FDM) data.

        Args:
            r: Dimensionless radius of the reference points, shape (N, 1).
            t: Dimensionless time of the reference points, shape (N, 1).
            T_true: Reference dimensionless temperature, shape (N, 1).
            u_true: Reference dimensionless pore pressure, shape (N, 1).

        Returns:
            Dictionary with MSE, relative L2 error, and NRMSE for T and
            u, plus the PDE residual RMS on these same points (see
            `physics_residual_check`) as independent evidence of
            physical understanding, not just accuracy against FDM.
            Relative L2 error (||pred-true|| / ||true||) can look huge
            for a field that is close to zero almost everywhere (like
            u* here, away from the pile) even when the absolute fit is
            good, because the denominator is dominated by a small
            localized signal. NRMSE (RMSE normalized by max(|true|)) is
            far less sensitive to that and better reflects fit quality
            relative to the field's actual dynamic range.
        """
        T_pred, u_pred = self.predict(r, t)

        T_mse, T_rel_l2, T_nrmse = error_metrics(T_pred, T_true)
        u_mse, u_rel_l2, u_nrmse = error_metrics(u_pred, u_true)

        metrics = {
            "T_mse": T_mse,
            "T_rel_l2": T_rel_l2,
            "T_nrmse": T_nrmse,
            "u_mse": u_mse,
            "u_rel_l2": u_rel_l2,
            "u_nrmse": u_nrmse,
        }
        metrics.update(self.physics_residual_check(r, t))
        return metrics
=======
        case_num: int,
        t_c: float = 1e7,
        u_c: float = 8e5,
        r_max: float = 30.0,
        T_initial: float = 12.0,
        delta_T: float = 38.0,
    ) -> None:
        try:
            from src.utils.data_loader import load_single_case
        except ImportError:
            print("Error: 'data_loader.py' module not found. Please check the import path.")
            return

        print(f"Loading reference FDM data for Case {case_num}...")
        result = load_single_case(case_num)

        if isinstance(result, dict):
            print(f"Failed to load data: {result['error_message']}")
            return

        temp_df, pressure_df = result

        r_true = temp_df.columns.astype(float).to_numpy()
        t_days = temp_df.index.astype(float).to_numpy()
        T_true = temp_df.to_numpy(dtype=float)
        u_true = pressure_df.to_numpy(dtype=float)

        seconds_per_day = 24.0 * 60.0 * 60.0
        t_sec = t_days * seconds_per_day

        r_mesh, t_mesh = np.meshgrid(r_true, t_sec, indexing="xy")

        r_star = r_mesh.flatten() / r_max
        t_star = t_mesh.flatten() / t_c

        r_tensor = torch.tensor(r_star, dtype=torch.float32).unsqueeze(1).to(self.device)
        t_tensor = torch.tensor(t_star, dtype=torch.float32).unsqueeze(1).to(self.device)

        T_pred_star, u_pred_star = self.predict(r_tensor, t_tensor)

        T_pred = (T_pred_star.reshape(T_true.shape) * delta_T) + T_initial
        u_pred = u_pred_star.reshape(u_true.shape) * u_c

        error_T = np.linalg.norm(T_true - T_pred) / np.linalg.norm(T_true)
        error_u = np.linalg.norm(u_true - u_pred) / (np.linalg.norm(u_true) + 1e-8)

        print(f"\n--- Evaluation Results for Case {case_num} ---")
        print(f"Temperature L2 Relative Error   : {error_T * 100:.3f}%")
        print(f"Pore Pressure L2 Relative Error : {error_u * 100:.3f}%")

        t_days_mesh = t_mesh / seconds_per_day

        error_T_map = np.abs(T_true - T_pred)
        error_u_map = np.abs(u_true - u_pred)

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        im0 = axes[0, 0].pcolormesh(r_mesh, t_days_mesh, T_true, shading="auto", cmap="hot")
        axes[0, 0].set_title("Temperature: FDM (Reference)")
        axes[0, 0].set_xlabel("Radius (m)")
        axes[0, 0].set_ylabel("Time (Days)")
        fig.colorbar(im0, ax=axes[0, 0], label="Temperature (°C)")

        im1 = axes[0, 1].pcolormesh(r_mesh, t_days_mesh, T_pred, shading="auto", cmap="hot")
        axes[0, 1].set_title("Temperature: PINN (Prediction)")
        axes[0, 1].set_xlabel("Radius (m)")
        axes[0, 1].set_ylabel("Time (Days)")
        fig.colorbar(im1, ax=axes[0, 1], label="Temperature (°C)")

        im2 = axes[0, 2].pcolormesh(r_mesh, t_days_mesh, error_T_map, shading="auto", cmap="Reds")
        axes[0, 2].set_title(f"Temp Error (Rel L2: {error_T * 100:.2f}%)")
        axes[0, 2].set_xlabel("Radius (m)")
        axes[0, 2].set_ylabel("Time (Days)")
        fig.colorbar(im2, ax=axes[0, 2], label="Error (°C)")

        im3 = axes[1, 0].pcolormesh(r_mesh, t_days_mesh, u_true, shading="auto", cmap="jet")
        axes[1, 0].set_title("Pore Pressure: FDM (Reference)")
        axes[1, 0].set_xlabel("Radius (m)")
        axes[1, 0].set_ylabel("Time (Days)")
        fig.colorbar(im3, ax=axes[1, 0], label="Excess Pore Pressure (Pa)")

        im4 = axes[1, 1].pcolormesh(r_mesh, t_days_mesh, u_pred, shading="auto", cmap="jet")
        axes[1, 1].set_title("Pore Pressure: PINN (Prediction)")
        axes[1, 1].set_xlabel("Radius (m)")
        axes[1, 1].set_ylabel("Time (Days)")
        fig.colorbar(im4, ax=axes[1, 1], label="Excess Pore Pressure (Pa)")

        im5 = axes[1, 2].pcolormesh(r_mesh, t_days_mesh, error_u_map, shading="auto", cmap="Reds")
        axes[1, 2].set_title(f"Pressure Error (Rel L2: {error_u * 100:.2f}%)")
        axes[1, 2].set_xlabel("Radius (m)")
        axes[1, 2].set_ylabel("Time (Days)")
        fig.colorbar(im5, ax=axes[1, 2], label="Error (Pa)")

        plt.tight_layout()
        plt.show()

class GeothermalInversePINN(nn.Module):
    """Physics-informed network for solving the INVERSE geothermal problem."""

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
            torch.tensor([initial_log_k], dtype=torch.float32, device=self.device)
        )

        self.loss_fn = loss_fn if loss_fn is not None else nn.MSELoss()

    def get_current_k(self) -> float:
        return (10 ** self.log_k).item()

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
        log_interval: int = 500,
        burn_in_epochs: int = 3000,
    ) -> Tuple[list, list, list]:
        
        def prepare_pts(pts):
            r = pts[:, 0:1].to(self.device).requires_grad_(True)
            t = pts[:, 1:2].to(self.device).requires_grad_(True)
            return r, t

        r_dom, t_dom = prepare_pts(domain_points)
        r_ic, t_ic = prepare_pts(ic_points)
        r_bc_p, t_bc_p = prepare_pts(bc_pile_points)
        r_bc_f, t_bc_f = prepare_pts(bc_far_points)
        
        r_sensor = sensor_points[:, 0:1].to(self.device)
        t_sensor = sensor_points[:, 1:2].to(self.device)
        u_sensor_true = sensor_u_true.to(self.device)

        print("\n" + "="*60)
        print(f" INVERSE MODEL TRAINING INITIATED (Burn-in: {burn_in_epochs} epochs)")
        print(f" Initial Permeability Guess (k): {self.get_current_k():.2e} m/s")
        print("="*60)

        self.net_u.train()
        
        opt_u = torch.optim.Adam(self.net_u.parameters(), lr=1e-3)
        opt_k = torch.optim.Adam([self.log_k], lr=1e-2) 

        scheduler_u = torch.optim.lr_scheduler.StepLR(opt_u, step_size=3000, gamma=0.5)

        T_dom_fixed = self.net_T(torch.cat((r_dom, t_dom), dim=1))
        T_t_fixed = torch.autograd.grad(T_dom_fixed, t_dom, torch.ones_like(T_dom_fixed), create_graph=True)[0].detach()

        history_epochs = []
        history_loss = []
        history_k = []

        lam_data = 200.0 

        for ep in range(epochs):
            opt_u.zero_grad()
            opt_k.zero_grad()

            current_k = 10 ** self.log_k
            C1, C2, C3_tensor, u_c_val = physics_config.calculate_physics_constants_tensor(
                k_tensor=current_k, Ks=2.0e6, t_c=t_c_val
            )

            u_dom = self.net_u(torch.cat((r_dom, t_dom), dim=1))
            u_r = torch.autograd.grad(u_dom, r_dom, torch.ones_like(u_dom), create_graph=True)[0]
            u_rr = torch.autograd.grad(u_r, r_dom, torch.ones_like(u_r), create_graph=True)[0]
            u_t = torch.autograd.grad(u_dom, t_dom, torch.ones_like(u_dom), create_graph=True)[0]

            res_u = u_t - C2 * T_t_fixed - C3_tensor * ((1.0 / r_dom) * u_r + u_rr)
            loss_pde = self.loss_fn(res_u, torch.zeros_like(res_u))

            u_ic = self.net_u(torch.cat((r_ic, t_ic), dim=1))
            loss_ic = self.loss_fn(u_ic, torch.zeros_like(u_ic))

            u_bc_p = self.net_u(torch.cat((r_bc_p, t_bc_p), dim=1))
            u_r_bc_p = torch.autograd.grad(u_bc_p, r_bc_p, torch.ones_like(u_bc_p), create_graph=True)[0]
            loss_bc_p = self.loss_fn(u_r_bc_p, torch.zeros_like(u_r_bc_p))

            u_bc_f = self.net_u(torch.cat((r_bc_f, t_bc_f), dim=1))
            loss_bc_f = self.loss_fn(u_bc_f, torch.zeros_like(u_bc_f))

            u_sensor_pred = self.net_u(torch.cat((r_sensor, t_sensor), dim=1))
            loss_data = self.loss_fn(u_sensor_pred, u_sensor_true)

            total_loss = loss_pde + 10.0 * (loss_ic + loss_bc_p + loss_bc_f) + lam_data * loss_data
            total_loss.backward()
            
            opt_u.step() 
            scheduler_u.step()
            
            if ep >= burn_in_epochs:
                opt_k.step()

            if ep % log_interval == 0:
                history_epochs.append(ep)
                history_loss.append(total_loss.item())
                history_k.append(self.get_current_k())
                print(f"[Inverse Iter {ep:5d}] Loss: {total_loss.item():.3e} | Estimated k: {self.get_current_k():.3e} m/s")

        print("\n>>> PHASE 2: Fine-tuning Inverse Model with L-BFGS...")
        opt_lbfgs = torch.optim.LBFGS(
            list(self.net_u.parameters()) + [self.log_k], 
            lr=0.1, max_iter=5000, history_size=100,
            tolerance_grad=1e-9, tolerance_change=1e-11, line_search_fn="strong_wolfe"
        )

        def closure_inverse():
            opt_lbfgs.zero_grad()
            current_k = 10 ** self.log_k
            _, C2, C3_tensor, _ = physics_config.calculate_physics_constants_tensor(
                k_tensor=current_k, Ks=2.0e6, t_c=t_c_val
            )
            u_dom = self.net_u(torch.cat((r_dom, t_dom), dim=1))
            u_r = torch.autograd.grad(u_dom, r_dom, torch.ones_like(u_dom), create_graph=True)[0]
            u_rr = torch.autograd.grad(u_r, r_dom, torch.ones_like(u_r), create_graph=True)[0]
            u_t = torch.autograd.grad(u_dom, t_dom, torch.ones_like(u_dom), create_graph=True)[0]
            res_u = u_t - C2 * T_t_fixed - C3_tensor * ((1.0 / r_dom) * u_r + u_rr)
            loss_pde = self.loss_fn(res_u, torch.zeros_like(res_u))

            u_ic = self.net_u(torch.cat((r_ic, t_ic), dim=1))
            loss_ic = self.loss_fn(u_ic, torch.zeros_like(u_ic))

            u_bc_p = self.net_u(torch.cat((r_bc_p, t_bc_p), dim=1))
            u_r_bc_p = torch.autograd.grad(u_bc_p, r_bc_p, torch.ones_like(u_bc_p), create_graph=True)[0]
            loss_bc_p = self.loss_fn(u_r_bc_p, torch.zeros_like(u_r_bc_p))

            u_bc_f = self.net_u(torch.cat((r_bc_f, t_bc_f), dim=1))
            loss_bc_f = self.loss_fn(u_bc_f, torch.zeros_like(u_bc_f))

            u_sensor_pred = self.net_u(torch.cat((r_sensor, t_sensor), dim=1))
            loss_data = self.loss_fn(u_sensor_pred, u_sensor_true)

            total_loss = loss_pde + 10.0 * (loss_ic + loss_bc_p + loss_bc_f) + lam_data * loss_data
            total_loss.backward()
            return total_loss

        opt_lbfgs.step(closure_inverse)
        print(f"[Inverse L-BFGS] Final Loss: {closure_inverse().item():.3e} | Estimated k: {self.get_current_k():.3e} m/s")
        print("\n>>> INVERSE TRAINING COMPLETE")

        return history_epochs, history_loss, history_k
>>>>>>> Stashed changes
