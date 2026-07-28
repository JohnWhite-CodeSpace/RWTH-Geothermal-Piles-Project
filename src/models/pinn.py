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
    """Physics-informed network for coupled temperature/pore-pressure fields.

    The underlying network predicts the dimensionless temperature T* and
    excess pore-water pressure u* as functions of dimensionless radius r*
    and time t*.
    """

    def __init__(
        self,
        net_u: nn.Module,
        loss_fn: Optional[nn.Module] = None,
        device: str = "cpu",
    ):
        """
        Initialize the PINN.

        Args:
            net_u: Underlying network mapping (r, t) -> (T, u).
            loss_fn: Loss function used for the PDE/IC/BC residuals.
                Defaults to `nn.MSELoss()`.
            device: Device to run the model on.
        """
        super().__init__()

        self.device = torch.device(device)
        self.net_u = net_u.to(self.device)
        self.adaptive_points = None  # Reserved for future adaptive sampling

        self.loss_fn = loss_fn if loss_fn is not None else nn.MSELoss()

    def net_u_forward(
        self, r: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the underlying network.

        Args:
            r: Dimensionless radius, shape (N, 1).
            t: Dimensionless time, shape (N, 1).

        Returns:
            Tuple (T, u) of dimensionless temperature and pore pressure.
        """
        inputs = torch.cat((r, t), dim=1)
        outputs = self.net_u(inputs)
        T = outputs[:, 0:1]
        u = outputs[:, 1:2]
        return T, u

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

        self.C1 = C1
        self.C2 = C2
        self.C3 = C3
        self.w_T = w_T
        self.w_u = w_u
        self.w_physical = w_physical

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
        self.net_u.eval()
        with torch.no_grad():
            r = r.to(self.device)
            t = t.to(self.device)
            T_pred, u_pred = self.net_u_forward(r, t)
            return T_pred.cpu().numpy(), u_pred.cpu().numpy()

    def evaluate(
        self,
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
