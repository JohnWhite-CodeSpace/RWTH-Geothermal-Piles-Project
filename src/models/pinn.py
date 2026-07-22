from typing import Optional, Tuple, Type

import torch
import torch.nn as nn


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
        return self.loss_fn(pde_residual, torch.zeros_like(pde_residual))

    def loss_ic(self, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute the initial condition loss (T*=0, u*=0 at t=0)."""
        T, u = self.net_u_forward(r, t)
        ic_loss_T = self.loss_fn(T, torch.zeros_like(T))
        ic_loss_u = self.loss_fn(u, torch.zeros_like(u))
        return ic_loss_T + ic_loss_u

    def loss_bc_pile(self, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute the pile boundary loss (r*=0.0167): T*=1, du/dr=0."""
        T, u = self.net_u_forward(r, t)

        bc_loss_T = self.loss_fn(T, torch.ones_like(T))

        u_r = torch.autograd.grad(u, r, torch.ones_like(u), create_graph=True)[0]
        bc_loss_u = self.loss_fn(u_r, torch.zeros_like(u_r))

        return bc_loss_T + bc_loss_u

    def loss_bc_far(self, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute the far-field boundary loss (r*=1.0): T*=0, u*=0."""
        T, u = self.net_u_forward(r, t)

        bc_loss_T = self.loss_fn(T, torch.zeros_like(T))
        bc_loss_u = self.loss_fn(u, torch.zeros_like(u))

        return bc_loss_T + bc_loss_u

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

        total_loss = pde_loss + ic_loss + bc_pile_loss + bc_far_loss
        total_loss.backward()

        if self.iter % self.log_interval == 0:
            print(
                f"[{self.optimizer_name}]: Iteration {self.iter}, "
                f"Total loss: {total_loss.item():.3e}, "
                f"PDE Loss: {pde_loss.item():.3e}, "
                f"IC Loss: {ic_loss.item():.3e}, "
                f"Pile BC Loss: {bc_pile_loss.item():.3e}, "
                f"Far BC Loss: {bc_far_loss.item():.3e}"
            )
        self.iter += 1
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
    ) -> None:
        """
        Train the network on the PDE, initial condition, and boundary losses.

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
        """
        self.domain_points = domain_points
        self.ic_points = ic_points
        self.bc_pile_points = bc_pile_points
        self.bc_far_points = bc_far_points

        self.C1 = C1
        self.C2 = C2
        self.C3 = C3

        self.net_u.train()
        self.iter = 0
        self.log_interval = log_interval

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

    def evaluate(self):
        """Evaluate predictions against FDM reference data."""
        raise NotImplementedError
