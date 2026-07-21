import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def calculate_physics_constants(k, Ks=2e6, t_c=1e7, u_c=1e6):
  n = 0.25                  # Porosity
  beta_s = 3.00e-5          # Thermal expansion soil (1/C)
  beta_w = 3.42e-4          # Thermal expansion water (1/C)
  alpha_w = 5.00e-10        # Compressibility water (1/Pa)
  rho_s = 2275.0            # Density soil (kg/m3)
  Gamma = 2.0               # Thermal conductivity (W/mC)
  C_s = 1674.4              # Specific heat soil (J/kgC)
  gamma_w = 9810.0          # Unit weight of water (N/m3)
  R_s = 30.0                
  delta_T = 38.0      
  alpha = Gamma / (rho_s * C_s)
    
  A = (1 - n) * beta_s + n * beta_w
  B = (n * alpha_w) + (1 / Ks)
  C = k / gamma_w
    
  C1 = (alpha * t_c) / (R_s ** 2)
  C2 = (A * delta_T) / (B * u_c)
  C3 = (C * t_c) / (B * (R_s ** 2))
    
  return C1, C2, C3

class Geothermal_PINN(nn.Module):
    def __init__(self, net_u, loss_fn=torch.nn.MSELoss(), device='cpu'):
        super(Geothermal_PINN, self).__init__()
        self.net_u = net_u.to(device)
        self.adaptive_points = None  # Initialize adaptive points

        self.loss_fn = loss_fn
        self.device = device
 # Forward pass: inputs are radius (r) and time (t)
    def net_u_forward(self, r, t):
        inputs = torch.cat((r, t), dim=1)
        outputs= self.net_u(inputs)
        T = outputs[:, 0:1]
        u = outputs[:, 1:2]
        return T, u
    
    def pde_residual(self, r, t):
        T, u = self.net_u_forward(r, t)
        # 1. Gradients for Temperature
        T_r = torch.autograd.grad(T, r, torch.ones_like(T), create_graph=True)[0]
        T_rr = torch.autograd.grad(T_r, r, torch.ones_like(T_r), create_graph=True)[0]
        T_t = torch.autograd.grad(T, t, torch.ones_like(T), create_graph=True)[0]

        # 2. Gradients for Pore-water Pressure
        u_r = torch.autograd.grad(u, r, torch.ones_like(u), create_graph=True)[0]
        u_rr = torch.autograd.grad(u_r, r, torch.ones_like(u_r), create_graph=True)[0]
        u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]

        # 3. Dimensionless Governing Equations
        pde_residual_T = T_t - self.C1 * ((1.0 / r) * T_r + T_rr)
        pde_residual_u = u_t - self.C2 * T_t - self.C3 * ((1.0 / r) * u_r + u_rr)
        pde_residual = torch.cat((pde_residual_T,  pde_residual_u), dim=1)
        return pde_residual
      
    
    # Calculates the loss for the PDE
    def loss_pde(self, r, t):
        pde_residual = self.pde_residual(r, t)
        pde_loss = self.loss_fn(pde_residual, torch.zeros_like(pde_residual))
        return pde_loss
    
    # Calculates the loss for the initial condition (T=12,T*=0,u=0,u*=0)
    def loss_ic(self, r, t):
        T, u = self.net_u_forward(r, t)
        ic_loss_T = self.loss_fn(T, torch.zeros_like(T))
        ic_loss_u = self.loss_fn(u, torch.zeros_like(u))
        return ic_loss_T + ic_loss_u
       
 # Calculates the loss for the boundary condition at the Pile (r* = 0.0167)
    def loss_bc_pile(self, r, t):
        T, u = self.net_u_forward(r, t)
        
        # T*=1.0 (50°C) at the pile
        bc_loss_T = self.loss_fn(T, torch.ones_like(T))
        
        # Impermeable pile: du/dr = 0
        u_r = torch.autograd.grad(u, r, torch.ones_like(u), create_graph=True)[0]
        bc_loss_u = self.loss_fn(u_r, torch.zeros_like(u_r))
        
        return bc_loss_T + bc_loss_u
    
  # Calculates the loss for the boundary condition Far Away (r* = 1.0)
    def loss_bc_far(self, r, t):
        T, u = self.net_u_forward(r, t)
        
        # T*=0.0 (12°C) and u*=0.0 (0 Pa) far away
        bc_loss_T = self.loss_fn(T, torch.zeros_like(T))
        bc_loss_u = self.loss_fn(u, torch.zeros_like(u))
        
        return bc_loss_T + bc_loss_u
    
    def pinn_loss(self):
        self.optimizer.zero_grad()

        
   # Extract coordinates and require gradients
        r_dom = self.domain_points[:, 0:1].to(self.device).requires_grad_(True)
        t_dom = self.domain_points[:, 1:2].to(self.device).requires_grad_(True)
        
        r_ic = self.ic_points[:, 0:1].to(self.device).requires_grad_(True)
        t_ic = self.ic_points[:, 1:2].to(self.device).requires_grad_(True)
        
        r_bc_pile = self.bc_pile_points[:, 0:1].to(self.device).requires_grad_(True)
        t_bc_pile = self.bc_pile_points[:, 1:2].to(self.device).requires_grad_(True)
        
        r_bc_far = self.bc_far_points[:, 0:1].to(self.device).requires_grad_(True)
        t_bc_far = self.bc_far_points[:, 1:2].to(self.device).requires_grad_(True)

    # Calculate losses
        pde_loss = self.loss_pde(r_dom, t_dom)
        ic_loss = self.loss_ic(r_ic, t_ic)
        bc_pile_loss = self.loss_bc_pile(r_bc_pile, t_bc_pile)
        bc_far_loss = self.loss_bc_far(r_bc_far, t_bc_far)

        total_loss = pde_loss + ic_loss + bc_pile_loss + bc_far_loss
        total_loss.backward() # backpropagate the total loss

    # log the losses 
        if self.iter % self.log_interval == 0:
            print(f'[{self.optimizer_name}]: Iteration {self.iter}, Total loss: {total_loss.item():.3e}, PDE Loss: {pde_loss.item():.3e}, IC Loss: {ic_loss.item():.3e}, Pile BC Loss: {bc_pile_loss.item():.3e}, Far BC Loss: {bc_far_loss.item():.3e}')
        self.iter += 1
        return total_loss

    def train_net(self, domain_points, ic_points, bc_pile_points, bc_far_points, C1, C2, C3, epochs, optimizer=torch.optim.Adam, log_interval=100):
        self.epochs = epochs
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
            self.optimizer_name = 'LBFGS'
            self.optimizer = optimizer(self.net_u.parameters(), lr=1, max_iter=epochs, max_eval=None, history_size=100, tolerance_grad=1e-8, tolerance_change=1e-8, line_search_fn='strong_wolfe')
            self.optimizer.step(self.pinn_loss)
        else:
            self.optimizer = optimizer(self.net_u.parameters(), lr=1e-3)
            self.optimizer_name = 'Adam'
            for epoch in range(epochs):
                self.pinn_loss()
                self.optimizer.step()

    def predict(self, r, t):
        self.net_u.eval()
        with torch.no_grad():
            r = r.to(self.device)
            t = t.to(self.device)
            T_pred, u_pred = self.net_u_forward(r, t)
            return T_pred.cpu().numpy(), u_pred.cpu().numpy()

  
    
    def evaluate(self):
        pass



