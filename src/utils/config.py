from typing import Tuple
import torch

DEFAULT_KS = 2.0e6  

class PhysicsConstants:
    """Fixed soil/water physical properties for the geothermal pile PDEs."""

    def __init__(self):
        self.n = 0.25
        self.T_s = 12
        self.T_f = 50
        self.t_c = 1e7
        self.beta_p = 3.00e-5
        self.beta_w = 3.42e-4
        self.alpha_w = 5.00e-10
        self.rho_s = 2275.0
        self.gamma_soil = 2.0
        self.C_s = 1674.4
        self.gamma_w = 9810.0
        self.R_s = 30.0
        self.delta_T = 38.0
        self.alpha = self.gamma_soil / (self.rho_s * self.C_s)

    def calculate_physics_constants(
        self, k: float, Ks: float = DEFAULT_KS, t_c: float = 1e7
    ) -> Tuple[float, float, float, float]:
        A = (1 - self.n) * self.beta_p + self.n * self.beta_w
        B = (self.n * self.alpha_w) + (1 / Ks)
        C = k / self.gamma_w
        natural_uc = (A * self.delta_T) / B

        C1 = (self.alpha * t_c) / (self.R_s**2)
        C2 = (A * self.delta_T) / (B * natural_uc)
        C3 = (C * t_c) / (B * (self.R_s**2))

        return C1, C2, C3, natural_uc

    def calculate_physics_constants_tensor(
        self, 
        k_tensor: torch.Tensor, 
        Ks: float = DEFAULT_KS, 
        t_c: float = 1e7
    ) -> Tuple[float, float, torch.Tensor, float]:
        n = self.n
        beta_s = self.beta_p
        beta_w = self.beta_w
        alpha_w = self.alpha_w
        rho_s = self.rho_s
        C_s = self.C_s
        Gamma = self.gamma_soil
        gamma_w = self.gamma_w
        Rs = self.R_s
        delta_T = self.delta_T

        A = (1 - n) * beta_s + n * beta_w
        B = (n * alpha_w) + (1 / Ks)

        u_c = (A * delta_T) / B
        alpha = Gamma / (rho_s * C_s)

        C1 = (alpha * t_c) / (Rs ** 2)
        C2 = (A * delta_T) / (B * u_c)

        C = k_tensor / gamma_w
        C3_tensor = (C * t_c) / (B * (Rs ** 2))

        return C1, C2, C3_tensor, u_c