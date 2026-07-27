<<<<<<< Updated upstream
# Permeability (m/s) associated with each FDM reference case (Table 1).
CASE_PERMEABILITIES = {1: 1e-8, 2: 1e-9, 3: 1e-10, 4: 1e-11, 5: 1e-12}

# Default soil compressibility (Pa), the first Ks value in Table 1.
DEFAULT_KS = 2e6
=======
from typing import Tuple
import torch



class PhysicsConstants:
    """Fixed soil/water physical properties for the geothermal pile PDEs.

    A single instance can be reused across all cases in the parametric
    study (Table 1), since only permeability, soil compressibility, and
    the characteristic scales change between cases.
    """

    def __init__(self):
        """Initialize soil and water physical properties."""
        self.n = 0.25  # Porosity
        self.T_s = 12
        self.T_f = 50
        self.t_c = 1e7
        self.beta_p = 3.00e-5  # Thermal expansion, soil particles (1/C)
        self.beta_w = 3.42e-4  # Thermal expansion, water (1/C)
        self.alpha_w = 5.00e-10  # Compressibility, water (1/Pa)
        self.rho_s = 2275.0  # Density, soil medium (kg/m3)
        self.gamma_soil = 2.0  # Thermal conductivity, soil medium (W/mC)
        self.C_s = 1674.4  # Specific heat, soil medium (J/kgC)
        self.gamma_w = 9810.0  # Unit weight of water (N/m3)
        self.R_s = 30.0  # Far-field radius (m)
        self.delta_T = 38.0  # Tf - Ts, Tf=50C case (C)
        self.alpha = self.gamma_soil / (
            self.rho_s * self.C_s
        )  # Thermal diffusivity (m2/s)

    def calculate_physics_constants(
        self, k: float, Ks: float = 2e6, t_c: float = 1e7
    ) -> Tuple[float, float, float, float]:
        """
        Compute the nondimensional coefficients for the governing PDEs.

        Args:
            k: Soil permeability (m/s).
            Ks: Soil compressibility (Pa).
            t_c: Characteristic time scale for nondimensionalization (s).
            u_c: Characteristic pressure scale for nondimensionalization (Pa).

        Returns:
            Tuple (C1, C2, C3): nondimensional coefficients for the heat
            equation and the coupled pore-pressure equation.
        """
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
        Ks: float, 
        t_c: float
    ):
        """
        Dynamically calculate non-dimensional physics constants using PyTorch tensors.
        This is REQUIRED for the Inverse PINN so that gradients can flow from 
        the PDE loss back to the learnable permeability parameter (k).

        Args:
            k_tensor: Permeability as a PyTorch Tensor with requires_grad=True.
            Ks: Soil compressibility (float).
            t_c: Characteristic time (float).

        Returns:
            C1 (float), C2 (float), C3 (torch.Tensor), u_c (float)
        """
        
        # Material constants
        n = 0.25
        beta_s = 3.00e-5
        beta_w = 3.42e-4
        alpha_w = 5.00e-10
        rho_s = 2275.0
        C_s = 1674.4
        Gamma = 2.0
        gamma_w = 1000.0 * 9.81
        Rs = 30.0
        delta_T = 38.0

        # Thermo-elastic parameters
        beta_bar = (1 - n) * beta_s + n * beta_w
        S = n * alpha_w + 1 / Ks

        # Reference scaling (u_c)
        u_c = (beta_bar * delta_T) / S

        # Thermal diffusivity (alpha)
        alpha = Gamma / (rho_s * C_s)

        # C1 and C2 are constants (independent of k)
        C1 = (alpha * t_c) / (Rs ** 2)
        C2 = 1.0  

        # =============================================================
        # C3 MUST BE A TENSOR
        # All operations here involve k_tensor, keeping the graph alive
        # =============================================================
        c_v_tensor = k_tensor / (gamma_w * S)
        D_tensor = c_v_tensor / alpha
        C3_tensor = D_tensor * C1

        return C1, C2, C3_tensor, u_c
>>>>>>> Stashed changes
