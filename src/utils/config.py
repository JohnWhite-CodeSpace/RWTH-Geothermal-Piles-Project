from typing import Tuple


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
        self.u_c = 8e5
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
        self, k: float, Ks: float = 2e6, t_c: float = 1e7, u_c: float = 1e6
    ) -> Tuple[float, float, float]:
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

        C1 = (self.alpha * t_c) / (self.R_s**2)
        C2 = (A * self.delta_T) / (B * u_c)
        C3 = (C * t_c) / (B * (self.R_s**2))

        return C1, C2, C3
