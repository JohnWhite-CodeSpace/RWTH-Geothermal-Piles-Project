from typing import Optional, Tuple
from src.utils.config import DEFAULT_KS


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
        self.case_permeabilities = {1: 1e-8, 2: 1e-9, 3: 1e-10, 4: 1e-11, 5: 1e-12}
        self.default_ks = 2e6

    def calculate_physics_constants(
        self,
        k: float,
        Ks: float = DEFAULT_KS,
        t_c: Optional[float] = None,
        u_c: Optional[float] = None,
    ) -> Tuple[float, float, float]:
        """
        Compute the nondimensional coefficients for the governing PDEs.

        Args:
            k: Soil permeability (m/s).
            Ks: Soil compressibility (Pa).
            t_c: Characteristic time scale for nondimensionalization (s).
                Defaults to this instance's `self.t_c`.
            u_c: Characteristic pressure scale for nondimensionalization
                (Pa). Defaults to this instance's `self.u_c`. Pass this
                explicitly only if you deliberately want a different
                scale than the rest of the project uses (e.g.
                `data_loader.py` always nondimensionalizes FDM data
                with this instance's own `self.t_c`/`self.u_c`).

        Returns:
            Tuple (C1, C2, C3): nondimensional coefficients for the heat
            equation and the coupled pore-pressure equation.
        """
        t_c = self.t_c if t_c is None else t_c
        u_c = self.u_c if u_c is None else u_c

        A = (1 - self.n) * self.beta_p + self.n * self.beta_w
        B = (self.n * self.alpha_w) + (1 / Ks)
        C = k / self.gamma_w

        C1 = (self.alpha * t_c) / (self.R_s**2)
        C2 = (A * self.delta_T) / (B * u_c)
        C3 = (C * t_c) / (B * (self.R_s**2))

        return C1, C2, C3
