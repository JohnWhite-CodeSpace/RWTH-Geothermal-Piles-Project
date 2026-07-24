# Governing Equations & Nondimensionalization

## 1. Dimensional Governing Equations

### Heat Equation

The heat conduction equation for constant thermal conductivity applies across the spatial domain $r \in [R, R_s]$:

$$
\frac{\partial T}{\partial t} = \alpha \frac{1}{r} \frac{\partial}{\partial r} \left( r \frac{\partial T}{\partial r} \right) = \alpha \left( \frac{1}{r} \frac{\partial T}{\partial r} + \frac{\partial^2 T}{\partial r^2} \right) \tag{1}
$$

where $\alpha = \frac{\Gamma}{\rho_s C_s}$ is the thermal diffusivity of the soil.

### Pore-Pressure Equation

Combining soil grain and water mass balance formulations yields the governing second-order parabolic PDE for excess pore pressure $u$ in the soil mass:

$$
-\left[(1-n)\beta_p + n\beta_w\right] \frac{\partial T}{\partial t} + \left(n\alpha_w + \frac{1}{K_s}\right) \frac{\partial u}{\partial t} - \frac{k}{\gamma_w} \frac{1}{r} \frac{\partial}{\partial r}\left(r \frac{\partial u}{\partial r}\right) = 0 \tag{2}
$$

where $\beta_p$ is the thermal expansion coefficient of the solid particles, $\beta_w$ is the thermal expansion coefficient of water, $\gamma_w = \rho_w \cdot g \approx 9810 \text{ Pa/m}$, $k$ is hydraulic permeability, $\alpha_w$ is water compressibility, and $K_s$ is the bulk modulus of soil grains.

### Dimensional Initial and Boundary Conditions

- **Pore Pressure Conditions:**

  $$
  \left(\frac{\partial u}{\partial r}\right)_{r=R} = 0, \qquad u(r=R_s, t) = 0 \quad \text{for all } t \ge 0 \tag{3}
  $$

  $$
  u(r, t=0) = 0 \quad \text{throughout the domain} \tag{4}
  $$

- **Temperature Conditions:**
  $$
  T(r=R, t) = T_f, \qquad T(r=R_s, t) = T_s \quad \text{for all } t \ge 0 \tag{5}
  $$
  $$
  T(r, t=0) = T_s \quad \text{throughout the domain}
  $$

---

## 2. Nondimensional Variables & Material Groups

Define the dimensionless space ($\xi$), time ($\tau$), temperature ($\Theta$), and pore pressure ($U$):

$$
\xi = \frac{r}{R_s}, \qquad
\tau = \frac{t}{t_E}, \qquad
\Theta = \frac{T - T_s}{T_f - T_s}, \qquad
U = \frac{u}{u_{\text{ref}}}
$$

where $t_E$ is the fixed total simulation timeframe ($15 \text{ days} = 1.296 \times 10^6 \text{ s}$).

Shorthand composite parameters:

$$
\bar{\beta} = (1-n)\beta_p + n\beta_w \ \ [1/^\circ\text{C}], \qquad
S = n\alpha_w + \frac{1}{K_s} \ \ [1/\text{Pa}]
$$

---

## 3. Dimensionless Scaling Parameters

$$
\alpha_n = \frac{\alpha\, t_E}{R_s^2} \quad \text{(Fourier number — temporal scale factor for heat diffusion)}
$$

$$
D = \frac{k}{\gamma_w\,\alpha\,S} \quad \text{(Diffusivity ratio — hydraulic vs. thermal dissipation speed)}
$$

$$
u_{\text{ref}} = \frac{\bar{\beta}\,(T_f - T_s)}{S} \quad \text{(Undrained pressure scale: theoretical max pressure when } k \to 0\text{)}
$$

---

## 4. Derivation of Non-Dimensional PDEs

### A. Non-Dimensional Heat Equation

Substituting $\xi, \tau, \Theta$ into Eq. (1) and dividing by $\frac{T_f - T_s}{t_E}$:

$$
\boxed{\ \frac{\partial \Theta}{\partial \tau} = \alpha_n \left( \frac{1}{\xi} \frac{\partial \Theta}{\partial \xi} + \frac{\partial^2 \Theta}{\partial \xi^2} \right)\ }
$$

### B. Non-Dimensional Excess Pore-Pressure Equation

Substituting $\xi, \tau, \Theta, U$ into Eq. (2) and multiplying through by $\frac{t_E}{S \cdot u_{\text{ref}}}$:

$$
-\underbrace{\frac{\bar{\beta}(T_f - T_s)}{S \cdot u_{\text{ref}}}}_{=\ 1} \frac{\partial \Theta}{\partial \tau} + \frac{\partial U}{\partial \tau} - \underbrace{\frac{k \cdot t_E}{\gamma_w R_s^2 S}}_{=\ D \cdot \alpha_n} \frac{1}{\xi} \frac{\partial}{\partial \xi}\left(\xi \frac{\partial U}{\partial \xi}\right) = 0
$$

Yielding the final operational PINN equation:

$$
\boxed{\ \frac{\partial U}{\partial \tau} = D \cdot \alpha_n \left( \frac{1}{\xi} \frac{\partial U}{\partial \xi} + \frac{\partial^2 U}{\partial \xi^2} \right) + \frac{\partial \Theta}{\partial \tau}\ }
$$

---

## 5. Non-Dimensional Initial and Boundary Conditions

$$
\Theta\left(\xi = \frac{R}{R_s},\ \tau\right) = 1, \qquad \Theta(\xi=1,\ \tau) = 0, \qquad \Theta(\xi,\ \tau=0) = 0
$$

$$
\left(\frac{\partial U}{\partial \xi}\right)_{\xi = R/R_s} = 0, \qquad U(\xi=1,\ \tau) = 0, \qquad U(\xi,\ \tau=0) = 0
$$
