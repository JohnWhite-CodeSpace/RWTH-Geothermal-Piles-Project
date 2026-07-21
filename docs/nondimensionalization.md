# Nondimensionalization

## 1. Nondimensional variables

$$
\xi = \frac{r}{R_s}, \qquad
\tau = \frac{t}{t_E}, \qquad
\Theta = \frac{T - T_s}{T_f - T_s}, \qquad
U = \frac{u}{u_{\text{ref}}}
$$

where $t_E$ is the simulation end time (chosen once, fixed across all runs) and $u_{\text{ref}}$ is defined below.

Shorthand for the material groups appearing in Eq. (2):

$$
\bar\beta = (1-n)\beta_s + n\beta_w \ \ [1/°C], \qquad
S = n\alpha_w + \frac{1}{K_s} \ \ [1/\text{Pa}], \qquad
\lambda_w = \gamma_w = \rho_w g \approx 9810\ \text{Pa/m}
$$

## 2. Dimensionless groups

$$
\alpha_n = \frac{\alpha\, t_E}{R_s^2} \quad\text{(Fourier number — heat diffusion progress over the run)}
$$

$$
D = \frac{k}{\lambda_w\,\alpha\,S} \quad\text{(diffusivity ratio — material property only, independent of $t_E$)}
$$

$$
u_{\text{ref}} = \frac{\bar\beta\,(T_f - T_s)}{S} \quad\text{(undrained pressure scale: max pressure if $k\to0$, no dissipation)}
$$

With $u_{\text{ref}}$ chosen this way, the thermal source term in the pressure equation is scaled to exactly 1 — see derivation below.

## 3. Heat equation

Dimensional:


$$
\frac{\partial T}{\partial t} = \alpha\left(\frac{1}{r}\frac{\partial T}{\partial r} + \frac{\partial^2 T}{\partial r^2}\right)
$$

Substituting $\xi,\tau,\Theta$ and dividing by $(T_f-T_s)/t_E$:

$$
\boxed{\ \frac{\partial \Theta}{\partial \tau} = \alpha_n\left(\frac{1}{\xi}\frac{\partial \Theta}{\partial \xi} + \frac{\partial^2 \Theta}{\partial \xi^2}\right)\ }
$$

Structurally identical to the dimensional form — the heat equation is self-similar under this scaling.

## 4. Pore-pressure equation

Dimensional:

$$
-\bar\beta\,\frac{\partial T_s}{\partial t} + S\,\frac{\partial u}{\partial t} - \frac{k}{\lambda_w}\frac{1}{r}\frac{\partial}{\partial r}\!\left(r\frac{\partial u}{\partial r}\right) = 0
$$




Substituting $\xi,\tau,\Theta,U$ and multiplying through by $t_E/(S\,u_{\text{ref}})$:

$$
-\underbrace{\frac{\bar\beta(T_f-T_s)}{S\,u_{\text{ref}}}}_{=\ 1\ \text{(by choice of } u_{\text{ref}})}\frac{\partial \Theta}{\partial \tau} + \frac{\partial U}{\partial \tau} - \underbrace{\frac{k\,t_E}{\lambda_w R_s^2 S}}_{=\ D\,\alpha_n}\frac{1}{\xi}\frac{\partial}{\partial \xi}\!\left(\xi\frac{\partial U}{\partial \xi}\right) = 0
$$

$$
\boxed{\ \frac{\partial U}{\partial \tau} = D\,\alpha_n\left(\frac{1}{\xi}\frac{\partial U}{\partial \xi} + \frac{\partial^2 U}{\partial \xi^2}\right) + \frac{\partial \Theta}{\partial \tau}\ }
$$

**Consistency check:** the pressure equation's diffusion coefficient ($D\,\alpha_n$) is the heat equation's own diffusion coefficient ($\alpha_n$) scaled by the material diffusivity ratio $D$. One equation borrows the other's time normalization; $D$ carries all the $k$/$K_s$-dependence.

## 5. Boundary and initial conditions (nondimensional)

$$
\Theta(\xi = R/R_s,\ \tau) = 1, \qquad \Theta(\xi=1,\ \tau) = 0, \qquad \Theta(\tau=0) = 0
$$

$$
\left(\frac{\partial U}{\partial \xi}\right)_{\xi = R/R_s} = 0, \qquad U(\xi=1,\ \tau) = 0, \qquad U(\tau=0) = 0
$$
