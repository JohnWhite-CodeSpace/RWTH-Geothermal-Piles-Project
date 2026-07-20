# Physics-Informed Neural Networks Theory

This document outlines the theoretical foundation of PINNs applied to geothermal pile analysis.

## Overview

Physics-Informed Neural Networks (PINNs) are neural networks trained to solve supervised learning tasks while respecting any given law of physics described by partial differential equations (PDEs).

## Governing Equations

### Heat Equation

For constant thermal conductivity in both pile and soil:

$$\frac{\partial T}{\partial t} = \alpha \frac{1}{r}\frac{\partial}{\partial r}\left(r\frac{\partial T}{\partial r}\right) = \alpha\left(\frac{\partial T}{\partial r} + \frac{\partial^2 T}{\partial r^2}\right)$$

where:
- $T$ = temperature
- $\alpha$ = thermal diffusivity
- $r$ = radial distance
- $t$ = time

### Coupled Hydro-Thermal Equation

For soil mass balance combining soil and water:

$$-[(1-n)\beta_s + \beta_w n]\frac{\partial T_s}{\partial t} + \left(n\omega_w + \frac{1}{K_s}\right)\frac{\partial u}{\partial t} - \lambda_w \frac{1}{r}\frac{\partial}{\partial r}\left(r\frac{\partial u}{\partial r}\right) = 0$$

where:
- $u$ = excess pore-water pressure
- $n$ = porosity
- $\beta_s$, $\beta_w$ = thermal expansion coefficients
- $K_s$ = soil compressibility
- $\lambda_w$ = water permeability-related parameter

## Boundary Conditions

1. **At pile-soil interface** ($r = 0.5$ m):
   - Constant temperature: $T(r=0.5) = T_f$
   
2. **At far field** ($r = R_s$):
   - Natural temperature: $T(r=R_s) = T_s$
   - Zero pressure gradient: $\frac{\partial u}{\partial r} = 0$
   
3. **Initial conditions**:
   - $u(t=0) = 0$ (hydrostatic equilibrium)
   - $T(t=0) = T_s$ (initial ground temperature)

## PINN Architecture

### Network Design

- **Input:** Spatial location $(r)$ and time $(t)$
- **Hidden layers:** Multiple fully-connected layers with activation functions
- **Output:** Temperature $T$ and excess pore pressure $u$

### Loss Function

The total loss is a weighted combination of:

$$\mathcal{L} = \lambda_{\text{physics}} \mathcal{L}_{\text{physics}} + \lambda_{\text{data}} \mathcal{L}_{\text{data}} + \lambda_{\text{bc}} \mathcal{L}_{\text{bc}}$$

where:

1. **Physics loss** (PDE residuals):
$$\mathcal{L}_{\text{physics}} = \frac{1}{N_p} \sum_{i=1}^{N_p} \left|\frac{\partial T}{\partial t} - \alpha\nabla^2 T\right|^2 + \left|\text{Hydro-thermal equation residuals}\right|^2$$

2. **Data loss** (FDM reference solutions):
$$\mathcal{L}_{\text{data}} = \frac{1}{N_d} \sum_{i=1}^{N_d} \left(T_i^{\text{pred}} - T_i^{\text{FDM}}\right)^2 + \left(u_i^{\text{pred}} - u_i^{\text{FDM}}\right)^2$$

3. **Boundary condition loss**:
$$\mathcal{L}_{\text{bc}} = \frac{1}{N_b} \sum_{i=1}^{N_b} \left(T_i^{\text{pred}}(r=0.5) - T_f\right)^2 + \left|\frac{\partial u}{\partial r}\bigg|_{r=R_s}\right|^2$$

## Training Strategy

### Forward PINN Model

- **Objective:** Learn temperature and pressure fields given boundary conditions
- **Training data:** FDM reference solutions + boundary conditions
- **Loss:** Combination of physics and data losses

### Inverse PINN Model

- **Objective:** Estimate soil permeability $k$ from observed data
- **Input:** Observed temperature and pressure fields
- **Output:** Soil parameter estimation

### Parametric PINN Model

- **Objective:** Parametrize solution with respect to input permeability
- **Input:** Spatial coordinates and permeability value
- **Output:** Temperature and pressure for any $k$ value

## Reference Solutions (FDM)

Finite Difference Method (FDM) solutions serve as reference data for:
- Training data points
- Validation and testing
- Error analysis and benchmarking

## Key Publications

1. Fuentes, R., Piñol, N., & Alonso, E. (2016). "Effect of temperature induced excess porewater pressures on the shaft bearing capacity of geothermal piles". *Geomechanics for Energy and the Environment*, 8:30-37.

2. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations". *Journal of Computational Physics*, 378:686-707.

---

See also:
- [README.md](../README.md) - Project overview
- [setup.md](setup.md) - Setup instructions
- [api.md](api.md) - Code API documentation

Last updated: 2026-07-20
