import os
import time
from scipy.stats.qmc import LatinHypercube
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# =========================================================
# 0. Global Setup, Reproducibility & Physical Parameters
# =========================================================

torch.set_default_dtype(torch.float32)
SEED = 1235
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# ---------------- Physical Parameters (Geothermal Energy Pile Domain) ----------------
R = 0.5
Rs = 30.0
Ts = 12.0
Tf = 50.0

n = 0.25
beta_s = 3.00e-5
beta_w = 3.42e-4
alpha_w = 5.00e-10
rho_s = 2275.0
Gamma = 2.0
C_s = 1674.4
gamma_w = 1000.0 * 9.81
t_E = 15 * 24 * 3600.0   # keep consistent with whatever the case CSVs' time horizon is

alpha = Gamma / (rho_s * C_s)
beta_bar = (1 - n) * beta_s + n * beta_w

xi_R = R / Rs
alpha_n = alpha * t_E / Rs**2   # depends only on alpha & t_E -> same for every case

print(f'alpha    = {alpha:.4e} m2/s')
print(f'beta_bar = {beta_bar:.4e} 1/C')
print(f'xi_R     = {xi_R:.5f} | alpha_n = {alpha_n:.5e}')

# =========================================================
# 1. Cases to run
# =========================================================

BASE_DIR = '/Users/seanxie/Desktop/RWTH PINN/Geothermal Piles Updated'
OUTPUT_DIR = os.path.join(BASE_DIR, 'pinn_task2_results_5cases')


CASES = [
    ('case1', 1e-8, 2e6),
    ('case2', 1e-9, 2e6),
    ('case3', 1e-10, 2e6),
    ('case4', 1e-11, 2e6),
    ('case5', 1e-12, 2e6),
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================================
# 2. Network Architecture & Latin Hypercube Sampler (shared across all cases)
# =========================================================

class MLP(nn.Module):
    def __init__(self, arch, act):
        super().__init__()
        layers = []
        for i in range(len(arch) - 1):
            layers.append(nn.Linear(arch[i], arch[i + 1]))
            if i < len(arch) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def lhs_sample(n_pts, spans):
    lhs = LatinHypercube(d=len(spans))
    s = lhs.random(n=n_pts)
    for i, span in enumerate(spans):
        s[:, i] = span[0] + (span[1] - span[0]) * s[:, i]
    return torch.tensor(s, dtype=torch.float32)


def lhs_sample_log_xi(n_pts, xi_min, xi_max, tau_min, tau_max):
    """
    LHS in (log(xi), tau) space, exponentiated back to xi. Concentrates points
    near xi_min: r stays within a couple of pile-radii of R for the whole
    simulation horizon (thermal/pressure diffusion length is only ~1-2m over
    tens of days), so a uniform-in-xi sampler wastes most of its budget on the
    far field where both fields are essentially flat/zero. Empirically, this
    puts ~50% of points within r<3.8m vs ~11% for uniform-in-xi sampling.
    """
    spans = [(np.log(xi_min), np.log(xi_max)), (tau_min, tau_max)]
    raw = lhs_sample(n_pts, spans)
    xi = torch.exp(raw[:, 0:1])
    tau = raw[:, 1:2]
    return xi, tau


N_DOM, N_IC, N_BC = 3000, 300, 300

xi_dom_raw, tau_dom_raw = lhs_sample_log_xi(N_DOM, xi_R, 1.0, 0.0, 1.0)
xi_dom = xi_dom_raw.clone().to(device).requires_grad_(True)
tau_dom = tau_dom_raw.clone().to(device).requires_grad_(True)

# log-spaced IC points for the same reason (denser near the pile, where the
# field actually varies with r at tau=0+)
xi_ic = torch.logspace(np.log10(xi_R), 0.0, N_IC).reshape(-1, 1).to(device)
tau_ic = torch.zeros_like(xi_ic).to(device)

tau_bc = torch.linspace(0.0, 1.0, N_BC).reshape(-1, 1).to(device)
xi_bcL = torch.full_like(tau_bc, xi_R).to(device).requires_grad_(True)
tau_bcL = tau_bc.clone().to(device).requires_grad_(True)
xi_bcR = torch.full_like(tau_bc, 1.0).to(device)
tau_bcR = tau_bc.clone().to(device)

mse = nn.MSELoss()

# =========================================================
# 3. Stage 1: Temperature PINN -- trained ONCE (doesn't depend on k or Ks)
# =========================================================

class PINN_Theta(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, xi, tau):
        return self.net(torch.cat([xi, tau], dim=1))

    def pde_residual(self, xi, tau):
        Th = self.forward(xi, tau)
        Th_xi = torch.autograd.grad(Th, xi, torch.ones_like(Th), create_graph=True)[0]
        Th_xixi = torch.autograd.grad(Th_xi, xi, torch.ones_like(Th_xi), create_graph=True)[0]
        Th_tau = torch.autograd.grad(Th, tau, torch.ones_like(Th), create_graph=True)[0]
        return Th_tau - alpha_n * (Th_xi / xi + Th_xixi)


net_T = MLP([2] + [64] * 3 + [1], nn.Tanh).to(device)
model_T = PINN_Theta(net_T).to(device)

opt_T = torch.optim.Adam(model_T.parameters(), lr=2e-3)
sched_T = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_T, factor=0.7, patience=300)

print('\n--- Stage 1A: Training Temperature PINN with Adam (once, shared by all cases) ---')
t0 = time.time()
EPOCHS_T = 4000
for ep in range(EPOCHS_T):
    opt_T.zero_grad()
    res = model_T.pde_residual(xi_dom, tau_dom)
    loss_pde = mse(res, torch.zeros_like(res))
    loss_ic = mse(model_T(xi_ic, tau_ic), torch.zeros_like(xi_ic))
    loss_bcL = mse(model_T(xi_bcL, tau_bcL), torch.ones_like(xi_bcL))
    loss_bcR = mse(model_T(xi_bcR, tau_bcR), torch.zeros_like(xi_bcR))
    loss = loss_pde + 10 * loss_ic + 10 * loss_bcL + 10 * loss_bcR
    loss.backward()
    opt_T.step()
    sched_T.step(loss.item())
    if ep % 1000 == 0 or ep == EPOCHS_T - 1:
        print(f'[T-Adam] ep {ep:5d} | loss={loss.item():.3e} | pde={loss_pde.item():.3e}')

print('--- Stage 1B: Fine-tuning Temperature PINN with L-BFGS ---')
opt_lbfgs_T = torch.optim.LBFGS(model_T.parameters(), lr=0.5, max_iter=10000, history_size=100,
                                tolerance_grad=1e-9, tolerance_change=1e-11, line_search_fn='strong_wolfe')
iter_T = [0]
def closure_T():
    opt_lbfgs_T.zero_grad()
    res = model_T.pde_residual(xi_dom, tau_dom)
    loss_pde = mse(res, torch.zeros_like(res))
    loss_ic = mse(model_T(xi_ic, tau_ic), torch.zeros_like(xi_ic))
    loss_bcL = mse(model_T(xi_bcL, tau_bcL), torch.ones_like(xi_bcL))
    loss_bcR = mse(model_T(xi_bcR, tau_bcR), torch.zeros_like(xi_bcR))
    loss = loss_pde + 10 * loss_ic + 10 * loss_bcL + 10 * loss_bcR
    loss.backward()
    if iter_T[0] % 50 == 0:
        print(f'[T-LBFGS] iter {iter_T[0]:4d} | loss={loss.item():.3e} | pde={loss_pde.item():.3e}')
    iter_T[0] += 1
    return loss

opt_lbfgs_T.step(closure_T)
print(f'Temperature training time: {time.time() - t0:.2f} s')

for p in model_T.parameters():
    p.requires_grad_(False)
model_T.eval()


# =========================================================
# 4. Stage 2: Pore Pressure PINN -- fresh network per case, D/alpha_n stored
#    as attributes (not globals) so nothing goes stale across loop iterations
# =========================================================

class PINN_U(nn.Module):
    def __init__(self, net, theta_model, D, alpha_n):
        super().__init__()
        self.net = net
        self.theta_model = theta_model
        self.D = D
        self.alpha_n = alpha_n

    def forward(self, xi, tau):
        return self.net(torch.cat([xi, tau], dim=1))

    def theta_tau(self, xi, tau):
        xi_ = xi.detach().clone().requires_grad_(True)
        tau_ = tau.detach().clone().requires_grad_(True)
        Th = self.theta_model(xi_, tau_)
        Th_tau = torch.autograd.grad(Th, tau_, torch.ones_like(Th), create_graph=True)[0]
        return Th_tau.detach()

    def pde_residual(self, xi, tau):
        U = self.forward(xi, tau)
        U_xi = torch.autograd.grad(U, xi, torch.ones_like(U), create_graph=True)[0]
        U_xixi = torch.autograd.grad(U_xi, xi, torch.ones_like(U_xi), create_graph=True)[0]
        U_tau = torch.autograd.grad(U, tau, torch.ones_like(U), create_graph=True)[0]
        src = self.theta_tau(xi, tau)
        return U_tau - self.D * self.alpha_n * (U_xi / xi + U_xixi) - src

    def neumann_bc(self, xi, tau):
        U = self.forward(xi, tau)
        return torch.autograd.grad(U, xi, torch.ones_like(U), create_graph=True)[0]


def plot_daily_lines(r_ref, t_days, ref_data, pred_data, ylabel, title, savepath):
    """
    One plot with every discrete day in the CSV overlaid: reference (solid) vs
    PINN (dashed), colored by day via a colorbar. This shows the actual profile
    shape at each sampled day directly, which the heatmap-style comparison can
    obscure (a 2D color map hides small offsets that are obvious on a line plot).
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    norm = mcolors.Normalize(vmin=t_days.min(), vmax=t_days.max())
    cmap = cm.viridis
    for i, td in enumerate(t_days):
        color = cmap(norm(td))
        ax.plot(r_ref, ref_data[i], '-', color=color, lw=1.6)
        ax.plot(r_ref, pred_data[i], '--', color=color, lw=1.6)
    ax.plot([], [], 'k-', label='Reference (FDM)')
    ax.plot([], [], 'k--', label='PINN')
    ax.legend(loc='upper right', framealpha=0.9)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Time (days)')
    ax.set_xlabel('Radius r (m)')
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight='bold')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(savepath, dpi=300, bbox_inches='tight')
    plt.close(fig)


EPOCHS_U = 4000
summary = []

for case_name, k_perm, Ks in CASES:
    S = n * alpha_w + 1 / Ks
    u_ref = beta_bar * (Tf - Ts) / S
    c_v = k_perm / (gamma_w * S)
    D = c_v / alpha

    print(f'\n================ {case_name}  (k={k_perm:.1e}, Ks={Ks:.1e}) ================')
    print(f'S={S:.4e} 1/Pa | u_ref={u_ref:.4e} Pa | D={D:.4f} | D*alpha_n={D*alpha_n:.4e}')

    net_U = MLP([2] + [64] * 3 + [1], nn.Tanh).to(device)
    model_U = PINN_U(net_U, model_T, D, alpha_n).to(device)

    opt_U = torch.optim.Adam(model_U.parameters(), lr=2e-3)
    sched_U = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_U, factor=0.7, patience=300)

    print(f'--- {case_name} Stage 2A: Adam ---')
    t0 = time.time()
    for ep in range(EPOCHS_U):
        opt_U.zero_grad()
        res = model_U.pde_residual(xi_dom, tau_dom)
        loss_pde = mse(res, torch.zeros_like(res))
        loss_ic = mse(model_U(xi_ic, tau_ic), torch.zeros_like(xi_ic))
        bc_neu = model_U.neumann_bc(xi_bcL, tau_bcL)
        loss_bcL = mse(bc_neu, torch.zeros_like(bc_neu))
        loss_bcR = mse(model_U(xi_bcR, tau_bcR), torch.zeros_like(xi_bcR))
        loss = loss_pde + 10 * loss_ic + 10 * loss_bcL + 10 * loss_bcR
        loss.backward()
        opt_U.step()
        sched_U.step(loss.item())
        if ep % 1000 == 0 or ep == EPOCHS_U - 1:
            print(f'[{case_name} U-Adam] ep {ep:5d} | loss={loss.item():.3e} | pde={loss_pde.item():.3e}')

    print(f'--- {case_name} Stage 2B: L-BFGS ---')
    opt_lbfgs_U = torch.optim.LBFGS(model_U.parameters(), lr=0.5, max_iter=100000, history_size=100,
                                     tolerance_grad=1e-9, tolerance_change=1e-11, line_search_fn='strong_wolfe')
    iter_U = [0]
    def closure_U():
        opt_lbfgs_U.zero_grad()
        res = model_U.pde_residual(xi_dom, tau_dom)
        loss_pde = mse(res, torch.zeros_like(res))
        loss_ic = mse(model_U(xi_ic, tau_ic), torch.zeros_like(xi_ic))
        bc_neu = model_U.neumann_bc(xi_bcL, tau_bcL)
        loss_bcL = mse(bc_neu, torch.zeros_like(bc_neu))
        loss_bcR = mse(model_U(xi_bcR, tau_bcR), torch.zeros_like(xi_bcR))
        loss = loss_pde + 10 * loss_ic + 10 * loss_bcL + 10 * loss_bcR
        loss.backward()
        if iter_U[0] % 50 == 0:
            print(f'[{case_name} U-LBFGS] iter {iter_U[0]:4d} | loss={loss.item():.3e} | pde={loss_pde.item():.3e}')
        iter_U[0] += 1
        return loss

    opt_lbfgs_U.step(closure_U)
    print(f'{case_name} pressure training time: {time.time() - t0:.2f} s')

    for p in model_U.parameters():
        p.requires_grad_(False)
    model_U.eval()

    torch.save({'net_T': net_T.state_dict(), 'net_U': net_U.state_dict(),
                'alpha_n': alpha_n, 'D': D, 'xi_R': xi_R, 'k_perm': k_perm, 'Ks': Ks},
               os.path.join(OUTPUT_DIR, f'{case_name}_k{k_perm:.0e}_weights.pt'))

    # ---------------- Evaluate against this case's own CSVs ----------------
    path_u_csv = os.path.join(BASE_DIR, case_name, 'excess_pore_pressure.csv')
    path_T_csv = os.path.join(BASE_DIR, case_name, 'temperature_distribution.csv')

    try:
        df_u = pd.read_csv(path_u_csv)
        df_T = pd.read_csv(path_T_csv)

        r_ref = df_T.columns[1:].astype(float).values
        t_days = df_T['Time (days)'].values
        T_ref = df_T.iloc[:, 1:].values
        u_ref_data = df_u.iloc[:, 1:].values   # assumed kPa, matching the case3 CSV convention

        xi_grid = torch.tensor(r_ref / Rs, dtype=torch.float32)
        tau_grid = torch.tensor((t_days * 86400.0) / t_E, dtype=torch.float32)
        XI, TAU = torch.meshgrid(xi_grid, tau_grid, indexing='ij')
        xi_eval = XI.reshape(-1, 1).to(device)
        tau_eval = TAU.reshape(-1, 1).to(device)

        with torch.no_grad():
            Theta_pred = model_T(xi_eval, tau_eval).cpu().numpy().reshape(len(r_ref), len(t_days))
            U_pred = model_U(xi_eval, tau_eval).cpu().numpy().reshape(len(r_ref), len(t_days))

        T_pred = (Ts + Theta_pred * (Tf - Ts)).T
        u_pred = (U_pred * u_ref).T   

        err_L2_T = np.linalg.norm(T_ref - T_pred) / np.linalg.norm(T_ref)
        err_L2_u = np.linalg.norm(u_ref_data - u_pred) / np.linalg.norm(u_ref_data)

        print(f'{case_name}: Temperature relL2={err_L2_T*100:.2f}%  Pressure relL2={err_L2_u*100:.2f}%')
        summary.append({'case': case_name, 'k': k_perm, 'Ks': Ks,
                         'T_relL2_pct': err_L2_T * 100, 'u_relL2_pct': err_L2_u * 100})

        tag = f'{case_name}_k{k_perm:.0e}_K{Ks:.0e}'
        extent = [t_days[0], t_days[-1], r_ref[0], r_ref[-1]]

        fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
        im0 = axes[0].imshow(T_pred.T, aspect='auto', origin='lower', extent=extent, cmap='plasma')
        axes[0].set_title('PINN Temperature (°C)', fontweight='bold'); axes[0].set_xlabel('Time (days)'); axes[0].set_ylabel('Radius r (m)')
        fig.colorbar(im0, ax=axes[0])
        im1 = axes[1].imshow(T_ref.T, aspect='auto', origin='lower', extent=extent, cmap='plasma')
        axes[1].set_title('Reference Temperature (°C)', fontweight='bold'); axes[1].set_xlabel('Time (days)')
        fig.colorbar(im1, ax=axes[1])
        im2 = axes[2].imshow(np.abs(T_pred - T_ref).T, aspect='auto', origin='lower', extent=extent, cmap='inferno')
        axes[2].set_title(f'Absolute Error (Rel L2: {err_L2_T*100:.2f}%)', fontweight='bold'); axes[2].set_xlabel('Time (days)')
        fig.colorbar(im2, ax=axes[2])
        plt.suptitle(f'{case_name} (k={k_perm:.0e}, K={Ks:.0e}) — Temperature Field Comparison', fontsize=14, fontweight='bold')
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, f'{tag}_temperature_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)

        fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
        im0 = axes[0].imshow(u_pred.T, aspect='auto', origin='lower', extent=extent, cmap='viridis')
        axes[0].set_title('PINN Pore Pressure (kPa)', fontweight='bold'); axes[0].set_xlabel('Time (days)'); axes[0].set_ylabel('Radius r (m)')
        fig.colorbar(im0, ax=axes[0])
        im1 = axes[1].imshow(u_ref_data.T, aspect='auto', origin='lower', extent=extent, cmap='viridis')
        axes[1].set_title('Reference Pore Pressure (kPa)', fontweight='bold'); axes[1].set_xlabel('Time (days)')
        fig.colorbar(im1, ax=axes[1])
        im2 = axes[2].imshow(np.abs(u_pred - u_ref_data).T, aspect='auto', origin='lower', extent=extent, cmap='inferno')
        axes[2].set_title(f'Absolute Error (Rel L2: {err_L2_u*100:.2f}%)', fontweight='bold'); axes[2].set_xlabel('Time (days)')
        fig.colorbar(im2, ax=axes[2])
        plt.suptitle(f'{case_name} (k={k_perm:.0e}, K={Ks:.0e}) — Excess Pore Pressure Comparison', fontsize=14, fontweight='bold')
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, f'{tag}_pressure_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)

        # --- discrete per-day line comparisons (two plots: temperature, pressure) ---
        plot_daily_lines(
            r_ref, t_days, T_ref, T_pred, ylabel='Temperature (°C)',
            title=f'{case_name} (k={k_perm:.0e}, K={Ks:.0e}) — Temperature by day',
            savepath=os.path.join(OUTPUT_DIR, f'{tag}_temperature_daily_lines.png'),
        )
        plot_daily_lines(
            r_ref, t_days, u_ref_data, u_pred, ylabel='Excess pore pressure (kPa)',
            title=f'{case_name} (k={k_perm:.0e}, K={Ks:.0e}) — Pressure by day',
            savepath=os.path.join(OUTPUT_DIR, f'{tag}_pressure_daily_lines.png'),
        )

    except FileNotFoundError:
        print(f'[Note] Reference CSVs not found for {case_name} at {path_u_csv} -- skipping evaluation/plots.')
        summary.append({'case': case_name, 'k': k_perm, 'Ks': Ks, 'T_relL2_pct': None, 'u_relL2_pct': None})

# =========================================================
# 5. Summary across all cases
# =========================================================
df_summary = pd.DataFrame(summary)
print('\n================ Summary (all cases) ================')
print(df_summary.to_string(index=False))
df_summary.to_csv(os.path.join(OUTPUT_DIR, 'summary_all_cases.csv'), index=False)
print(f'\nSaved plots + weights + summary_all_cases.csv to: {OUTPUT_DIR}')
