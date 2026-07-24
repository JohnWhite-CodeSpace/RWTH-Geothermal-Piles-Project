import os
import time
from scipy.stats.qmc import LatinHypercube
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# =========================================================
# 0. Global Setup & Device Allocation
# =========================================================

torch.set_default_dtype(torch.float32)
SEED = 7
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))
print(f'Using device: {device}')

# Directory Setup
BASE_DIR = '/Users/seanxie/Desktop/RWTH PINN/Geothermal Piles Updated'
DATA_DIR = os.path.join(BASE_DIR, 'case3')
os.makedirs(DATA_DIR, exist_ok=True)

# Physical & Domain Parameters
R, Rs, Ts, Tf = 0.5, 30.0, 12.0, 50.0
n = 0.25
alpha_w = 5.00e-10
beta_s, beta_w = 3.00e-5, 3.42e-4
rho_s, Gamma, C_s = 2275.0, 2.0, 1674.4
gamma_w = 1000.0 * 9.81
t_E = 15 * 24 * 3600.0
alpha = Gamma / (rho_s * C_s)
beta_bar = (1 - n) * beta_s + n * beta_w
xi_R = R / Rs
alpha_n = alpha * t_E / Rs**2

K_TRUE, KS_ASSUMED = 1.0e-10, 2.0e6   # case3's known k
S_known = n * alpha_w + 1.0 / KS_ASSUMED
u_ref_scale = beta_bar * (Tf - Ts) / S_known
print(f'S_known={S_known:.4e}  u_ref_scale={u_ref_scale:.4e} Pa')

# Load Real Measurement CSV Data
df_T = pd.read_csv(os.path.join(DATA_DIR, 'case3_temperature.csv'))
df_u = pd.read_csv(os.path.join(DATA_DIR, 'case3_porepressure.csv'))
r_ref = df_T.columns[1:].astype(float).values
t_days = df_T['time_days'].values
u_ref_full = df_u.iloc[:, 1:].values

# =========================================================
# 1. Neural Network Definition & Pre-trained Temperature Model
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

# Architecture updated to match checkpoint depth: [2] + [64]*5 + [1]
net_T = MLP([2] + [64] * 5 + [1], nn.Tanh).to(device)

# Robust Weight Loading
ckpt_path = os.path.join(DATA_DIR, 'net_T_case3.pt')
checkpoint = torch.load(ckpt_path, map_location=device)

if isinstance(checkpoint, dict) and 'net_T' in checkpoint:
    net_T.load_state_dict(checkpoint['net_T'])
else:
    net_T.load_state_dict(checkpoint)

for p in net_T.parameters():
    p.requires_grad_(False)
net_T.eval()

def model_T(xi, tau):
    return net_T(torch.cat([xi, tau], dim=1))

# ---------------- sparse sensor set from real pore-pressure CSV ----------------
STRIDE_R = 100   # ~30 radii out of 2951; all 16 days
r_obs = r_ref[::STRIDE_R]
t_obs_days = t_days
RR, TT = np.meshgrid(r_obs, t_obs_days)
xi_obs = torch.tensor((RR / Rs).reshape(-1, 1), dtype=torch.float32).to(device)
tau_obs = torch.tensor(((TT * 86400.0) / t_E).reshape(-1, 1), dtype=torch.float32).to(device)
U_obs_target = torch.tensor(
    (u_ref_full[:, ::STRIDE_R] / u_ref_scale).reshape(-1, 1), dtype=torch.float32
).to(device)
print(f'Sensor set: {len(r_obs)} radii x {len(t_obs_days)} days = {xi_obs.shape[0]} points')

# ---------------- domain / IC / BC points ----------------
def lhs_sample_log_xi(n_pts, xi_min, xi_max, tau_min, tau_max, seed):
    lhs = LatinHypercube(d=2, seed=seed)
    s = lhs.random(n=n_pts)
    logxi = np.log(xi_min) + (np.log(xi_max) - np.log(xi_min)) * s[:, 0]
    tau = tau_min + (tau_max - tau_min) * s[:, 1]
    return torch.tensor(np.exp(logxi), dtype=torch.float32).reshape(-1, 1), \
           torch.tensor(tau, dtype=torch.float32).reshape(-1, 1)

N_DOM, N_IC, N_BC = 3000, 300, 300
xi_dom_raw, tau_dom_raw = lhs_sample_log_xi(N_DOM, xi_R, 1.0, 0.0, 1.0, SEED)
xi_dom = xi_dom_raw.to(device).requires_grad_(True)
tau_dom = tau_dom_raw.to(device).requires_grad_(True)

xi_ic = torch.logspace(np.log10(xi_R), 0.0, N_IC).reshape(-1, 1).to(device)
tau_ic = torch.zeros_like(xi_ic).to(device)

tau_bc = torch.linspace(0.0, 1.0, N_BC).reshape(-1, 1).to(device)
xi_bcL = torch.full_like(tau_bc, xi_R).to(device).requires_grad_(True)
tau_bcL = tau_bc.clone().to(device).requires_grad_(True)
xi_bcR = torch.full_like(tau_bc, 1.0).to(device)
tau_bcR = tau_bc.clone().to(device)

mse = nn.MSELoss()

# =========================================================
# 2. Inverse PINN Architecture
# =========================================================

class InversePINN_U(nn.Module):
    def __init__(self, net, theta_model, gamma_w, alpha, S_known, alpha_n, log10_k_init):
        super().__init__()
        self.net = net
        self.theta_model = theta_model
        self.gamma_w, self.alpha, self.S_known, self.alpha_n = gamma_w, alpha, S_known, alpha_n
        self.log10_k = nn.Parameter(torch.tensor(log10_k_init, dtype=torch.float32))

    @property
    def k(self):
        return 10.0 ** self.log10_k

    @property
    def D(self):
        return self.k / (self.gamma_w * self.alpha * self.S_known)

    def forward(self, xi, tau):
        return self.net(torch.cat([xi, tau], dim=1))

    def theta_tau(self, xi, tau):
        xi_ = xi.detach().clone().requires_grad_(True)
        tau_ = tau.detach().clone().requires_grad_(True)
        Th = self.theta_model(xi_, tau_)
        return torch.autograd.grad(Th, tau_, torch.ones_like(Th), create_graph=True)[0].detach()

    def pde_residual(self, xi, tau, src=None):
        U = self.forward(xi, tau)
        U_xi = torch.autograd.grad(U, xi, torch.ones_like(U), create_graph=True)[0]
        U_xixi = torch.autograd.grad(U_xi, xi, torch.ones_like(U_xi), create_graph=True)[0]
        U_tau = torch.autograd.grad(U, tau, torch.ones_like(U), create_graph=True)[0]
        if src is None:
            src = self.theta_tau(xi, tau)
        return U_tau - self.D * self.alpha_n * (U_xi / xi + U_xixi) - src

    def neumann_bc(self, xi, tau):
        U = self.forward(xi, tau)
        return torch.autograd.grad(U, xi, torch.ones_like(U), create_graph=True)[0]


def run_inversion(lambda_data, log10_k_init, epochs, tag, clamp=(-13.0, -6.0)):
    # Architecture updated to match checkpoint depth: [2] + [64]*5 + [1]
    net_U = MLP([2] + [64] * 5 + [1], nn.Tanh).to(device)
    model = InversePINN_U(net_U, model_T, gamma_w, alpha, S_known, alpha_n, log10_k_init).to(device)

    opt = torch.optim.Adam([
        {'params': model.net.parameters(), 'lr': 2e-3},
        {'params': [model.log10_k], 'lr': 2e-2},
    ])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.7, patience=300)

    src_fixed = model.theta_tau(xi_dom, tau_dom)

    k_hist, data_hist, pde_hist = [], [], []
    best = {'data_loss': float('inf'), 'k': None, 'epoch': None, 'state': None}

    t0 = time.time()
    for ep in range(epochs):
        opt.zero_grad()
        res = model.pde_residual(xi_dom, tau_dom, src=src_fixed)
        loss_pde = mse(res, torch.zeros_like(res))
        loss_ic = mse(model(xi_ic, tau_ic), torch.zeros_like(xi_ic))
        bc_neu = model.neumann_bc(xi_bcL, tau_bcL)
        loss_bcL = mse(bc_neu, torch.zeros_like(bc_neu))
        loss_bcR = mse(model(xi_bcR, tau_bcR), torch.zeros_like(xi_bcR))
        loss_data = mse(model(xi_obs, tau_obs), U_obs_target)

        loss = loss_pde + loss_ic + loss_bcL + loss_bcR + lambda_data * loss_data
        loss.backward()
        opt.step()
        sched.step(loss.item())

        with torch.no_grad():
            model.log10_k.clamp_(*clamp)

        if loss_data.item() < best['data_loss']:
            best.update(data_loss=loss_data.item(), k=model.k.item(), epoch=ep)

        k_hist.append(model.k.item())
        data_hist.append(loss_data.item())
        pde_hist.append(loss_pde.item())

        if ep % 500 == 0 or ep == epochs - 1:
            print(f'[{tag}] ep {ep:5d} | loss={loss.item():.3e} | pde={loss_pde.item():.3e} | '
                  f'data={loss_data.item():.3e} | k_est={model.k.item():.4e}')

    print(f'[{tag}] time={time.time() - t0:.1f}s | final k={model.k.item():.4e} '
          f'(err {abs(model.k.item() - K_TRUE) / K_TRUE * 100:.1f}%) | '
          f'best-checkpoint k={best["k"]:.4e} (ep {best["epoch"]}, err {abs(best["k"] - K_TRUE) / K_TRUE * 100:.1f}%)')
    return {'tag': tag, 'k_hist': k_hist, 'data_hist': data_hist, 'pde_hist': pde_hist,
            'final_k': model.k.item(), 'best_k': best['k'], 'best_epoch': best['epoch']}


# =========================================================
# 3. Execution & Visualization
# =========================================================

print(f'\nTrue k for case3 = {K_TRUE:.2e}\n')

results = []
results.append(run_inversion(lambda_data=50.0,  log10_k_init=-10.0, epochs=4000, tag='lam=50'))
results.append(run_inversion(lambda_data=200.0, log10_k_init=-10.0, epochs=4000, tag='lam=200'))
results.append(run_inversion(lambda_data=1000.0, log10_k_init=-10.0, epochs=4000, tag='lam=1000'))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for r in results:
    axes[0].plot(r['k_hist'], label=r['tag'])
axes[0].axhline(K_TRUE, color='k', ls='--', label='true k')
axes[0].set_yscale('log')
axes[0].set_xlabel('epoch')
axes[0].set_ylabel('k (m/s)')
axes[0].legend()
axes[0].set_title('k convergence, real case3 data')

for r in results:
    axes[1].plot(r['data_hist'], label=r['tag'])
axes[1].set_yscale('log')
axes[1].set_xlabel('epoch')
axes[1].set_ylabel('data loss')
axes[1].legend()
axes[1].set_title('data loss')

fig.tight_layout()
out_fig_path = os.path.join(DATA_DIR, 'inversion_lambda_sweep.png')
fig.savefig(out_fig_path, dpi=150, bbox_inches='tight')
print(f'\nSaved {out_fig_path}')