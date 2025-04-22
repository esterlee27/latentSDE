import sys
# sys.path.append('../Pontryagin-Differentiable-Programming/')

# from PDP import PDP
# from JinEnv import JinEnv
from casadi import *
import math
import scipy.io as sio
import numpy as np

import logging
import os
from typing import Sequence

import fire
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import tqdm
from torch import nn
from torch import optim
from torch.distributions import Normal
import time
import matplotlib
import torchsde
import copy

seed = 'cartpole'

INI_STATE = [-0.067, -0.55, -0.35, 0.53]

_BARRIER_ITERATIONS = 40
_NUM = 100
_STEPS = 100
_DT = 0.1


class LinearScheduler(object):
    def __init__(self, iters, maxval=1.0):
        self._iters = max(1, iters)
        self._val = maxval / self._iters
        self._maxval = maxval

    def step(self):
        self._val = min(self._maxval, self._val + self._maxval / self._iters)

    @property
    def val(self):
        return self._val

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Encoder, self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size)
        self.lin = nn.Linear(hidden_size, output_size)

    def forward(self, inp):
        out, _ = self.gru(inp)
        out = self.lin(out)
        return out

class Controller(nn.Module):
    def __init__(self, data_size=4, hidden_size=16, u_size=1):
        super(Controller, self).__init__()
        self.lin1 = nn.Linear(data_size, u_size)
        self.lin1.weight.data = torch.Tensor([[-0.0652, -0.2577, -1.3080, -0.6947]])

    def forward(self, state):
        x = state
        return self.lin1(x)

class BarrierNN(nn.Module):
    def __init__(self, state_size, hidden_size):
        super(BarrierNN, self).__init__()
        self.B_net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

        nn.init.kaiming_normal_(self.B_net[0].weight, mode='fan_in')
        nn.init.kaiming_normal_(self.B_net[2].weight, mode='fan_in')
        nn.init.kaiming_normal_(self.B_net[4].weight, mode='fan_in')
    def forward(self, state):
        return self.B_net(state)

class LatentSDE(nn.Module):
    sde_type = "ito"
    noise_type = "diagonal"

    def __init__(self, data_size, latent_size, context_size, hidden_size, u_size=1):
        super(LatentSDE, self).__init__()
        # Encoder.
        self.encoder = Encoder(input_size=data_size + u_size, hidden_size=hidden_size, output_size=context_size)
        self.qz0_net = nn.Linear(context_size, latent_size + latent_size)

        # Decoder.
        self.f_net = nn.Sequential(
            nn.Linear(latent_size + context_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, latent_size),
        )

        self.c_net = Controller().to('cuda')


        self.h_net = nn.Sequential(
            nn.Linear(latent_size + u_size, latent_size),
        )
        # This needs to be an element-wise function for the SDE to satisfy diagonal noise.
        self.g_nets = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(1, hidden_size),
                    nn.Softplus(),
                    nn.Linear(hidden_size, 1),
                    nn.Sigmoid()
                )
                for _ in range(latent_size)
            ]
        )
        self.projector = nn.Linear(latent_size, data_size)

        self.pz0_mean = nn.Parameter(torch.zeros(1, latent_size))
        self.pz0_logstd = nn.Parameter(torch.zeros(1, latent_size))

        self._ctx = None

    def contextualize(self, ctx):
        self._ctx = ctx  # A tuple of tensors of sizes (T,), (T, batch_size, d).

    def f(self, t, y):
        ts, ctx = self._ctx
        i = min(torch.searchsorted(ts, t, right=True), len(ts) - 1)
        return self.f_net(torch.cat((y, ctx[i]), dim=1)) 

    def h(self, t, y):
        u = self.c_net(self.projector(y))
        y = torch.cat((y, u), dim=1)
        return self.h_net(y)

    def g(self, t, y):  # Diagonal diffusion.
        y = torch.split(y, split_size_or_sections=1, dim=1)
        out = [g_net_i(y_i) for (g_net_i, y_i) in zip(self.g_nets, y)]
        return torch.cat(out, dim=1)

    def forward(self, xs, ts, noise_std, adjoint=False, method="euler"):
        # Contextualization is only needed for posterior inference.
        us = self.c_net(xs)
        xus = torch.cat((xs, us), dim=2)
        ctx = self.encoder(torch.flip(xus, dims=(0,)))
        ctx = torch.flip(ctx, dims=(0,))
        self.contextualize((ts, ctx))

        qz0_mean, qz0_logstd = self.qz0_net(ctx[0]).chunk(chunks=2, dim=1)
        z0 = qz0_mean + 0.2*qz0_logstd.exp() * torch.randn_like(qz0_mean)

        if adjoint:
            # Must use the argument `adjoint_params`, since `ctx` is not part of the input to `f`, `g`, and `h`.
            adjoint_params = (
                    (ctx,) +
                    tuple(self.f_net.parameters()) + tuple(self.g_nets.parameters()) + tuple(self.h_net.parameters())
            )
            zs, log_ratio = torchsde.sdeint_adjoint(
                self, z0, ts, adjoint_params=adjoint_params, dt=1e-2, logqp=True, method=method)
        else:
            zs, log_ratio = torchsde.sdeint(self, z0, ts, dt=1e-2, logqp=True, method=method)

        _xs = self.projector(zs)
        r = torch.sum(torch.mean(torch.norm(_xs, dim=2), dim=(1)))
        xs_dist = Normal(loc=_xs, scale=noise_std)
        log_pxs = xs_dist.log_prob(xs).sum(dim=(0, 2)).mean(dim=0)

        qz0 = torch.distributions.Normal(loc=qz0_mean, scale=qz0_logstd.exp())
        pz0 = torch.distributions.Normal(loc=self.pz0_mean, scale=self.pz0_logstd.exp())
        logqp0 = torch.distributions.kl_divergence(qz0, pz0).sum(dim=1).mean(dim=0)
        logqp_path = log_ratio.sum(dim=0).mean(dim=0)
        return log_pxs, logqp0 + 35*logqp_path, r

    def sample(self, batch_size, ts, dt=1e-2, bm=None):
        eps = 0.3*torch.randn(size=(batch_size, *self.pz0_mean.shape[1:]), device=self.pz0_mean.device)
        z0 = self.pz0_mean + self.pz0_logstd.exp() * eps
        zs = torchsde.sdeint(self, z0, ts, names={'drift': 'h', 'diffusion':'g'}, dt=dt, bm=bm)
        # Most of the times in ML, we don't sample the observation noise for visualization purposes.
        _xs = self.projector(zs)               
        return _xs

    def setlatentInit(self):
        for _ in range(10):
            initx0 = self.projector(self.pz0_mean)
            target = torch.Tensor([INI_STATE]).to('cuda')
           
            opt = torch.optim.Adam([self.pz0_mean], lr=0.1)
            loss = torch.cdist(initx0, target, p=2)
            loss.backward()
            opt.step()


def make_dataset(batch_size, control, device):

    def _f(x, u):
        """
        Return the control-independent part of the control-affine dynamics.
        args:
            x: bs x self.n_dims tensor of state
            params: a dictionary giving the parameter values for the system. If None,
                    default to the nominal parameters used at initialization
        returns:
            f: bs x self.n_dims x 1 tensor
        """
        # Extract batch size and set up a tensor for holding the result
        f = np.zeros(4)

        pos = x[0]
        theta = x[1]
        v = x[2]
        theta_dot = x[3]

        f[0] = v
        f[1] = theta_dot
        f[2] =  (u + np.sin(theta)*(theta_dot**2 - np.cos(theta))) / (1 + np.sin(theta)**2)
        f[3] = (
                u*np.cos(theta) + 
                theta_dot**2 * np.cos(theta) * np.sin(theta) -
                2*np.sin(theta)) / (1 + np.sin(theta)**2)

        return f

    def next_state(x, u, ctrl_step):
        simulate_step = 1e-2
        steps = int(ctrl_step/simulate_step)
        for i in range(steps):
            x = x + simulate_step * (_f(x, u))
        return x

    demos = np.zeros((batch_size, _STEPS, 4))  # for data storage

    for bs in range(batch_size):
        ini_state = copy.copy(INI_STATE)
        ini_state[0] += np.random.uniform(-0.1, 0.1, 1)[0]
        ini_state[1] += np.random.uniform(-0.05, 0.05, 1)[0]
        state = np.array(ini_state)

        for i in range(_STEPS):
            r_noise = 0.03*np.random.normal(0, np.sqrt(_DT), 1)[0]
            v_noise = 0.03*np.random.normal(0, np.sqrt(_DT), 1)[0]
            diffusion = np.array([r_noise, v_noise, 0, 0])
            demos[bs,i] = state
            curr_u = control(torch.from_numpy(state).to(device).float()).detach().cpu().numpy()
            state = next_state(state, curr_u, _DT) + diffusion

    xs = torch.from_numpy(demos).float().to(device)
    ts = torch.linspace(0, _STEPS*_DT, steps=_STEPS, device=device)
    plt.plot(demos[0, :, 0], demos[0, :, 1])
    plt.savefig('cartpole_test.png')
    xs = torch.swapaxes(xs, 0, 1)


    return xs, ts


def vis(xs, ts, latent_sde, bm_vis, img_path, num_samples=10):
    fig = plt.figure(figsize=(20, 8))
    gs = gridspec.GridSpec(1, 2)
    ax00 = fig.add_subplot(gs[0, 0])
    ax01 = fig.add_subplot(gs[0, 1])

    # Left plot: data.
    z1, z2 = np.split(xs.cpu().numpy()[:, :, :2], indices_or_sections=2, axis=-1)
    [ax00.plot(z1[:, i, 0], z2[:, i, 0]) for i in range(num_samples)]
    ax00.scatter(z1[0, :num_samples, 0], z2[0, :num_samples, 0], marker='x')
    ax00.set_xlabel('$s_1$', labelpad=0., fontsize=23)
    ax00.set_ylabel('$s_2$', labelpad=.5, fontsize=23)
    ax00.set_title('Real', fontsize=30)

    # === 添加不安全区域标记：pos < -0.75 ===
    ax00.axvspan(-2, -0.75, color='red', alpha=0.2, label='Unsafe Region')
    ax00.legend(fontsize=15)

    # Right plot: samples from learned model.
    xs = latent_sde.sample(batch_size=xs.size(1), ts=ts, bm=bm_vis).detach().cpu().numpy()
    z1, z2 = np.split(xs[:, :, :2], indices_or_sections=2, axis=-1)
    
    [ax01.plot(z1[:, i, 0], z2[:, i, 0]) for i in range(num_samples)]
    ax01.scatter(z1[0, :num_samples, 0], z2[0, :num_samples, 0], marker='x')
    ax01.set_title('Generative', fontsize=30)
    ax01.set_xlabel('$\hat{s}_1$', labelpad=0., fontsize=23)
    ax01.set_ylabel('$\hat{s}_2$', labelpad=.5, fontsize=23)
    lab = ax00.get_xticklabels() + ax00.get_yticklabels() + ax01.get_xticklabels() + ax01.get_yticklabels()
    for l in lab:
        l.set_fontsize(18)

    # === 安全率计算 ===
    unsafe_mask = xs[:, :, 0] < -0.75  # 只看 pos 分量
    unsafe_per_traj = unsafe_mask.any(axis=0)  # [batch]，每条轨迹是否有不安全时刻
    num_safe = (~unsafe_per_traj).sum().item()
    total = xs.shape[1]
    safe_rate = num_safe / total * 100
    print(f"Trajectory Safety Rate: {safe_rate:.2f}% ({num_safe}/{total})")

    plt.savefig(img_path, bbox_inches='tight')
    plt.close()

# Keeping the function but commenting out its contents - it won't be called

def f_func(x, latent_sde):
    """Extract the drift function from latentSDE for state x"""

    # In the latentSDE model, states are in the projected space
    # We need to work directly in that space
    
    # Get control input
    u = latent_sde.c_net(x)  # [batch_size, 1]
    
    # In the original model, h_net takes [latent_state, control]
    # But we're providing actual state x, not latent state
    # We need to project x to latent space first (this is a simplification)
    latent_state = x  # Assume x is already in the correct space
    
    # print("latent_state shape:", latent_state.shape)
    # print("u shape:", u.shape)

    # Concatenate with control
    y_with_u = torch.cat([latent_state, u], dim=1)
    
    # Apply drift function
    return latent_sde.h_net(y_with_u)

def g_func(x, latent_sde):
    """Extract the diffusion function from latentSDE for state x"""
    # In latentSDE, diffusion is applied in latent space
    # Again, we assume x is already in the correct space
    
    # Apply g element-wise as in the original model
    y = torch.split(x, split_size_or_sections=1, dim=1)
    
    g_outputs = []
    for idx, (g_net_i, y_i) in enumerate(zip(latent_sde.g_nets, y)):
        # y_i shape: [64, 1]
        out = g_net_i(y_i)        # → [64, 1]
        g_outputs.append(out)     # 不需要 reshape，已经对了
    return torch.cat(g_outputs, dim=1)  # → [64, 4]

def compute_second_derivatives(B, x):
    grads = torch.autograd.grad(B, x, grad_outputs=torch.ones_like(B), create_graph=True, allow_unused=True)[0]

    # Check if grads is None
    if grads is None:
        return torch.zeros(x.size(0), 4, 4, device=x.device)  # Changed to 4x4 for cartpole state

    hessian = []
    for i in range(4):  # Changed to 4 for cartpole state
        grad2 = torch.autograd.grad(grads[:, i], x, grad_outputs=torch.ones_like(grads[:, i]), create_graph=True)[0]
        hessian.append(grad2.unsqueeze(-1))
    return torch.cat(hessian, dim=2)

def compute_Q(Phi, delta=1.0):
    sp = nn.functional.softplus(delta * Phi) / delta
    sig = torch.sigmoid(delta * Phi)
    return 2.0 * sp * sig

def compute_Phi(x, barrier_model, latent_sde, delta=1.0):
    """Compute Φ using the barrier model and SDE dynamics"""
    if not x.requires_grad:
        x.requires_grad_(True)
    
    # Apply barrier function
    B = barrier_model(x)
    
    # Compute gradient of B with respect to x
    grad_B = torch.autograd.grad(B.sum(), x, create_graph=True, allow_unused=True)[0]
    
    # Check if grad_B is None
    if grad_B is None:
        return torch.zeros(x.size(0), device=x.device)
    
    # Get drift and diffusion
    f_val = f_func(x, latent_sde)
    
    # Drift term: <∇B, f>
    drift_term = torch.sum(grad_B * f_val, dim=1)
    
    # Compute Hessian of B
    hessian_B = compute_second_derivatives(B, x)
    
    # Get diffusion
    g_val = g_func(x, latent_sde)
    
    # Reshape g for matrix multiplication
    # For SDE with diagonal noise, g is [batch, state_dim]
    # We need to reshape it appropriately for ggᵀ
    batch_size, state_dim = x.shape
    
    # Compute ggᵀ (diffusion matrix)
    ggT = torch.zeros(batch_size, state_dim, state_dim, device=x.device)
    for i in range(batch_size):
        g_i = g_val[i].view(-1, 1)  # Make column vector
        ggT[i] = g_i @ g_i.t()
    
    # Compute diffusion term: 0.5 * Tr(ggᵀ * Hess(B))
    diff_term = 0.5 * torch.sum(ggT * hessian_B, dim=(1, 2))
    
    # Φ = <∇B, f> + 0.5 * Tr(ggᵀ * Hess(B))
    return drift_term + diff_term

def compute_term4(x, barrier_model, latent_sde, delta=1.0):
    batch_size = x.size(0)
    dim = x.size(1)  # Should be 4 for cartpole
    
    if not x.requires_grad:
        x.requires_grad_(True)
    
    # Compute Phi and Q
    Phi_val = compute_Phi(x, barrier_model, latent_sde, delta=delta)
    Q_val = compute_Q(Phi_val, delta=delta)
    
    # Compute f and g
    f_val = f_func(x, latent_sde)
    g_val = g_func(x, latent_sde)
    
    # Create column vector from g_val
    g_val = g_val.view(batch_size, dim, 1)
    
    # Compute M = 0.5 * Q * ggT
    ggT = torch.bmm(g_val, torch.transpose(g_val, 1, 2))
    M_mat = 0.5 * Q_val.unsqueeze(-1).unsqueeze(-1) * ggT
    
    # Compute div(M)
    div_M = torch.zeros(batch_size, dim, device=x.device)
    for i in range(dim):
        for j in range(dim):
            Mij = M_mat[:, i, j]
            grad_Mij = torch.autograd.grad(Mij.sum(), x, create_graph=True)[0][:, j]
            div_M[:, i] += grad_Mij
    
    # Compute div(Qf)
    Qf = Q_val.unsqueeze(-1) * f_val
    div_Qf = torch.zeros(batch_size, dim, device=x.device)
    for i in range(dim):
        grad_Qf_i = torch.autograd.grad(Qf[:, i].sum(), x, create_graph=True)[0]
        div_Qf[:, i] = grad_Qf_i[:, i]
    
    # Compute div(div(M))
    div_div_M = torch.zeros(batch_size, dim, device=x.device)
    for i in range(dim):
        grad_divMi = torch.autograd.grad(div_M[:, i].sum(), x, create_graph=True)[0]
        div_div_M[:, i] = grad_divMi[:, i]
    
    # Calculate the squared difference and sum across dimensions
    term4 = (div_div_M - div_Qf).pow(2).sum(dim=1)
    
    return term4

# def compute_LB(x0, xu, xk, barrier_model, latent_sde, lambda_val=1, delta=1.0):
#     B_x0 = barrier_model(x0).mean()
#     B_xu = barrier_model(xu)
#     term2 = nn.functional.relu(1.0 - B_xu.clone()).mean()
    
#     # Add requires_grad if needed
#     if not xk.requires_grad:
#         xk.requires_grad_(True)
        
#     Phi_xk = compute_Phi(xk[-1], barrier_model, latent_sde, delta=delta)
#     term3 = nn.functional.relu(Phi_xk.clone()).mean()
#     # term4_vals = compute_term4(xk, barrier_model, latent_sde, delta=delta)
#     # term4 = term4_vals.mean()
        
#     LB = B_x0.clone() + 1.5 * term2 + term3 
#     # LB = B_x0.clone() + 1.5 * term2 + term3 + lambda_val * term4
#     return LB

def compute_LB(x0, xu, xs_trajectory, barrier_model, latent_sde, lambda_val=1, delta=1.0):
    """Compute barrier certificate loss with trajectory data"""
    # Initial state conditions
    B_x0 = barrier_model(x0).mean()
    
    # Unsafe state conditions
    B_xu = barrier_model(xu)
    term2 = nn.functional.relu(1.0 - B_xu.clone()).mean()
    
    # Reshape trajectory data
    time_steps, batch_size, state_dim = xs_trajectory.shape
    xk_flat = xs_trajectory.reshape(-1, state_dim)
    
    # Ensure requires_grad
    if not xk_flat.requires_grad:
        xk_flat.requires_grad_(True)
    
    # Compute Phi for all states
    Phi_xk = compute_Phi(xk_flat, barrier_model, latent_sde, delta=delta)
    
    # Reshape back
    Phi_xk = Phi_xk.reshape(time_steps, batch_size)
    
    # Average over time steps for term3
    term3 = nn.functional.relu(Phi_xk.clone()).mean()
    
    # Compute term4 (simplified)
    subsample_indices = torch.linspace(0, time_steps-1, 5, dtype=torch.long)
    xk_subsample = xs_trajectory[subsample_indices].reshape(-1, state_dim)
    
    if not xk_subsample.requires_grad:
        xk_subsample.requires_grad_(True)
    
    # term4_vals = compute_term4(xk_subsample, barrier_model, latent_sde, delta=delta)
    # term4 = term4_vals.mean()
    
    # Print component values for debugging
    print(f"B_x0: {B_x0.item():.6f}, term2: {term2.item():.6f}, term3: {term3.item():.6f}")
    # print(f"B_x0: {B_x0.item():.6f}, term2: {term2.item():.6f}, term3: {term3.item():.6f}, term4: {term4.item():.6f}")
    # Combine terms
    LB = B_x0.clone() + 1.5 * term2 + term3
    return LB

def trainBarrier(latent_sde, Barrier, batch_size=64, device='cuda', Test=False, lambda_val=1.0, delta=1.0):
    optimizer = torch.optim.Adam(Barrier.parameters())
    con_opt = torch.optim.Adam(latent_sde.c_net.parameters(), lr=1e-4)

    ### samples ###
    for it in range(_BARRIER_ITERATIONS):
        # Generate initial states
        _x0 = []
        for _ in range(batch_size):
            ini_state = copy.copy(INI_STATE)
            ini_state[0] += np.random.uniform(-0.1, 0.1, 1)[0]
            ini_state[1] += np.random.uniform(-0.05, 0.05, 1)[0]
            _x0.append(np.array(ini_state))
        _x0 = torch.from_numpy(np.array(_x0)).to(device).float()

        # Generate unsafe states
        _xu = []
        for _ in range(batch_size):
            pos = np.random.uniform(-2, -0.75)
            theta = np.random.uniform(-2, 2)
            v = np.random.uniform(-3, 3)
            theta_dot = np.random.uniform(-3, 3)
            _xu.append(np.array([pos, theta, v, theta_dot]))
        _xu = torch.from_numpy(np.array(_xu)).to(device).float()
        
        # Sample trajectories
        ts = torch.linspace(0, _STEPS*_DT, steps=_STEPS, device=device)
        _xs = latent_sde.sample(batch_size=_x0.size(0), ts=ts, dt=5e-2)
        
        # Compute reward term
        R = torch.mean(torch.mean(torch.norm(_xs, dim=2), dim=(1)))
   
        optimizer.zero_grad()
        con_opt.zero_grad()
        
        # Fix: properly pass _xu and _xs (no asterisks)
        LB = compute_LB(_x0, _xu, _xs, Barrier, latent_sde, lambda_val, delta)
        
        # Combine with reward term
        loss = LB + R
        loss.backward()

        if (it+1) % 5 == 0:
            print(f'Iter:{it}, loss:{loss.item():.2f}, Reward:-{R.item():.2f}, LB:{LB.item():.2f}', flush=True)
        
        optimizer.step()
        con_opt.step()
    return LB.item()

# def trainBarrier(latent_sde, Barrier, batch_size=64, device='cuda', Test=False):
    """
    optimizer = torch.optim.Adam(Barrier.parameters())
    con_opt = torch.optim.Adam(latent_sde.c_net.parameters(), lr=1e-4)

    ### samples ###
    for it in range(_BARRIER_ITERATIONS):
        _x0 = []
        for _ in range(batch_size):
            ini_state = copy.copy(INI_STATE)
            ini_state[0] += np.random.uniform(-0.1, 0.1, 1)[0]
            ini_state[1] += np.random.uniform(-0.05, 0.05, 1)[0]
            _x0.append(np.array(ini_state))
        _x0 = torch.from_numpy(np.array(_x0)).to(device).float()

        _xu = []
        for _ in range(batch_size):
            # random_vec = np.random.normal(0, 1, size=(1, 4))
            # normalize_vec = random_vec / np.linalg.norm(random_vec)
            # length = np.random.uniform(1, 1.5, 1)[0]
            # _xu.append(length * normalize_vec[0])
            pos = np.random.uniform(-2, -0.75)[0]
            theta = np.random.uniform(-2, 2)[0]
            v = np.random.uniform(-3, 3)[0]
            theta_dot = np.random.uniform(-3, 3)[0]
            _xu.append(np.array([pos, theta, v, theta_dot]))

        _xu = torch.from_numpy(np.array(_xu)).to(device).float()

        
        ts = torch.linspace(0, _STEPS*_DT, steps=_STEPS, device=device)
        _xs = latent_sde.sample(batch_size=_x0.size(0), ts=ts, dt=5e-2)
        
        R = torch.mean(torch.mean(torch.norm(_xs, dim=2), dim=(1)))
   
        optimizer.zero_grad()
        con_opt.zero_grad()

        barrier_xs = Barrier(_xs)

        Lie = torch.mean(barrier_xs[-1] - barrier_xs[0])
        Lie_max = torch.max(barrier_xs[-1] - barrier_xs[0])
        for i in range(_xs.size(0) - 1):
            Lie += torch.abs(torch.mean(barrier_xs[i+1] -barrier_xs[i])) 
            Lie_max = max(torch.max(barrier_xs[i+1] -barrier_xs[i]), Lie_max) 

        barrier_xu = Barrier(_xu)
        barrier_x0 = Barrier(_x0)
        Unsafe = torch.mean(1 - barrier_xu)
        Unsafe_min = torch.min(barrier_xu)
        Init = torch.mean(barrier_x0)
        Init_max = torch.max(barrier_x0)

        loss = Lie / 2.5 + 0.2*((1 - Unsafe_min) + Init_max) + R
        loss.backward()

        if (it+1) % 5 == 0:
            print('Iter:{}, loss:{:.2f}, Reward:-{:.2f}, Lie:{:.2f}, Unsafe:{:.2f}, Init:{:.2f}, Lie_max:{:.2f}, Unsafe_min:{:.2f}, Init_max:{:.2f}'
                .format(it,  loss.item(), R.item(), Lie.item(), Unsafe.item(), Init.item(), Lie_max.item(), Unsafe_min.item(), Init_max.item()), flush=True)
        optimizer.step()
        con_opt.step()
    """
    # pass  # Do nothing - commenting out the barrier training function

def plot_losses(loss_dict, step, save_path):
    """
    绘制多个 loss 子图（subplot）
    :param loss_dict: dict，key是 loss 名，value是 list of loss values
    :param step: 当前 step，用于文件名
    :param save_path: 保存路径
    """
    num_losses = len(loss_dict)
    fig, axes = plt.subplots(num_losses, 1, figsize=(8, 2.5 * num_losses), sharex=True)

    if num_losses == 1:
        axes = [axes]

    for ax, (name, values) in zip(axes, loss_dict.items()):
        ax.plot(values, label=name)
        ax.set_ylabel(name)
        ax.grid(True)
        ax.legend()

    axes[-1].set_xlabel("Iteration")
    fig.suptitle(f"Loss curves up to step {step}")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(save_path, f"losses_{step}.png"))
    plt.close()


def main(
        batch_size=64,
        latent_size=4,
        data_size=4,
        context_size=32,
        hidden_size=32,
        lr_init=1e-2,
        t0=0.,
        lr_gamma=0.997,
        num_iters=1000,  # 增加迭代次数，因为是从头训练
        kl_anneal_iters=1000,
        pause_every=500,
        noise_std=0.01,
        adjoint=False,
        train_dir='./train/cartpole_BCtest/',  # 新的保存目录
        method="euler",
):
    # 确保训练目录存在
    os.makedirs(train_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 初始化一个新的 LatentSDE 模型，不加载任何检查点
    latent_sde = LatentSDE(
        data_size=data_size,
        latent_size=latent_size,
        context_size=context_size,
        hidden_size=hidden_size,
    ).to(device)
    
    # 初始化 Controller 权重 (可选的自定义初始化)
    # 下面这行代码可以保留或修改，取决于您希望如何初始化控制器
    latent_sde.c_net.lin1.weight.data = torch.Tensor([[-0.1, -0.3, -1.0, -0.5]]).to(device)

    ## fine-tune controller
    Barrier = BarrierNN(state_size=4, hidden_size=64).to(device)

    # 从零开始训练的主循环
    total_num = 0
    
    # === Loss records ===
    loss_history = []
    log_pxs_history = []
    kl_history = []
    reward_history = []
    lb_history = []
    term4_history = []

    for ep in range(100):  # 增加训练轮数
        # trainBarrier(latent_sde, Barrier, batch_size=64, lambda_val=1.0, delta=1.0)
        LB = trainBarrier(
            latent_sde, Barrier,
            batch_size=batch_size,
            lambda_val=1.0, delta=1.0
        )
        lb_history.append(LB)
        # term4_history.append(term4)

        # 使用当前控制器生成数据集
        print(f"Epoch {ep}: Generating dataset...")
        xs, ts = make_dataset(batch_size=batch_size, control=latent_sde.c_net, device=device)
        
        # 打印当前状态和控制器权重
        print("Final state:", xs[-1, 0, :])
        print("Controller weights:", latent_sde.c_net.lin1.weight.data)
        
        # 为所有模型参数创建优化器
        optimizer = optim.Adam([
                                {'params': latent_sde.pz0_mean},
                                {'params': latent_sde.pz0_logstd},
                                {'params': latent_sde.h_net.parameters()},
                                {'params': latent_sde.projector.parameters()},
                                {'params': latent_sde.g_nets.parameters()},
                                {'params': latent_sde.f_net.parameters()},
                                {'params': latent_sde.qz0_net.parameters()},
                                {'params': latent_sde.encoder.parameters()},
                                {'params': latent_sde.c_net.parameters()},  # 也优化控制器
                            ], lr=lr_init)  # 使用较高的学习率

        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=lr_gamma)
        kl_scheduler = LinearScheduler(iters=kl_anneal_iters)

        # 固定相同的布朗运动用于可视化
        bm_vis = torchsde.BrownianInterval(
            t0=t0, t1=_STEPS*_DT, size=(batch_size, latent_size,), device=device, levy_area_approximation="space-time")
        
        # 每轮训练的迭代次数
        num = _NUM * 2  # 增加迭代次数
        
        for global_step in tqdm.tqdm(range(1, num + 1)):
            # 每500步尝试设置潜在状态初始化
            if global_step % 500 == 0:
                latent_sde.setlatentInit()
                
            latent_sde.zero_grad()
            log_pxs, log_ratio, r = latent_sde(xs, ts, noise_std, adjoint, method)
            loss = -log_pxs + log_ratio * kl_scheduler.val
            loss.backward()

            # === 记录主模型 loss ===
            loss_history.append(loss.item())
            log_pxs_history.append(log_pxs.item())
            kl_history.append(log_ratio.item())
            reward_history.append(r.item())

            optimizer.step()
            scheduler.step()
            kl_scheduler.step()
            total_num += 1

            if total_num % pause_every == 0:
                lr_now = optimizer.param_groups[0]['lr']
                print(
                    f'global_step: {global_step:06d}, lr: {lr_now:.5f}, '
                    f'log_pxs: {log_pxs:.4f}, log_ratio: {log_ratio:.4f} loss: {loss:.4f}, kl_coeff: {kl_scheduler.val:.4f}'
                )
                img_path = os.path.join(train_dir, f'step_{total_num}.png')
                vis(xs, ts, latent_sde, bm_vis, img_path)

                plot_losses({
                    "Total Loss": loss_history,
                    "-log p(x)": log_pxs_history,
                    "KL": kl_history,
                    "Reward": reward_history,
                    "LB": lb_history,
                    # "term3": term3_history
                }, step=total_num, save_path=train_dir)


                model_path = os.path.join(train_dir, f'model_{total_num}.pth')
                torch.save({'pz0_mean': latent_sde.pz0_mean, 'pz0_logstd': latent_sde.pz0_logstd, 
                    'h_net':latent_sde.h_net, 'projector':latent_sde.projector, 'g_nets':latent_sde.g_nets, 
                    'encoder':latent_sde.encoder, 'qz0_net':latent_sde.qz0_net, 'f_net':latent_sde.f_net, 'c_net':latent_sde.c_net}, model_path)


if __name__ == "__main__":
    fire.Fire(main)


"""
export CUDA_VISIBLE_DEVICES=5
nohup python -m examples.segway1 &>train_log/segway1.log &
echo $! > train_log/pid_segway1.txt
"""

