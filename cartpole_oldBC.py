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

    # === 添加不安全区域标记：pos < -0.75 ===
    ax01.axvspan(-2, -0.75, color='red', alpha=0.2, label='Unsafe Region')
    ax01.legend(fontsize=15)

    # === 安全率计算 ===
    unsafe_mask = xs[:, :, 0] < -0.75  # 只看 pos 分量
    unsafe_per_traj = unsafe_mask.any(axis=0)  # [batch]，每条轨迹是否有不安全时刻
    num_safe = (~unsafe_per_traj).sum().item()
    total = xs.shape[1]
    safe_rate = num_safe / total * 100
    print(f"Trajectory Safety Rate: {safe_rate:.2f}% ({num_safe}/{total})")

    plt.savefig(img_path, bbox_inches='tight')
    plt.close()

def trainBarrier(latent_sde, Barrier, batch_size=64, device='cuda', Test=False):
    optimizer = torch.optim.Adam(Barrier.parameters())
    con_opt = torch.optim.Adam(latent_sde.c_net.parameters(), lr=1e-4)

    for it in range(_BARRIER_ITERATIONS):
        # Sample initial states
        _x0 = []
        for _ in range(batch_size):
            ini_state = copy.copy(INI_STATE)
            ini_state[0] += np.random.uniform(-0.1, 0.1, 1)[0]
            ini_state[1] += np.random.uniform(-0.05, 0.05, 1)[0]
            _x0.append(np.array(ini_state))
        _x0 = torch.from_numpy(np.array(_x0)).to(device).float()

        # Sample unsafe states
        _xu = []
        for _ in range(batch_size):
            pos = np.random.uniform(-2, -0.75)
            theta = np.random.uniform(-2, 2)
            v = np.random.uniform(-3, 3)
            theta_dot = np.random.uniform(-3, 3)
            _xu.append(np.array([pos, theta, v, theta_dot]))
        _xu = torch.from_numpy(np.array(_xu)).to(device).float()

        # Simulate trajectories
        ts = torch.linspace(0, _STEPS * _DT, steps=_STEPS, device=device)
        _xs = latent_sde.sample(batch_size=_x0.size(0), ts=ts, dt=5e-2)

        # Reward term
        R = torch.mean(torch.norm(_xs, dim=2).mean(dim=1))

        optimizer.zero_grad()
        con_opt.zero_grad()

        barrier_xs = Barrier(_xs)
        barrier_x0 = Barrier(_x0)
        barrier_xu = Barrier(_xu)

        # Lie term
        Lie = torch.mean(barrier_xs[-1] - barrier_xs[0])
        Lie_max = torch.max(barrier_xs[-1] - barrier_xs[0])
        for i in range(_xs.size(0) - 1):
            Lie += torch.abs(torch.mean(barrier_xs[i + 1] - barrier_xs[i]))
            Lie_max = max(torch.max(barrier_xs[i + 1] - barrier_xs[i]), Lie_max)

        # Unsafe and Init
        Unsafe_min = torch.min(barrier_xu)
        Init_max = torch.max(barrier_x0)
        Unsafe = torch.mean(1 - barrier_xu)
        Init = torch.mean(barrier_x0)

        # Individual loss terms
        lie_loss = Lie / 2.5
        safety_loss = 0.2 * ((1 - Unsafe_min) + Init_max)
        reward_loss = R

        # Total loss
        loss = lie_loss + safety_loss + reward_loss
        loss.backward()

        if (it + 1) % 5 == 0:
            print('Iter:{}, loss:{:.2f}, Lie Loss:{:.2f}, Safety Loss:{:.2f}, Reward Loss:{:.2f}, Lie_max:{:.2f}, Unsafe_min:{:.2f}, Init_max:{:.2f}'.format(
                it, loss.item(), lie_loss.item(), safety_loss.item(), reward_loss.item(), Lie_max.item(), Unsafe_min.item(), Init_max.item()
            ), flush=True)

        optimizer.step()
        con_opt.step()

    return loss.item()

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
        train_dir='./train/cartpole_oldBC/',  # 新的保存目录
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
            batch_size=batch_size
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

