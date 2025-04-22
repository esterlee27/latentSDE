# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Train a latent SDE on data from a stochastic Lorenz attractor.
Reproduce the toy example in Section 7.2 of https://arxiv.org/pdf/2001.01328.pdf
To run this file, first run the following to install extra requirements:
pip install fire
To run, execute:
python -m examples.latent_sde_lorenz --train-dir ./train/
"""
import logging
import os
from typing import Sequence

import fire
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch
import tqdm
from torch import nn
from torch import optim
from torch.distributions import Normal
import time
import matplotlib

import torchsde

seed = '1'

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

class StochasticLorenz(object):
    """Stochastic Lorenz attractor.
    Used for simulating ground truth and obtaining noisy data.
    Details described in Section 7.2 https://arxiv.org/pdf/2001.01328.pdf
    Default a, b from https://openreview.net/pdf?id=HkzRQhR9YX
    """
    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self, control, b: Sequence = (0, 0.2), nominal=False):
        super(StochasticLorenz, self).__init__()
        self.b = b
        self.nominal = nominal
        self.control = control

    def f(self, t, y):
        x1, x2 = torch.split(y, split_size_or_sections=(1, 1), dim=1)

        if self.nominal:
            f1 = 0.8*x2
            f2 = -x1 - x2 - 0.3*x1**3

        else:
            u = self.control(y)

            f1 = x2
            f2 = (u - 0.5*x1**3)

        return torch.cat([f1, f2], dim=1)

    def g(self, t, y):
        x1, x2 = torch.split(y, split_size_or_sections=(1, 1), dim=1)
        b1, b2 = self.b
        if self.nominal:
            g1 = x1 * 0
            g2 = x2 * 0
        else:
            g1 = x1 * b1
            g2 = x2 * b2

        return torch.cat([g1, g2], dim=1)

    @torch.no_grad()
    def sample(self, x0, ts, noise_std, normalize):
        """Sample data for training. Store data normalization constants if necessary."""
        xs = torchsde.sdeint(self, x0, ts)
        if normalize:
            mean, std = torch.mean(xs, dim=(0, 1)), torch.std(xs, dim=(0, 1))
            xs.sub_(mean).div_(std).add_(torch.randn_like(xs) * noise_std)
        return xs


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
    def __init__(self, data_size=2, u_size=1):
        super(Controller, self).__init__()
        self.lin = nn.Linear(data_size, u_size, bias=False)
        self.lin.weight.data = torch.Tensor([[-1,  -1]])

    def forward(self, state):
        return self.lin(state)

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

    def __init__(self, data_size, latent_size, context_size, hidden_size):
        super(LatentSDE, self).__init__()
        # Encoder.
        self.encoder = Encoder(input_size=data_size + 1, hidden_size=hidden_size, output_size=context_size)
        self.qz0_net = nn.Linear(context_size, latent_size + latent_size)

        # Decoder.
        self.f_net = nn.Sequential(
            nn.Linear(latent_size + context_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, latent_size),
        )

        self.c_net = Controller()

        # for param in self.c_net.parameters():
        #     param.requires_grad = False


        self.h_net = nn.Sequential(
            nn.Linear(latent_size + 1, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, latent_size),
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
        self.fixu = False

    def contextualize(self, ctx):
        self._ctx = ctx  # A tuple of tensors of sizes (T,), (T, batch_size, d).

    def f(self, t, y):
        ts, ctx = self._ctx
        i = min(torch.searchsorted(ts, t, right=True), len(ts) - 1)
        return self.f_net(torch.cat((y, ctx[i]), dim=1)) 

    def h(self, t, y):
        if self.fixu:
            with torch.no_grad():
                u = self.c_net(self.projector(y))
        else:
            u = self.c_net(self.projector(y))
        y = torch.cat((y, u), dim=1)
        return self.h_net(y)

    def g(self, t, y):  # Diagonal diffusion.
        y = torch.split(y, split_size_or_sections=1, dim=1)
        out = [g_net_i(y_i) for (g_net_i, y_i) in zip(self.g_nets, y)]
        return torch.cat(out, dim=1)

    def forward(self, xs, ts, noise_std, adjoint=False, method="euler"):
        # Contextualization is only needed for posterior inference.
        if self.fixu:
            with torch.no_grad():
                us = self.c_net(xs)
        else:
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
        if not self.fixu:
            eps = 0.1*torch.randn(size=(batch_size, *self.pz0_mean.shape[1:]), device=self.pz0_mean.device)
            z0 = self.pz0_mean + self.pz0_logstd.exp() * eps
            zs = torchsde.sdeint(self, z0, ts, names={'drift': 'h', 'diffusion':'g'}, dt=dt, bm=bm)
            # Most of the times in ML, we don't sample the observation noise for visualization purposes.
            _xs = self.projector(zs)
        else:
            with torch.no_grad():
                eps = 0.1*torch.randn(size=(batch_size, *self.pz0_mean.shape[1:]), device=self.pz0_mean.device)
                # img GCN -> encode -> z0 -> sdeint -> zs -> xs y traject
                # img GCN -> NN -> u ->
                z0 = self.pz0_mean + self.pz0_logstd.exp() * eps
                zs = torchsde.sdeint(self, z0, ts, names={'drift': 'h', 'diffusion':'g'}, dt=dt, bm=bm)
                # Most of the times in ML, we don't sample the observation noise for visualization purposes.
                _xs = self.projector(zs)                
        return _xs

    def setlatentInit(self):
        for _ in range(10):
            initx0 = self.projector(self.pz0_mean)
            target = torch.Tensor([[-2, 0]]).to('cuda')
           
            opt = torch.optim.Adam([self.pz0_mean], lr=0.1)
            loss = torch.cdist(initx0, target, p=2)
            loss.backward()
            opt.step()


def make_dataset(t0, t1, batch_size, data_size, noise_std, train_dir, device, control, nominal=False):
    data_path = os.path.join(train_dir, 'data.pth')

    angle = 2* np.pi * torch.rand(size=(batch_size, 1))
    length = torch.sqrt(0.01*torch.rand(size=(batch_size, 1)))
    x = length * np.cos(angle) - 2
    y = length * np.sin(angle)

    _y0 = torch.cat((x, y), dim=1).to(device)

    ts = torch.linspace(t0, t1, steps=100, device=device)
    prev = time.time()
    xs = StochasticLorenz(control=control, nominal=nominal).sample(_y0, ts, noise_std, normalize=False)
    return xs, ts


def vis(xs, ts, latent_sde, bm_vis, img_path, num_samples=10):
    fig = plt.figure(figsize=(20, 8))
    gs = gridspec.GridSpec(1, 2)
    ax00 = fig.add_subplot(gs[0, 0])
    ax01 = fig.add_subplot(gs[0, 1])

    # Left plot: data.
    ax00.add_patch(matplotlib.patches.Rectangle((-1, 1.2), 1, 0.5, color='pink'))
    z1, z2 = np.split(xs.cpu().numpy(), indices_or_sections=2, axis=-1)
    [ax00.plot(z1[:, i, 0], z2[:, i, 0]) for i in range(num_samples)]
    ax00.scatter(z1[0, :num_samples, 0], z2[0, :num_samples, 0], marker='x')
    ax00.set_xlabel('$s_1$', labelpad=0., fontsize=23)
    ax00.set_ylabel('$s_2$', labelpad=.5, fontsize=23)
    ax00.set_title('Real', fontsize=30)

    # ax00.set_xticks(fontsize=18)
    # ax00.set_yticks(fontsize=18)


    # Right plot: samples from learned model.
    xs = latent_sde.sample(batch_size=xs.size(1), ts=ts, bm=bm_vis).cpu().numpy()
    z1, z2 = np.split(xs, indices_or_sections=2, axis=-1)
    
    ax01.add_patch(matplotlib.patches.Rectangle((-1, 1.2), 1, 0.5, color='pink'))
    [ax01.plot(z1[:, i, 0], z2[:, i, 0]) for i in range(num_samples)]
    ax01.scatter(z1[0, :num_samples, 0], z2[0, :num_samples, 0], marker='x')
    ax01.set_title('Generative', fontsize=30)
    ax01.set_xlabel('$\hat{s}_1$', labelpad=0., fontsize=23)
    ax01.set_ylabel('$\hat{s}_2$', labelpad=.5, fontsize=23)
    # ax01.set_xticks(fontsize=18)
    # ax01.set_yticks(fontsize=18)
    lab = ax00.get_xticklabels() + ax00.get_yticklabels() + ax01.get_xticklabels() + ax01.get_yticklabels()
    for l in lab:
        l.set_fontsize(18)

    plt.savefig(img_path, bbox_inches='tight')
    plt.close()

def trainBarrier(latent_sde, batch_size=1024, device='cuda', Test=False):

    Barrier = BarrierNN(state_size=2, hidden_size=64).to(device)

    if Test:
        Barrier.load_state_dict(torch.load('D:/UoS/yixuan_icml/safe_unknown_control/train/Barrier.pth'))
        print('here')

    optimizer = torch.optim.Adam(Barrier.parameters())
    con_opt = torch.optim.Adam(latent_sde.c_net.parameters(), lr=0.01)

    ## TO do, sampling-based Barrier is not sound

    ### samples ###
    for it in range(100):
        angle = 2* np.pi * torch.rand(size=(batch_size, 1))
        length = torch.sqrt(0.01*torch.rand(size=(batch_size, 1)))
        x = length * np.cos(angle) - 2
        y = length * np.sin(angle)
        _x0 = torch.cat((x, y), dim=1).to(device)
        
        # x = 6*torch.rand(size=(batch_size, 1)) - 3
        x = -1 + 1*torch.rand(size=(batch_size, 1))
        y = 1.2 + 0.5*torch.rand(size=(batch_size, 1))
        _xu = torch.cat((x, y), dim=1).to(device)

        x = 6*torch.rand(size=(batch_size, 1)) - 3
        y = 6*torch.rand(size=(batch_size, 1)) - 3
        _xx = torch.cat((x, y), dim=1).to(device)

        ts = torch.linspace(0, 1.2, steps=30, device=device)
        _xs = latent_sde.sample(batch_size=_xx.size(0), ts=ts)
        
        R = 0.1*torch.sum(torch.mean(torch.norm(_xs, dim=2), dim=(1)))
        # weight = [0.5, 1, 2] 
        optimizer.zero_grad()
        con_opt.zero_grad()

        Lie = torch.mean(Barrier(_xs[-1]) - Barrier(_xs[0]))
        Lie_max = torch.max(Barrier(_xs[-1]) - Barrier(_xs[0]))
        for i in range(_xs.size(0) - 1):
            Lie += torch.mean(Barrier(_xs[-1]) - Barrier(_xs[i+1])) 
            Lie_max = max(torch.max(Barrier(_xs[i+1]) - Barrier(_xs[i])), Lie_max) 
        Lie /= 10

        Unsafe = torch.mean(1 - Barrier(_xu))
        Unsafe_min = torch.min(Barrier(_xu))
        Init = torch.mean(Barrier(_x0))
        # print(Barrier(_x0), Barrier(_x0).shape)
        Init_max = torch.max(Barrier(_x0))

        if Test:
            print(Unsafe.item(), Unsafe_min, Init.item(), 
                Init_max, Lie.item(), Lie_max)
            assert False

        if Lie_max <= 0.005 and Unsafe_min >= 0.99 and Init_max < 0.02:
            print(torch.max(Barrier(_x0)))
            print('learned controller is: ', latent_sde.c_net.lin.weight.data)
            torch.save(Barrier.state_dict(), 'D:/UoS/yixuan_icml/safe_unknown_control/train/Barrier'+seed+'.pth')
            
            mean_list = []
            std_list = []
            for j in range(_xs.size(0)):
                value = Barrier(_xs[j]).detach().cpu().numpy()
                mean_list.append(np.mean(value))
                std_list.append(np.std(value))

            mean_list = np.array(mean_list)
            std_list = np.array(std_list)
            plt.clf()
            plt.plot(np.arange(len(mean_list)), mean_list)
            plt.fill_between(np.arange(len(mean_list)), mean_list - std_list,  mean_list + std_list, alpha=0.3)
            plt.savefig('barrier'+seed+'.png')
            return True

        loss = Lie_max + (1 - Unsafe_min) + Init_max + R
        loss.backward()
        optimizer.step()
        con_opt.step()


        if it % 10 == 0:
            print('Iter:{}, loss:{:.2f}, Reward:-{:.2f}, Lie:{:.2f}, Unsafe:{:.2f}, Init:{:.2f}, Lie_max:{:.2f}, Unsafe_min:{:.2f}, Init_max:{:.2f}'
                .format(it,  loss.item(), R.item(), Lie.item(), Unsafe.item(), Init.item(), Lie_max.item(), Unsafe_min.item(), Init_max.item()))
            print(latent_sde.c_net.lin.weight.data, latent_sde.c_net.lin.weight.data.grad)

    return False


def main(
        batch_size=1024,
        latent_size=4,
        data_size=2,
        context_size=64,
        hidden_size=128,
        lr_init=1e-2,
        t0=0.,
        t1=6,
        lr_gamma=0.997,
        num_iters=5000,
        kl_anneal_iters=50,
        pause_every=50,
        noise_std=0.01,
        adjoint=False,
        train_dir='./train/2D/',
        method="euler",
):

    # 确保训练目录存在
    os.makedirs(train_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    latent_sde = LatentSDE(
        data_size=data_size,
        latent_size=latent_size,
        context_size=context_size,
        hidden_size=hidden_size,
    ).to(device)

    # checkpoint = torch.load('./train/new/model.pth')
    checkpoint = torch.load('model_1.pth')
    latent_sde.pz0_mean = checkpoint['pz0_mean']
    latent_sde.pz0_logstd = checkpoint['pz0_logstd']
    latent_sde.h_net = checkpoint['h_net']
    latent_sde.projector = checkpoint['projector']
    latent_sde.g_nets = checkpoint['g_nets']
    latent_sde.f_net = checkpoint['f_net']
    latent_sde.qz0_net = checkpoint['qz0_net']
    latent_sde.encoder = checkpoint['encoder']
    latent_sde.c_net = checkpoint['c_net']
    latent_sde.setlatentInit()
    
    ## fine-tune controller
    # Barrier = BarrierNN(state_size=2, hidden_size=64).to(device)
    prev_flag = False
    for _ in range(20):
        latent_sde.setlatentInit()
        latent_sde.fixu = False
        # flag = trainBarrier(latent_sde)            

        latent_sde.fixu = True
        print('controller after control optimization: ', latent_sde.c_net.lin.weight.data)
        xs, ts = make_dataset(t0=t0, t1=t1, batch_size=batch_size, data_size=data_size, noise_std=noise_std, train_dir=train_dir, device=device, control=latent_sde.c_net, nominal=False)
        optimizer = optim.Adam(params=latent_sde.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=lr_gamma)
        kl_scheduler = LinearScheduler(iters=kl_anneal_iters)

        # Fix the same Brownian motion for visualization.
        bm_vis = torchsde.BrownianInterval(
            t0=t0, t1=t1, size=(batch_size, latent_size,), device=device, levy_area_approximation="space-time")

        for global_step in tqdm.tqdm(range(0, 200 + 1)):
            latent_sde.zero_grad()
            log_pxs, log_ratio, r = latent_sde(xs, ts, noise_std, adjoint, method)
            # print(log_pxs, log_ratio)
            # assert False
            loss = -log_pxs + log_ratio * kl_scheduler.val
            loss.backward()

            optimizer.step()
            scheduler.step()
            kl_scheduler.step()

            if global_step % pause_every == 0:
                lr_now = optimizer.param_groups[0]['lr']
                logging.warning(
                    f'global_step: {global_step:06d}, lr: {lr_now:.5f}, '
                    f'log_pxs: {log_pxs:.4f}, log_ratio: {log_ratio:.4f} loss: {loss:.4f}, kl_coeff: {kl_scheduler.val:.4f}'
                )
                img_path = os.path.join(train_dir, f'seed_'+seed+'_step_'+str(global_step)+'_.pdf')
                vis(xs, ts, latent_sde, bm_vis, img_path)

            # if global_step % 100 == 0:
            #     model_path = os.path.join(train_dir, 'model_'+seed+'.pth')
            #     torch.save({'pz0_mean': latent_sde.pz0_mean, 'pz0_logstd': latent_sde.pz0_logstd, 
            #         'h_net':latent_sde.h_net, 'projector':latent_sde.projector, 'g_nets':latent_sde.g_nets, 
            #         'encoder':latent_sde.encoder, 'qz0_net':latent_sde.qz0_net, 'f_net':latent_sde.f_net, 'c_net':latent_sde.c_net}, model_path)

        print('controller after generative modeling: ', latent_sde.c_net.lin.weight.data)

        if prev_flag and flag:
            # [-0.5309, -3.3263]
            print('learned controller is: ', latent_sde.c_net.lin.weight.data)
            assert False

        prev_flag = flag
    ## training
    ## for training generative modeling of SDE/ learn a starting point
    # optimizer = optim.Adam(params=latent_sde.parameters(), lr=lr_init)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=lr_gamma)
    # kl_scheduler = LinearScheduler(iters=kl_anneal_iters)

    # # Fix the same Brownian motion for visualization.
    # bm_vis = torchsde.BrownianInterval(
    #     t0=t0, t1=t1, size=(batch_size, latent_size,), device=device, levy_area_approximation="space-time")

    # for global_step in tqdm.tqdm(range(1, num_iters + 1)):
    #     latent_sde.zero_grad()
    #     log_pxs, log_ratio, r = latent_sde(xs, ts, noise_std, adjoint, method)
    #     loss = -log_pxs + log_ratio * kl_scheduler.val
    #     loss.backward()

    #     # r.backward()
    #     # for names, param in latent_sde.c_net.named_parameters():
    #     #     print(names, param.shape, param.grad)
    #     #     assert False

    #     optimizer.step()
    #     scheduler.step()
    #     kl_scheduler.step()

    #     # print(latent_sde.c_net.lin.weight.data)
    #     # assert False

    #     if global_step % pause_every == 0:
    #         lr_now = optimizer.param_groups[0]['lr']
    #         logging.warning(
    #             f'global_step: {global_step:06d}, lr: {lr_now:.5f}, '
    #             f'log_pxs: {log_pxs:.4f}, log_ratio: {log_ratio:.4f} loss: {loss:.4f}, kl_coeff: {kl_scheduler.val:.4f}'
    #         )
    #         img_path = os.path.join(train_dir, f'global_step_{global_step:06d}.pdf')
    #         vis(xs, ts, latent_sde, bm_vis, img_path)

    #     if global_step % 100 == 0:
    #         model_path = os.path.join(train_dir, 'model.pth')
    #         torch.save({'pz0_mean': latent_sde.pz0_mean, 'pz0_logstd': latent_sde.pz0_logstd, 
    #             'h_net':latent_sde.h_net, 'projector':latent_sde.projector, 'g_nets':latent_sde.g_nets, 
    #             'encoder':latent_sde.encoder, 'qz0_net':latent_sde.qz0_net, 'f_net':latent_sde.f_net, 'c_net':latent_sde.c_net}, model_path)


if __name__ == "__main__":
    fire.Fire(main)
