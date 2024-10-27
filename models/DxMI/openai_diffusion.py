"""
Wrapper for diffusion models provided by OpenAI repositories
"""
import torch
import torch.nn as nn
from models.cm.karras_diffusion import get_sigmas_karras, get_ancestral_step
from models.cm.nn import append_dims


class OpenAIDiffusion:
    def __init__(self, model, diffusion, n_timesteps, sample_shape, class_cond=False, num_classes=0,
                 trainable_beta=False, sigma_min=0.002, sigma_max=80., stochastic_last=False, rho=7.0):
        """
        model: unet
        diffusion: KarrasDenoiser     
        n_timesteps: the number of timesteps
        sample_shape: tuple for shape (C, H, W)
        trainable_beta: True, False, or fix_last
        stochastic_last: This option will let the last noise step have non-zero sigma (sigma_min)
                         EDM originally sets the last noise as 0. (stochastic_last=False)
        """
        self.net = model
        self.diffusion = diffusion
        self.class_cond = class_cond
        self.num_classes = num_classes
        self.sample_shape = sample_shape
        self.n_timesteps = n_timesteps
        self.sigma_max = sigma_max
        if stochastic_last:
            self.sigmas = get_sigmas_karras(n_timesteps+1, sigma_min, sigma_max, rho=rho, device='cpu')[:-1]
        else:
            self.sigmas = get_sigmas_karras(n_timesteps, sigma_min, sigma_max, rho=rho, device='cpu')

        # compute ancestral sampling sigmas
        # sigma_up is what actually added as noise
        self.sigma_down, self.sigma_up = self.get_ancestral_step(self.sigmas)

        print(self.sigmas)
        # print(self.sigma_down)  last sigmas are all zero
        # print(self.sigma_up)
        # self.sigmas[0] has the max sigma
        self.trainable_beta = trainable_beta
        if trainable_beta:
            sigma_lowerbound = 1e-3
            self.net.register_parameter("log_betas", nn.Parameter(torch.log(self.sigma_up.clamp(sigma_lowerbound))))
            # the last sigmas is 0, so to avoid -inf to be our parameter
        else:
            self.net.register_buffer("log_betas", torch.log(self.sigma_up))

    def get_ancestral_step(self, sigmas):
        sigma_from, sigma_to = sigmas[:-1], sigmas[1:]
        sigma_up = (
            sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2
        ) ** 0.5
        sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5
        return sigma_down, sigma_up

    def train(self):
        self.net.train()

    def eval(self):
        self.net.eval()

    def parameters(self):
        return self.net.parameters()

    def sample_step(self, x, indices, **model_kwargs):
        dims = x.ndim
        indices = indices.cpu()
        sigma = self.sigmas[indices].to(x.device)
        _, denoised = self.diffusion.denoise(self.net, x, sigma, **model_kwargs)
        # sigma_down, sigma_up = get_ancestral_step(self.sigmas[indices], self.sigmas[indices + 1])
        sigma_down, sigma_up = self.sigma_down[indices].to(x.device), self.sigma_up[indices].to(x.device)

        d = (x - denoised) / append_dims(sigma, dims)
        dt = sigma_down - sigma
        dt = append_dims(dt, dims)
        mu = x + d * dt
        if self.trainable_beta:
            if hasattr(self.net, 'module'):
                sigma = torch.exp(self.net.module.log_betas[indices])
            else:
                sigma = torch.exp(self.net.log_betas[indices])
            if self.trainable_beta == 'fix_last':
                # sigma[indices == self.n_timesteps - 1] = sigma_up[indices == self.n_timesteps-1]
                # to avoid in-place operation
                terminal = (indices == self.n_timesteps - 1).to(x.device)
                sigma = sigma * ~terminal + sigma_up * terminal
            elif self.trainable_beta == 'fix_last3':
                non_terminal = (indices < self.n_timesteps - 3).to(x.device)
                sigma = sigma * non_terminal + sigma_up * (~non_terminal)
            sigma_up = sigma

        samples = mu + torch.randn_like(mu) * append_dims(sigma_up, dims)
        d_step = {'sample': samples,
                "mean": mu, 
                "sigma": sigma_up.clamp(1e-4,None)  # avoid -inf when log
                }
        return d_step 

    def sample(self, n_sample, device, i_class=None, enable_grad=False, x0=None):
        if self.class_cond:
            if i_class is None:
                i_class = torch.randint(0, self.num_classes, (n_sample,), device=device)
            elif isinstance(i_class, int):
                i_class = torch.tensor([i_class] * n_sample, device=device, dtype=torch.long)
            model_kwargs = {"y": i_class}
        else:
            i_class = None
            model_kwargs = {}

        if x0 is None:
            x = torch.randn(n_sample, *self.sample_shape, device=device) * self.sigma_max
        else:
            x = x0.to(device)
        l_x = [x]
        l_mean = []; l_sigma = []
        for i in range(self.n_timesteps):
            with torch.set_grad_enabled(enable_grad):
                d_step = self.sample_step(x, i*torch.ones(len(x), dtype=torch.long), **model_kwargs)
            x = d_step['sample']
            l_x.append(x)
            l_mean.append(d_step['mean'])
            l_sigma.append(d_step['sigma'])

        d_sample = {'sample': l_x[-1], 'l_sample': l_x, 'y': i_class, 'mean': l_mean, 'sigma': l_sigma}
        return d_sample


