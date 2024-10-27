"""
adopted and refactored from 
https://github.com/UW-Madison-Lee-Lab/SFT-PG/blob/main/finetune.py
"""
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from ..modules import process_single_t

unsqueeze3x = lambda x: x[..., None, None, None]

diffusion_config = {
    "beta_0": 0.0001,
    "beta_T": 0.02,
    "T": 1000,
}

def calc_diffusion_hyperparams(T, beta_0, beta_T):
    """
    Compute diffusion process hyperparameters

    Parameters:
    T (int):                    number of diffusion steps
    beta_0 and beta_T (float):  beta schedule start/end value, 
                                where any beta_t in the middle is linearly interpolated
    
    Returns:
    a dictionary of diffusion hyperparameters including:
        T (int), Beta/Alpha/Alpha_bar/Sigma (torch.tensor on cpu, shape=(T, ))
    """

    Beta = torch.linspace(beta_0, beta_T, T)
    Alpha = 1 - Beta
    Alpha_bar = Alpha + 0
    Beta_tilde = Beta + 0
    for t in range(1, T):
        Alpha_bar[t] *= Alpha_bar[t-1]
        Beta_tilde[t] *= (1-Alpha_bar[t-1]) / (1-Alpha_bar[t])
    Sigma = torch.sqrt(Beta_tilde)
    
    _dh = {}
    _dh["T"], _dh["Beta"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"] = T, Beta, Alpha, Alpha_bar, Sigma
    diffusion_hyperparams = _dh
    return diffusion_hyperparams

def bisearch(f, domain, target, eps=1e-8):
    """
    find smallest x such that f(x) > target

    Parameters:
    f (function):               function
    domain (tuple):             x in (left, right)
    target (float):             target value
    
    Returns:
    x (float)
    """
    # 
    sign = -1 if target < 0 else 1
    left, right = domain
    for _ in range(1000):
        x = (left + right) / 2 
        if f(x) < target:
            right = x
        elif f(x) > (1 + sign * eps) * target:
            left = x
        else:
            break
    return x


def get_VAR_noise(S, schedule='linear'):
    """
    Compute VAR noise levels

    Parameters:
    S (int):            approximante diffusion process length
    schedule (str):     linear or quadratic
    
    Returns:
    np array of noise levels, size = (S, )
    """
    target = np.prod(1 - np.linspace(diffusion_config["beta_0"], diffusion_config["beta_T"], diffusion_config["T"])) # target = alpha_T_bar

    if schedule == 'linear':
        g = lambda x: np.linspace(diffusion_config["beta_0"], x, S)
        domain = (diffusion_config["beta_0"], 0.99)
    elif schedule == 'quadratic':
        g = lambda x: np.array([diffusion_config["beta_0"] * (1+i*x) ** 2 for i in range(S)])
        domain = (0.0, 0.95 / np.sqrt(diffusion_config["beta_0"]) / S)
    else:
        raise NotImplementedError

    f = lambda x: np.prod(1 - g(x))
    largest_var = bisearch(f, domain, target, eps=1e-4)
    return g(largest_var)


def _log_gamma(x):
    # Gamma(x+1) ~= sqrt(2\pi x) * (x/e)^x  (1 + 1 / 12x)
    y = x - 1
    return np.log(2 * np.pi * y) / 2 + y * (np.log(y) - 1) + np.log(1 + 1 / (12 * y))


def _log_cont_noise(t, beta_0, beta_T, T):
    # We want log_cont_noise(t, beta_0, beta_T, T) ~= np.log(Alpha_bar[-1].numpy())
    delta_beta = (beta_T - beta_0) / (T - 1)
    _c = (1.0 - beta_0) / delta_beta
    t_1 = t + 1
    return t_1 * np.log(delta_beta) + _log_gamma(_c + 1) - _log_gamma(_c - t_1 + 1)


# VAR
def _precompute_VAR_steps(diffusion_hyperparams, user_defined_eta, device=None):
    _dh = diffusion_hyperparams
    T, Alpha, Alpha_bar, Beta = _dh["T"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Beta"]
    assert len(Alpha_bar) == T

    # compute diffusion hyperparameters for user defined noise
    T_user = len(user_defined_eta)
    Beta_tilde = torch.from_numpy(user_defined_eta).to(torch.float32).to(device)

    Gamma_bar = 1 - Beta_tilde
    for t in range(1, T_user):
        Gamma_bar[t] *= Gamma_bar[t-1]

    assert Gamma_bar[0] <= Alpha_bar[0] and Gamma_bar[-1] >= Alpha_bar[-1]
    
    continuous_steps = []
    with torch.no_grad():
        for t in range(T_user-1, -1, -1):
            t_adapted = None
            for i in range(T - 1):
                if Alpha_bar[i] >= Gamma_bar[t] > Alpha_bar[i+1]:
                    t_adapted = bisearch(f=lambda _t: _log_cont_noise(_t, Beta[0].cpu().numpy(), Beta[-1].cpu().numpy(), T), 
                                            domain=(i-0.01, i+1.01), 
                                            target=np.log(Gamma_bar[t].cpu().numpy()))
                    break
            if t_adapted is None:
                t_adapted = T - 1
            continuous_steps.append(t_adapted)  # must be decreasing
    return continuous_steps


def VAR_get_params(diffusion_hyperparams, user_defined_eta, kappa, continuous_steps):
    """modified to remove map_gpu"""

    _dh = diffusion_hyperparams
    T, Alpha, Alpha_bar, Beta = _dh["T"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Beta"]
    assert len(Alpha_bar) == T
    assert 0.0 <= kappa <= 1.0

    # compute diffusion hyperparameters for user defined noise
    T_user = len(user_defined_eta)
    Beta_tilde = torch.from_numpy(user_defined_eta).to(torch.float32)
    Gamma_bar = 1 - Beta_tilde
    for t in range(1, T_user):
        Gamma_bar[t] *= Gamma_bar[t-1]

    assert Gamma_bar[0] <= Alpha_bar[0] and Gamma_bar[-1] >= Alpha_bar[-1]
    
    x_prev_multiplier = torch.zeros(T_user)
    theta_multiplier = torch.zeros(T_user)
    std = torch.zeros(T_user)
    diffusion_steps_list = torch.zeros(T_user)


    for i, tau in enumerate(continuous_steps):
        diffusion_steps_list[i] = tau
        if i == T_user - 1:  # the next step is to generate x_0
            assert abs(tau) < 0.1
            alpha_next = torch.tensor(1.0) 
            sigma = torch.tensor(0.0) 
        else:
            alpha_next = Gamma_bar[T_user-1-i - 1]
            sigma = kappa * torch.sqrt((1-alpha_next) / (1-Gamma_bar[T_user-1-i]) * (1 - Gamma_bar[T_user-1-i] / alpha_next))
        x_prev_multiplier[i] = torch.sqrt(alpha_next / Gamma_bar[T_user-1-i])
        theta_multiplier[i] = torch.sqrt(1 - alpha_next - sigma ** 2) - torch.sqrt(1 - Gamma_bar[T_user-1-i]) * torch.sqrt(alpha_next / Gamma_bar[T_user-1-i]) 
        if i == T_user - 1: 
            std[i] = 0.001
        else:
            std[i] = sigma

    # return map_gpu(x_prev_multiplier), map_gpu(theta_multiplier), map_gpu(std), map_gpu(diffusion_steps_list)
    return x_prev_multiplier, theta_multiplier, std, diffusion_steps_list


def VAR_log_prob(net, x_prev, x_next, t, x_prev_multiplier, theta_multiplier, std, diffusion_steps_list):
    # net.eval()
    # net.train()
    diffusion_steps = diffusion_steps_list[t] # shape ([bs])
    epsilon_theta = net(x_prev, diffusion_steps)
    # epsilon_theta_seq = net(torch.cat(x_seq[:10]), diffusion_steps)
    pred_mean = x_prev*unsqueeze3x(x_prev_multiplier[t]) + unsqueeze3x(theta_multiplier[t])*epsilon_theta 
    pred_std = unsqueeze3x(std[t])
    dist = Normal(pred_mean, pred_std)
    log_prob = dist.log_prob(x_next.detach()).mean(dim = -1).mean(dim = -1).mean(dim = -1)

    return log_prob



def VAR_sampling(net, size, diffusion_hyperparams, user_defined_eta, 
                    kappa, continuous_steps, device, trainable_beta = False,
                    enable_grad=False, adhoc_scale1=1):
    """
    Copy for not breaking other functions..

    Perform the complete sampling step according to user defined variances

    Parameters:
    net (torch network):            the model
    size (tuple):                   size of tensor to be generated, 
                                    usually is (number of audios to generate, channels=1, length of audio)
    diffusion_hyperparams (dict):   dictionary of diffusion hyperparameters returned by calc_diffusion_hyperparams
                                    note, the tensors need to be cuda tensors
    user_defined_eta (np.array):    User defined noise       
    kappa (float):                  factor multipled over sigma, between 0 and 1
    continuous_steps (list):        continuous steps computed from user_defined_eta

    Returns:
    the generated images in torch.tensor, shape=size
    """
    # net.eval()
    _dh = diffusion_hyperparams
    T, Alpha, Alpha_bar, Beta = _dh["T"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Beta"]
    assert len(Alpha_bar) == T
    assert len(size) == 4
    assert 0.0 <= kappa <= 1.0

    # compute diffusion hyperparameters for user defined noise
    T_user = len(user_defined_eta)
    Beta_tilde = torch.from_numpy(user_defined_eta).to(torch.float32)
    Gamma_bar = 1 - Beta_tilde
    for t in range(1, T_user):
        Gamma_bar[t] *= Gamma_bar[t-1]

    assert Gamma_bar[0] <= Alpha_bar[0] and Gamma_bar[-1] >= Alpha_bar[-1]
    

    x = torch.randn(size, device=device)
    x_seq = [x.detach().clone()]
    log_prob_list = []
    control_list = []
    pred_mean_list = []
    pred_std_list = []

    with torch.set_grad_enabled(enable_grad):
        for i, tau in enumerate(continuous_steps):
            diffusion_steps = tau * (torch.ones(size[0], device=device))

            epsilon_theta = net(x, diffusion_steps)

            if i == T_user - 1:  # the next step is to generate x_0
                assert abs(tau) < 0.1
                alpha_next = torch.tensor(1.0) 
                sigma = torch.tensor(0.0) 
            else:
                alpha_next = Gamma_bar[T_user-1-i - 1]
                sigma = kappa * torch.sqrt((1-alpha_next) / (1-Gamma_bar[T_user-1-i]) * (1 - Gamma_bar[T_user-1-i] / alpha_next))
            x *= torch.sqrt(alpha_next / Gamma_bar[T_user-1-i]) # x_prev multiplier
            c = torch.sqrt(1 - alpha_next - sigma ** 2) - torch.sqrt(1 - Gamma_bar[T_user-1-i]) * torch.sqrt(alpha_next / Gamma_bar[T_user-1-i]) # theta multiplier
            control = c * epsilon_theta * adhoc_scale1
            pred_mean = x + control

            # recompute beta
            if trainable_beta:
                if trainable_beta == 'fix_last':
                    if hasattr(net, 'module'):
                        log_betas_all = torch.cat([net.module.log_betas[:-1], net.module.std[-1].log().unsqueeze(0)])
                    else:
                        log_betas_all = torch.cat([net.log_betas[:-1], net.std[-1].log().unsqueeze(0)])
                    sigma = torch.exp(log_betas_all[i])

                else:
                    if hasattr(net, 'module'):
                        sigma = torch.exp(net.module.log_betas[i])
                    else:
                        sigma = torch.exp(net.log_betas[i])
            else:            
                if i == T_user - 1:
                    sigma = torch.tensor(0.001).to(device)

            x += control + sigma * torch.randn(size, device=device)

            pred_std = (unsqueeze3x(sigma.repeat(len(x))).to(device))
            dist = Normal(pred_mean, pred_std)
            log_prob = dist.log_prob(x.detach().clone()).mean(dim = -1).mean(dim = -1).mean(dim = -1)

            x_seq.append(x.detach().clone())
            log_prob_list.append(log_prob)
            control_list.append(control.detach().clone())
            pred_mean_list.append(pred_mean.detach().clone())
            pred_std_list.append(pred_std.detach().clone())

    return x_seq, log_prob_list, control_list, pred_mean_list, pred_std_list


class VARSampler(nn.Module):
    def __init__(self, net, n_timesteps, sample_shape, 
            trainable_beta=True, adhoc_scale1=1., adhoc_scale2=1.):
        """
        trainable_beta: Bool or String. trainable diffusion coefficient (noise) 
        adhoc_scale: adhoc scaling of theta_multiplier and sigma. used in T=4. default is 1.
        """
        super().__init__()
        self.net = net
        self.n_timesteps = n_timesteps  # corresponds to T_user
        self.sample_shape = sample_shape
        self.adhoc_scale1 = adhoc_scale1
        self.adhoc_scale2 = adhoc_scale2
        self.trainable_beta = trainable_beta
        assert trainable_beta in {True, False, 'fix_last'}
        self.init_schedule()

        x_prev_multiplier, theta_multiplier, std, diffusion_steps_list = VAR_get_params( 
                self.diffusion_hyperparams, self.user_defined_eta, self.kappa, self.continuous_steps)
        self.register_buffer("x_prev_multiplier", x_prev_multiplier)
        self.register_buffer("theta_multiplier", theta_multiplier)
        self.register_buffer("std", std)
        self.register_buffer("diffusion_steps_list", diffusion_steps_list)
        if self.trainable_beta == 'fix_last':
            self.net.register_buffer("std", std)

    def init_schedule(self):
        schedule = 'quadratic'
        diffusion_hyperparams = calc_diffusion_hyperparams(**diffusion_config)
        self.diffusion_hyperparams = diffusion_hyperparams
        self.kappa = 1.0
        self.user_defined_eta = get_VAR_noise(self.n_timesteps, schedule)
        continuous_steps = torch.tensor(_precompute_VAR_steps(diffusion_hyperparams, self.user_defined_eta))
        self.register_buffer("continuous_steps", continuous_steps)

        Alpha_bar = diffusion_hyperparams["Alpha_bar"]
        Beta_tilde = torch.from_numpy(self.user_defined_eta).to(torch.float32)
        Gamma_bar = 1 - Beta_tilde
        for t in range(1, self.n_timesteps):
            Gamma_bar[t] *= Gamma_bar[t-1]

        assert Gamma_bar[0] <= Alpha_bar[0] and Gamma_bar[-1] >= Alpha_bar[-1]
        self.register_buffer("Gamma_bar", Gamma_bar)

        l_sigma = []
        for t in range(self.n_timesteps):
            if t == self.n_timesteps - 1:
                sigma = torch.tensor(0.001)
            else:
                alpha_next = self.Gamma_bar[self.n_timesteps-1-t - 1]
                sigma = self.kappa * torch.sqrt((1-alpha_next) / (1-self.Gamma_bar[self.n_timesteps-1-t]) * (1 - self.Gamma_bar[self.n_timesteps-1-t] / alpha_next))
            l_sigma.append(sigma)
        sigmas = torch.stack(l_sigma, dim=0)

        if self.trainable_beta:
            self.net.log_betas = nn.Parameter(torch.log(sigmas * self.adhoc_scale2))  # it's actually log sigma

    def sample_step(self, x, t, y=None):
        """
        assume t is a scalar
        """
        device = x.device
        t = process_single_t(x, t)  # t is now 1D tensor
        diffusion_steps = self.continuous_steps[t]
        is_last_t = t == self.n_timesteps - 1
        
        epsilon_theta = self.net(x, diffusion_steps)
        alpha_next = self.Gamma_bar[self.n_timesteps-1-t - 1]
        alpha_next = alpha_next * (~is_last_t) + (is_last_t) * 1.0
        sigma = self.kappa * torch.sqrt((1-alpha_next) / (1-self.Gamma_bar[self.n_timesteps-1-t]) * (1 - self.Gamma_bar[self.n_timesteps-1-t] / alpha_next))
        sigma = sigma * (~is_last_t) + (is_last_t) * 0  # last step is assigned later
        sigma = sigma.to(device)

        x_mult = torch.sqrt(alpha_next / self.Gamma_bar[self.n_timesteps-1-t]) # x_prev multiplier
        x_mult = x_mult.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        x = x * x_mult
        c = torch.sqrt(1 - alpha_next - sigma ** 2) - torch.sqrt(1 - self.Gamma_bar[self.n_timesteps-1-t]) * torch.sqrt(alpha_next / self.Gamma_bar[self.n_timesteps-1-t]) # theta multiplier
        c = c.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        control = c * epsilon_theta * self.adhoc_scale1

        pred_mean = x + control

        # recompute sigma
        if self.trainable_beta:
            if self.trainable_beta == 'fix_last':
                if hasattr(self.net, 'module'):
                    log_betas_all = torch.cat([self.net.module.log_betas[:-1], self.net.module.std[-1].log().unsqueeze(0)])
                else:
                    log_betas_all = torch.cat([self.net.log_betas[:-1], self.net.std[-1].log().unsqueeze(0)])
                sigma = torch.exp(log_betas_all[t])
            else:
                if hasattr(self.net, 'module'):
                    sigma = torch.exp(self.net.module.log_betas[t])
                else:
                    sigma = torch.exp(self.net.log_betas[t])
        else:
            sigma = sigma * (~is_last_t) + (is_last_t) * 0.001
        sigma = sigma.unsqueeze(1).unsqueeze(2).unsqueeze(3)

        x = pred_mean + sigma * torch.randn_like(x)

        dist = Normal(pred_mean, sigma)
        log_prob = dist.log_prob(x.detach().clone()).mean(-1).mean(-1).mean(-1)

        entropy = torch.log(sigma)  # normalize by dimensionality for convenience
        d_step = {"sample": x, "logp": log_prob, "logp_terminal": torch.zeros(len(x), device=device),
                  "mean": pred_mean, "sigma": sigma,
                  "entropy": entropy, "control": control}
        return d_step
    

    def sample(self, n_sample, device="cpu", enable_grad=False):
        size = (n_sample, *self.sample_shape)
        samples, log_prob_list, control_list, pred_mean_list, pred_std_list = VAR_sampling(self.net, size,
                self.diffusion_hyperparams, self.user_defined_eta, 
                self.kappa, self.continuous_steps, device=device, 
                trainable_beta=self.trainable_beta, enable_grad=enable_grad,
                adhoc_scale1=self.adhoc_scale1)
        x = samples[-1]

        logp_terminal = torch.zeros(len(x), device=device)
        d_sample = {'sample': x,
                    'l_sample': samples,
                    'logp': log_prob_list,
                    'logp_terminal': logp_terminal,
                    'mean': pred_mean_list,
                    'sigma': pred_std_list,
                    'control': control_list}
        return d_sample


    def log_prob_step(self, x_prev, x_next, t):

        x_prev_multiplier, theta_multiplier, std, diffusion_steps_list = VAR_get_params( 
                self.diffusion_hyperparams, self.user_defined_eta, self.kappa, self.continuous_steps)
        device = x_prev.device
        x_prev_multiplier = x_prev_multiplier.to(device)
        theta_multiplier = theta_multiplier.to(device)
        diffusion_steps_list = diffusion_steps_list.to(device)
        std = std.to(device)

        log_prob = VAR_log_prob(self.net, x_prev, x_next, t, 
                x_prev_multiplier, theta_multiplier, 
                std, diffusion_steps_list)
        return log_prob