"""
Diffusion by MaxEnt IRL for 32x32 images (e.g., CIFAR-10)
"""
import torch
import torch.nn.functional as F
from torch.distributions import Normal
from ..diffusion import extract, make_beta_schedule 
from ..modules import process_single_t


def append_buffer(state_buffer, d_sample):
    x_seq = d_sample["l_sample"]
    n_sample = len(x_seq[0])
    n_seq = len(x_seq) - 1
    # logp_terminal = d_sample["logp_terminal"]
    device = x_seq[0].device

    for t in range(n_seq):
        state_buffer["state"] = torch.cat((state_buffer["state"], x_seq[t].detach()))
        state_buffer["next_state"] = torch.cat(
            (state_buffer["next_state"], x_seq[t + 1].detach())
        )
        state_buffer["timestep"] = torch.cat(
            (state_buffer["timestep"], torch.tensor([t] * n_sample).to(device))
        )
        state_buffer["final"] = torch.cat((state_buffer["final"], x_seq[-1].detach()))
        if 'logp' in d_sample:
            logp = d_sample["logp"]
            state_buffer["logp"] = torch.cat((state_buffer["logp"], logp[t].detach()))
        if 'control' in d_sample:
            state_buffer["control"] = torch.cat((state_buffer["control"], 
                                                 d_sample['control'][t].detach()))
        if 'entropy' in d_sample:
            state_buffer["entropy"] = torch.cat((state_buffer["entropy"], 
                                                 d_sample['entropy'][t].detach()))
        if 'mean' in d_sample:
            state_buffer["mean"] = torch.cat((state_buffer["mean"], 
                                                 d_sample['mean'][t].detach()))
        if 'sigma' in d_sample:
            state_buffer["sigma"] = torch.cat((state_buffer["sigma"], 
                                                 d_sample['sigma'][t].detach()))
        if 'y' in d_sample:
            state_buffer["y"] = torch.cat((state_buffer["y"], d_sample["y"].detach()))
    return state_buffer

def reset_buffer(device):
    state_dict = {}
    state_dict['state'] = torch.FloatTensor().to(device)
    state_dict['next_state'] = torch.FloatTensor().to(device)
    state_dict['timestep'] = torch.LongTensor().to(device)
    state_dict['final'] = torch.FloatTensor().to(device)
    state_dict['logp'] = torch.FloatTensor().to(device)
    state_dict['control'] = torch.FloatTensor().to(device)
    state_dict['entropy'] = torch.FloatTensor().to(device)
    state_dict['mean'] = torch.FloatTensor().to(device)
    state_dict['sigma'] = torch.FloatTensor().to(device)
    state_dict["y"] = torch.LongTensor().to(device)
    return state_dict 



class SFT_PG_Trainer:
    """OOP refactoring of Fan and Lee 2023"""
    def __init__(self, batchsize):
        self.batchsize = batchsize

    def set_models(self, f, v, sampler, optimizer, optimizer_fstar, optimizer_v):
        self.f = f
        self.v = v
        self.sampler = sampler
        self.optimizer = optimizer
        self.optimizer_fstar = optimizer_fstar
        self.optimizer_v = optimizer_v

    def update_f_v(self, img, d_sample, state_dict):
        x_seq = d_sample['l_sample']
        x0 = x_seq[-1]
        n_steps = len(x_seq)-1
        batchsize = self.batchsize
        self.f.train()
        self.v.train()
        # take the last s_0 to compute f; then update v(s_0) - v(s_T)
        output = self.f(torch.cat((img.detach(),x0.detach()),0))
            
        pos_e = output[:x0.shape[0]]
        neg_e = output[x0.shape[0]:]
        d_loss = pos_e.mean()- neg_e.mean()
        
        d_loss.backward()

        permutation = torch.randperm(batchsize * n_steps)
        indices = permutation + (state_dict['state'].shape[0] - (batchsize * n_steps))
        state = state_dict['state'][indices]
        timestep = state_dict['timestep'][indices]
        final = state_dict['final'][indices]
        v_loss = F.mse_loss(self.v(state, timestep), self.f(final))*n_steps
        v_loss.backward()
        self.optimizer_v.step()
        self.optimizer_v.zero_grad()
        self.optimizer_fstar.step()
        self.optimizer_fstar.zero_grad()
        
        d_energy = {
            'ebm/d_loss_': d_loss.item(),
            'ebm/v_loss_': v_loss.item(),
            'ebm/pos_e_': pos_e.mean().item(),
            'ebm/neg_e_': neg_e.mean().item(),
        }
        
        return d_energy

    def update_sampler(self, state_dict, n_generator, d_sample=None):
        permutation = torch.randperm(state_dict['state'].shape[0])
        for m in range(0, self.batchsize*n_generator, self.batchsize):
            self.optimizer.zero_grad()
            indices = permutation[m:m + self.batchsize]
            final = state_dict['final'][indices]
            state = state_dict['state'][indices]
            next_state = state_dict['next_state'][indices]
            timestep = state_dict['timestep'][indices]
            with torch.no_grad():
                adv = (self.f(final)-self.v(state, timestep)).detach().squeeze()
            with torch.enable_grad():
                log_prob_t = self.sampler.log_prob_step(state, next_state, timestep)

                loss = (adv * log_prob_t).mean()
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(self.sampler.net.parameters(), 0.1)
                self.optimizer.step()

        d_sampler = {'sampler/loss_': loss.item(),
                    'sampler/log_prob_t_': log_prob_t.mean().item(),
                    'sampler/adv_': adv.mean().item(),}
        return d_sampler

    def train_one_epoch(self):
        pass

class SFT_PPO_Trainer(SFT_PG_Trainer):
    def __init__(self, batchsize, eps_clip = 0.2):
        self.batchsize = batchsize
        self.eps_clip = eps_clip

    def update_sampler(self, state_dict, n_generator, d_sample=None):
        permutation = torch.randperm(state_dict['state'].shape[0])
        for m in range(0, self.batchsize*n_generator, self.batchsize):
            self.optimizer.zero_grad()
            indices = permutation[m:m + self.batchsize]
            final = state_dict['final'][indices]
            state = state_dict['state'][indices]
            next_state = state_dict['next_state'][indices]
            timestep = state_dict['timestep'][indices]
            with torch.no_grad():
                adv = (self.f(final)-self.v(state, timestep)).detach().squeeze()
            with torch.enable_grad():
                log_prob_t = self.sampler.log_prob_step(state, next_state, timestep)
                ratios = torch.exp(log_prob_t - state_dict['logp'][indices])
                surr1 = ratios * adv
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * adv
                loss = torch.min(surr1, surr2).mean() # sign flip since we want to minimize E_{x_T}[f(x_T)]
                loss.backward()
                self.optimizer.step()

        d_sampler = {'sampler/loss_': loss.item(),
                    'sampler/log_prob_t_': log_prob_t.mean().item(),
                    'sampler/adv_': adv.mean().item(),}
        return d_sampler


class DxMI_Trainer(SFT_PG_Trainer):
    def __init__(self, batchsize, tau1=0., tau2=0., gamma=None, gamma_pos=None,
                 q_beta_schedule='constant', 
                 q_beta_start=1., q_beta_end=1., running_cost='brownian',
                 adavelreg=None,
                 n_timesteps=10, value_update_order='backward', 
                 entropy_in_value=None,
                 velocity_in_value=None,
                 use_sampler_beta=False, aug=None,
                 time_cost=None, time_cost_avr=None, time_cost_sig=None, time_cost_fix=None,
                 time_cost_sig_center=None,
                 value_monotone=None, monotone_margin=None,
                 repeat_value_update=1, skip_running_last=False,
                 value_resample=False,
                 value_grad_clip=False,
                 skip_sampler_tau=0,):
        """
        Soft Value Iteration approach for GCD.

        n_timesteps: number of steps for the diffusion process
        tau1: coefficient for the entropy term
        tau2: coefficient for the velocity term
        gamma: coefficient for the regularization term

        running_cost: 'brownian' or 'rescale'
        value_update_order: 'backward' or 'random'
        entropy_in_value: if False, do not add entropy to value. if integer, for that number of the last
                          steps, do not add entropy to value. Otherwise, add entropy to value
        q_beta_schedule: schedule for the beta of q(x_t|x_{t+1})
        use_sampler_beta: if True, use sampler's beta for q(x_t|x_{t+1})
        adavelreg: None or float. Coefficient for the adaptive velocity regularization term.

        use_aug: use non-leaking augmentation proposed by StudioGAN2-ADA
        value_monotone: enforcing V(x_t, t) = V(x_t, t+1)
        value_resample: if True, during value update, sample again next state from the current policy.
                        corresponds to "SAC" approach
        """
        self.batchsize = batchsize
        self.n_timesteps = n_timesteps
        self.gamma = gamma
        self.gamma_pos = gamma_pos
        self.tau1 = tau1
        self.tau2 = tau2
        self.running_cost = running_cost
        self.value_update_order = value_update_order
        self.entropy_in_value = entropy_in_value
        self.velocity_in_value = velocity_in_value
        self.skip_running_last = skip_running_last
        self.q_beta_schedule = q_beta_schedule
        self.q_beta_start = q_beta_start
        self.q_beta_end = q_beta_end
        self.adavelreg = adavelreg
        
        self.use_sampler_beta = use_sampler_beta
        self.aug = aug 
        self.time_cost = time_cost  # time_cost can be 'avr'
        self.time_cost_avr = time_cost_avr
        self.time_cost_sig = time_cost_sig
        self.time_cost_sig_center = time_cost_sig_center
        self.time_cost_fix = time_cost_fix
        self.value_monotone = value_monotone
        self.monotone_margin = monotone_margin
        self.repeat_value_update = repeat_value_update
        self.value_resample = value_resample
        self.value_grad_clip = value_grad_clip
        self.skip_sampler_tau = skip_sampler_tau


    def set_models(self, f, v, sampler, optimizer, optimizer_fstar, optimizer_v):
        self.f = f
        self.v = v
        self.sampler = sampler
        self.optimizer = optimizer
        self.optimizer_fstar = optimizer_fstar
        self.optimizer_v = optimizer_v   

        if self.use_sampler_beta:
            if hasattr(self.sampler, 'user_defined_eta'):
                self.betas_for_q = torch.tensor(self.sampler.user_defined_eta, dtype=torch.float32)
                # print(user_defined_eta)
                # [1.00000e-04 1.10250e-02 4.00000e-02 8.70250e-02 1.52100e-01 2.35225e-01
                # 3.36400e-01 4.55625e-01 5.92900e-01 7.48225e-01]
            elif hasattr(self.sampler, 'log_betas'):
                self.betas_for_q = torch.exp(self.sampler.log_betas).detach()
            elif hasattr(self.sampler.net, 'log_betas'):
                self.betas_for_q = torch.exp(self.sampler.net.log_betas).detach()
            print(f'betas_for_q: {self.betas_for_q}')
        else:
            self.betas_for_q = make_beta_schedule( # model q(x_t|x_{t+1}) as ddpm style forward process
                schedule=self.q_beta_schedule,
                n_timesteps=self.n_timesteps,
                start=self.q_beta_start,
                end=self.q_beta_end,
            )

    def get_running_cost(self, state, next_state, pred_mean, pred_std, t):
        """compute running cost given transition"""
        if self.running_cost == 'rescale': # model q(x_t|x_{t+1}) as ddpm style forward process
            t_reversed = (self.n_timesteps - t - 1)
            beta_next = extract(self.betas_for_q, t_reversed, state).to(state.device)
            running_cost = ((torch.sqrt(1-beta_next)*next_state - state) ** 2)/ (2 * beta_next)
            running_cost = running_cost.view(len(state), -1).mean(dim=1)
        elif self.running_cost == 'brownian':
            t_reversed = (self.n_timesteps - t - 1)
            beta_next = extract(self.betas_for_q, t_reversed, state).to(state.device)
            running_cost = ((next_state - state) ** 2)/ (2 * beta_next)
            running_cost = running_cost.view(len(state), -1).mean(dim=1)
        elif self.running_cost == 'rescale_no_var':
            t_reversed = (self.n_timesteps - t - 1)
            beta_next = extract(self.betas_for_q, t_reversed, state).to(state.device)
            running_cost = ((torch.sqrt(1-beta_next)*pred_mean - state) ** 2)/ (2 * beta_next)
            running_cost = running_cost.view(len(state), -1).mean(dim=1)
            running_cost += (((1-beta_next) * pred_std ** 2 / (2 * beta_next))).squeeze()
        elif self.running_cost == 'brownian_no_var':
            t_reversed = (self.n_timesteps - t - 1)
            beta_next = extract(self.betas_for_q, t_reversed, state).to(state.device)
            running_cost = ((pred_mean - state) ** 2)/ (2 * beta_next)
            running_cost = running_cost.view(len(state), -1).mean(dim=1)
            running_cost += ((pred_std ** 2 / (2 * beta_next))).squeeze()
        else:
            running_cost = torch.tensor(0.0)
        return running_cost

    def sample_guidance(self, n_sample, device, x0=None, guidance_scale=None, t_select=None):
        """sample guidance from the value function"""
        self.v.eval()
        if x0 is None:
            x0 = torch.randn(n_sample, *self.sampler.sample_shape).to(device)
        if guidance_scale is None:
            guidance_scale = self.guidance_scale
        l_x = [x0.detach().clone()]
        l_guidance = []
        l_logp = []  # logp of guided samples from the guided policy
        l_logp_orig = []  # logp of guided samples from the original policy
        l_control = []
        x = x0

        for t in range(self.n_timesteps):
            tt = process_single_t(x, t)
            with torch.no_grad():
                d_step = self.sampler.sample_step(x, tt)
            next_x = d_step['sample'].detach()
            with torch.enable_grad():
                next_x = next_x.requires_grad_(True)
                # DDP does not support autograd.grad. So we use v.module.
                if isinstance(self.v, torch.nn.parallel.DistributedDataParallel):
                    value = self.v.module(next_x, tt+1).squeeze()
                else:
                    value = self.v(next_x, tt+1).squeeze()
                grad = torch.autograd.grad(value.sum(), next_x)[0]
            guidance = grad * guidance_scale * d_step['sigma']
            logp = d_step['logp']  # logp is the same after guidance
            if t_select is None or t in t_select:
                x = next_x + guidance
            else:
                x = next_x
            orig_dist = Normal(d_step['mean'], d_step['sigma'])
            logp_orig = orig_dist.log_prob(x).mean(-1).mean(-1).mean(-1)
            l_guidance.append(guidance)
            l_logp.append(logp)
            l_logp_orig.append(logp_orig)
            l_control.append(d_step['control'])
            l_x.append(x.detach().clone())

        d_sample = {'sample': x, 'l_sample': l_x,
                    'logp': l_logp, 'logp_on': l_logp_orig,
                    'logp_traj': torch.stack(l_logp).sum(dim=0), 
                    'logp_on_traj': torch.stack(l_logp_orig).sum(dim=0),
                    'guidance': l_guidance,
                    'control': l_control}
        return d_sample
    
    def update_adaptive_vel_reg(self, d_sample):
        """
        update betas_for_q
        """
        device = d_sample['sample'].device
        samples = torch.stack(d_sample['l_sample'])  # (n_timesteps, n_sample, C, H, W)
        # from t=0 to t=T-1
        diff = (samples[1:] - samples[:-1]) ** 2
        diff = diff.view(diff.shape[0], -1).mean(dim=1).flip(0).to(device)

        self.betas_for_q = (self.betas_for_q.to(device) * self.adavelreg + (1 - self.adavelreg) * diff).detach()

    def update_f_v(self, img, d_sample, state_dict):
        x_seq = d_sample['l_sample']
        if self.adavelreg is not None:
            self.update_adaptive_vel_reg(d_sample)

        self.optimizer_v.zero_grad()
        x0 = x_seq[-1]
        n_steps = self.n_timesteps
        batchsize = self.batchsize
        device = img.device
        if self.f is not None:
            self.f.train()
        self.v.train()
        # treat the last step of value function as energy
        T = (n_steps) * torch.ones(len(img) + len(x0), dtype=torch.long).to(device)
        inputs = torch.cat((img.detach(), x0.detach()),0)
        if self.aug is not None:
            inputs = self.aug(inputs)

        if self.f is None:
            output = self.v(inputs, T)
        else:
            output = self.f(inputs)
            
        pos_e = output[:x0.shape[0]]
        neg_e = output[x0.shape[0]:]
        d_loss = pos_e.mean()- neg_e.mean()
        if self.gamma is not None:
            reg = pos_e.pow(2).mean() + neg_e.pow(2).mean()
            d_loss += self.gamma * reg
        elif self.gamma_pos is not None:
            reg = pos_e.pow(2).mean()
            d_loss += self.gamma_pos * reg
        else:
            reg = torch.tensor(0)
        
        d_loss.backward()
        if self.f is None:
            self.optimizer_v.step()
            self.optimizer_v.zero_grad()
        else:
            self.optimizer_fstar.step()
            self.optimizer_fstar.zero_grad()
            self.f.eval()

        # temporal difference value estimation
        permutation = torch.randperm(batchsize * n_steps)
        indices = permutation + (state_dict['state'].shape[0] - (batchsize * n_steps))
        d_running_cost = {}
        d_value = {}
        for _ in range(self.repeat_value_update):
            if self.value_update_order == 'random':
                update_order = torch.randperm(n_steps)

            for i in range(n_steps):
                if self.value_update_order == 'random':
                    update_t = update_order[i]
                else:
                    update_t = (n_steps - i - 1)
                if self.value_update_order == 'shuffle':
                    train_indices = torch.arange(batchsize) + i * batchsize
                else:
                    train_indices = torch.nonzero(state_dict["timestep"][indices] == update_t).flatten()
                state = state_dict["state"][indices][train_indices]
                timestep = state_dict["timestep"][indices][train_indices]
                if self.value_resample:
                    d_sample_step = self.sampler.sample_step(state, timestep)
                    next_state = d_sample_step['sample']
                    pred_mean = d_sample_step['mean']
                    pred_std = d_sample_step['sigma']
                else:
                    next_state = state_dict["next_state"][indices][train_indices]
                    pred_mean = state_dict["mean"][indices][train_indices]
                    pred_std = state_dict["sigma"][indices][train_indices] # (bs, 1, 1, 1)
                running_cost = self.get_running_cost(state, next_state, pred_mean, pred_std, timestep)
                entropy = torch.log(pred_std.squeeze())

                self.v.eval()
                if self.aug is not None:
                    state = self.aug(state)
                    next_state = self.aug(next_state)
                if i == n_steps - 1 and self.f is not None:  
                    # use energy for the last step
                    v_xtp1 = self.f(next_state).squeeze()
                    target = v_xtp1 + running_cost * self.tau2
                else:
                    v_xtp1 = self.v(next_state, timestep+1).squeeze()
                    # if self.skip_running_last:
                    #     assert isinstance(self.skip_running_last, int), 'skip_running_last should be integer'
                    #     non_terminal = (timestep < n_steps - self.skip_running_last).float() 
                    #     target = v_xtp1 + running_cost * self.tau2 * non_terminal
                    # else:
                    #     target = v_xtp1 + running_cost * self.tau2 
                target = v_xtp1

                if self.velocity_in_value is not None:
                    non_terminal = (timestep < n_steps - self.velocity_in_value).float()
                    target += running_cost * self.tau2 * non_terminal

                if self.time_cost_sig is not None:
                    if self.time_cost_sig_center is None:
                        center = self.n_timesteps // 2
                    else:
                        center = self.time_cost_sig_center

                    target = target + self.time_cost_sig * torch.sigmoid(-timestep + center) \
                            - self.time_cost_sig * torch.sigmoid(-timestep - 1 + center)
                if self.time_cost_fix is not None:
                    non_terminal = (timestep < n_steps - self.time_cost_fix).float() 
                    target = target - 0.1 * torch.log(self.sampler.std[timestep]) * non_terminal

                if self.time_cost_avr is not None:
                    target = target + self.time_cost_avr * self.betas_for_q[i]  # betas_for_q accepts t_reversed
                if self.time_cost is not None:
                    target = target + self.time_cost
                if self.entropy_in_value or self.entropy_in_value == 0:
                    # do not add entropy to target for the "entropy_in_value" number of last steps 
                    assert isinstance(self.entropy_in_value, int), "self.entropy_in_value should be interger"
                    non_terminal = (timestep < n_steps - self.entropy_in_value).float() 
                    target -= entropy * self.tau1 * non_terminal 
                self.v.train()
                v_xt = self.v(state, timestep).squeeze()
                v_loss = F.mse_loss(v_xt, target.detach())
                if self.value_monotone is not None:
                    assert self.f is None, "not implemented yet for energy-value separation case"
                    # consistency_loss = F.mse_loss(v_xt.detach(), self.v(state, timestep+1).squeeze())
                    # monotonicity loss
                    # margin = 0.1
                    consistency_loss = F.relu(v_xtp1.detach() - v_xt + self.monotone_margin).mean()
                    v_loss += consistency_loss * self.value_monotone
                v_loss.backward()
                if self.value_grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.v.parameters(), 0.1)
                self.optimizer_v.step()
                self.optimizer_v.zero_grad()

                d_running_cost[f'running_cost/step_{update_t}_'] = running_cost.mean().item()
                d_value[f"value/step_{update_t}_"] = v_xt.mean().item()

        d_energy = {
            'ebm/d_loss_': d_loss.item(),
            'ebm/v_loss_': v_loss.item(),
            'ebm/pos_e_': pos_e.mean().item(),
            'ebm/neg_e_': neg_e.mean().item(),
            'ebm/running_cost_': running_cost.mean().item(),
            'ebm/reg_': reg.item(),
        }
        d_energy.update(d_running_cost)
        d_energy.update(d_value)

        if self.adavelreg is not None:
            for t, beta in enumerate(self.betas_for_q):  # t=0 is data
                d_energy[f'adavelreg/beta{t}_'] = beta.item()
        if self.value_monotone is not None:
            d_energy['ebm/v_monotone_'] = monotone_loss.item()
        
        return d_energy
    
    def update_sampler(self, state_dict, n_generator, d_sample=None):
        if self.f is not None:
            self.f.eval()
        self.v.eval()
        self.sampler.train()
        permutation = torch.randperm(state_dict["state"].shape[0])
        batchsize = self.batchsize
        n_data = min(len(permutation), batchsize * n_generator)
        for m in range(0, n_data, batchsize):
            self.optimizer.zero_grad()
            indices = permutation[m : m + batchsize]
            state = state_dict["state"][indices]
            t = state_dict["timestep"][indices]
            d_sample_step = self.sampler.sample_step(state, t)
            next_state = d_sample_step["sample"]
            pred_mean = d_sample_step["mean"]
            pred_std = d_sample_step["sigma"] # (bs, 1, 1, 1)
            running_cost = self.get_running_cost(state, next_state, pred_mean, pred_std, t)
            # skip running cost during sampler update -- seems like it hurts the performance
            # if self.skip_running_last:
            #     # do not apply running cost at last step
            #     non_terminal = (t < self.n_timesteps - self.skip_running_last).float() 
            #     running_cost = running_cost * non_terminal
            # running_cost = running_cost.mean()

            causal_entropy = torch.log(pred_std.squeeze())
            # causal_entropy = d_sample_step["entropy"].mean()
            if self.aug is not None:
                next_state = self.aug(next_state)
            if self.f is None:
                sampler_value_loss = self.v(next_state, t + 1).squeeze()
            else:
                if (t == self.n_timesteps - 1).sum() == 0:
                    f_state = torch.tensor([])
                    v_state = self.v(next_state, t+1).flatten()
                elif (t != self.n_timesteps - 1).sum() == 0:
                    f_state = self.f(next_state).flatten()
                    v_state = torch.tensor([])
                else:
                    f_state = self.f(next_state[t == self.n_timesteps - 1]).flatten()
                    v_state = self.v(next_state[t != self.n_timesteps - 1], t[t!= self.n_timesteps-1]+1).flatten()
                sampler_value_loss = torch.cat((f_state, v_state)).mean()

            non_terminal = (t < self.n_timesteps - self.skip_sampler_tau).float()
            sampler_loss = (
                sampler_value_loss
                + (running_cost * self.tau2 - causal_entropy * self.tau1) * non_terminal
            ).mean()
            sampler_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.sampler.parameters(), 0.1)
            self.optimizer.step()

        d_train_sample = {
            "sampler/sampler_loss_": sampler_loss.item(),
            "sampler/sampler_value_loss_": sampler_value_loss.mean().item(),
            "sampler/running_cost_": running_cost.mean().item(),
            "sampler/causal_entropy_": causal_entropy.mean().item(),
        }

        if self.sampler.trainable_beta:
            if hasattr(self.sampler.net, 'module'):
                sigma = torch.exp(self.sampler.net.module.log_betas)
            else:
                sigma = torch.exp(self.sampler.net.log_betas)

            for t in range(len(sigma)):
                d_train_sample["sigma/sigma_{}_".format(t)] = sigma[t].item()               

        
        return d_train_sample