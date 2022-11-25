import math
import torch
from torch import nn
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm
import copy
import light_model.sr3_modules.perceptual as perceptual
loss_func = nn.L1Loss(reduction='sum')
layer_indexs = [14]
def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) /
            n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas


# gaussian diffusion trainer class

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        image_size,
        channels=3,
        loss_type='l1',
        is_leader = False,
        conditional=True,
        schedule_opt=None
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn
        self.loss_type = loss_type
        self.is_leader = is_leader
        self.conditional = conditional
        self.device = torch.device('cuda')
        self.perceptual_loss = perceptual.PerceptualLoss(loss_func,layer_indexs,self.device)
        if schedule_opt is not None:
            pass
            # self.set_new_noise_schedule(schedule_opt)

    def set_loss(self, device):
        self.feature_loss = nn.MSELoss(reduction='sum').to(device)
        if self.loss_type == 'l1':
            self.loss_func = nn.L1Loss(reduction='sum').to(device)
        elif self.loss_type == 'l2':
            self.loss_func = nn.MSELoss(reduction='sum').to(device)
        else:
            raise NotImplementedError()

    def set_new_noise_schedule(self, schedule_opt, device):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)

        betas = make_beta_schedule(
            schedule=schedule_opt['schedule'],
            n_timestep=schedule_opt['n_timestep'],
            linear_start=schedule_opt['linear_start'],
            linear_end=schedule_opt['linear_end'])
        betas = betas.detach().cpu().numpy() if isinstance(
            betas, torch.Tensor) else betas
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod_prev = np.sqrt(
            np.append(1., alphas_cumprod))

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev',
                             to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod',
                             to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod - 1)))
        self.register_buffer('test_sqrt_recip_alphas', to_torch(
            np.sqrt(1. / alphas)))
        self.register_buffer('test_sqrt_recipm1_alphas',
                             to_torch(np.sqrt(1. / alphas - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * \
            (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance',
                             to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(
            np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))


    # calc ddim alpha
    def compute_alpha(self, beta, t):
        beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
        a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
        return a

    def slerp(self, z1, z2, alpha):
        theta = torch.acos(torch.sum(z1 * z2) / (torch.norm(z1) * torch.norm(z2)))
        return (
                torch.sin((1 - alpha) * theta) / torch.sin(theta) * z1
                + torch.sin(alpha * theta) / torch.sin(theta) * z2
        )

    def predict_start_from_noise(self, x_t, t, noise):
        return self.sqrt_recip_alphas_cumprod[t] * x_t - \
            self.sqrt_recipm1_alphas_cumprod[t] * noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = self.posterior_mean_coef1[t] * \
            x_start + self.posterior_mean_coef2[t] * x_t
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_log_variance_clipped

    def compute_x_next(self, x, t, condition_x=None):
        batch_size = x.shape[0]
        noise_level = torch.FloatTensor(
            [self.sqrt_alphas_cumprod_prev[t+1]]).repeat(batch_size, 1).to(x.device)
        if condition_x is not None:
            noise, feature = self.denoise_fn(torch.cat([condition_x, x], dim=1), noise_level)
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=noise)
        else:
            noise, feature = self.denoise_fn(x, noise_level)
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=noise)

        x_next =x*self.test_sqrt_recip_alphas[t]-x_recon*self.test_sqrt_recipm1_alphas[t]
        return x_next

    def p_mean_variance(self, x, t, clip_denoised: bool, condition_x=None):
        batch_size = x.shape[0]
        noise_level = torch.FloatTensor(
            [self.sqrt_alphas_cumprod_prev[t+1]]).repeat(batch_size, 1).to(x.device)
        if condition_x is not None:
            noise,feature = self.denoise_fn(torch.cat([condition_x, x], dim=1), noise_level)
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=noise)
        else:
            noise, feature = self.denoise_fn(x, noise_level)
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=noise)

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, condition_x=None):
        model_mean, model_log_variance = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, condition_x=condition_x)
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        return model_mean + noise * (0.5 * model_log_variance).exp()

    @torch.no_grad()
    def p_sample_loop(self, x_in, hr_in = None,continous=False,condition_ddim = False,steps = 2000,eta = 1.0):
        device = self.betas.device
        if condition_ddim:
            timesteps = steps
            ddim_eta = eta
            alpha = 1.0

            sample_inter = (1 | (timesteps // 10))

            x = copy.deepcopy(x_in)

            ret_img = hr_in
            ret_img = torch.cat([ret_img, x_in], dim=0)

            skip = self.num_timesteps // timesteps
            seq = range(0, self.num_timesteps, skip)
            seq_next = [-1] + list(seq[:-1])

            batch_size = x.shape[0]

            # 初始化噪声
            shape = x.shape
            z1 = torch.randn([shape[0], 3, shape[2], shape[3]], device=device)
            z2 = torch.randn([shape[0], 3, shape[2], shape[3]], device=device)

            # for alpha in alpha_scale:
            x = self.slerp(z1, z2, alpha)
            for i, j in tqdm(zip(reversed(seq), reversed(seq_next)), desc='sampling loop time step', total=len(seq)):
                t = (torch.ones(batch_size) * i).to(x.device)
                next_t = (torch.ones(batch_size) * j).to(x.device)

                at = self.compute_alpha(self.betas, t.long())
                at_next = self.compute_alpha(self.betas, next_t.long())

                noise_level = torch.FloatTensor([self.sqrt_alphas_cumprod_prev[i + 1]]).repeat(batch_size, 1).to(
                    x.device)
                et,feature = self.denoise_fn(torch.cat([x_in, x], dim=1), noise_level)

                # print(et.shape,at.shape)

                x0_t = (x - et * (1 - at).sqrt()) / at.sqrt()

                # x0_t.clamp_(-1., 1.)

                c1 = (
                        ddim_eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
                )
                c2 = ((1 - at_next) - c1 ** 2).sqrt()
                # print( at_next.sqrt(), c2)
                xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et

                # print(torch.max(xt_next),torch.min(xt_next),  at_next.sqrt(), c2)

                x = xt_next

                if i % sample_inter == 0 or (i == len(seq) - 1):
                    ret_img = torch.cat([ret_img, xt_next], dim=0)
        else:
            sample_inter = (1 | (self.num_timesteps//10))
            if not self.conditional:
                shape = x_in
                #print(shape)
                img = torch.randn(shape, device=device)
                ret_img = img
                for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
                    img = self.p_sample(img, i)
                    if i % sample_inter == 0:
                        ret_img = torch.cat([ret_img, img], dim=0)
            else:
                x = x_in
                shape = x.shape
                img = torch.randn(shape, device=device)
                ret_img = hr_in
                ret_img = torch.cat([ret_img, x], dim=0)
                for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
                    img = self.p_sample(img, i, condition_x=x)
                    #img = self.compute_x_next(img, i, condition_x=x)
                    if i % sample_inter == 0:
                        ret_img = torch.cat([ret_img, img], dim=0)

        if continous:
            return ret_img
        else:
            return ret_img[-1]

    @torch.no_grad()
    def sample(self, batch_size=1, continous=False):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size), continous)

    @torch.no_grad()
    def super_resolution(self, x_in,hr_in=None, continous=False,condition_ddim = False,steps = 2000,eta = 1):
        return self.p_sample_loop(x_in, hr_in, continous,condition_ddim,steps,eta)

    def q_sample(self, x_start, continuous_sqrt_alpha_cumprod, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        # random gama
        return (
            continuous_sqrt_alpha_cumprod * x_start +
            (1 - continuous_sqrt_alpha_cumprod**2).sqrt() * noise
        )

    def p_losses(self, x_in, x_leader=None, feature=None, time=None, noise=None):
        x_start = x_in['HR']
        # print(x_start.shape)
        # print(torch.max(x_start[0]),torch.min(x_start[0]))
        [b, c, h, w] = x_start.shape
        if self.is_leader == True:
            t = np.random.randint(1, self.num_timesteps + 1)
            continuous_sqrt_alpha_cumprod = torch.FloatTensor(
                np.random.uniform(
                    self.sqrt_alphas_cumprod_prev[t - 1],
                    self.sqrt_alphas_cumprod_prev[t],
                    size=b
                )
            ).to(x_start.device)
            continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(
                b, -1)

            noise = default(noise, lambda: torch.randn_like(x_start))
            x_noisy = self.q_sample(
                x_start=x_start, continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1),
                noise=noise)

            if not self.conditional:
                x_recon = self.denoise_fn(x_noisy, continuous_sqrt_alpha_cumprod)
            else:
                x_recon, first_feature = self.denoise_fn(
                    torch.cat([x_in['SR'], x_noisy], dim=1), continuous_sqrt_alpha_cumprod)
            return x_recon, first_feature, t, noise
        else:
            continuous_sqrt_alpha_cumprod = torch.FloatTensor(
                np.random.uniform(
                    self.sqrt_alphas_cumprod_prev[time - 1],
                    self.sqrt_alphas_cumprod_prev[time],
                    size=b
                )
            ).to(x_start.device)
        continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(
            b, -1)

        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(
            x_start=x_start, continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1), noise=noise)

        if not self.conditional:
            x_recon = self.denoise_fn(x_noisy, continuous_sqrt_alpha_cumprod)
        else:
            x_recon, first_feature = self.denoise_fn(
                torch.cat([x_in['SR'], x_noisy], dim=1), continuous_sqrt_alpha_cumprod)
            # print(x_recon.shape)

            # optim loss
            # t = t - 1
            # x_ = self.predict_start_from_noise(x_noisy.detach(), t=t, noise=x_recon.detach())
            # model_mean, posterior = self.q_posterior(x_start=x_, x_t=x_noisy.detach(), t=t)
            # noise_ = torch.randn_like(x_noisy) if t>0 else torch.zeros_like(x_noisy)
            # next_x = model_mean + noise_ * (0.5 * posterior).exp()
            # # optim_loss = self.optim_loss(next_x, x_in['HR'])
            # #
            # # loss = self.loss_func(noise, x_recon) + optim_loss
            #     print(x_leader.shape,x_recon.shape)

            #loss = self.loss_func(noise, x_recon) * 0.65 + self.loss_func(feature,first_feature) * 0.2 + self.loss_func(x_leader, x_recon) * 0.15
            loss = self.loss_func(noise, x_recon) * 0.8 + self.loss_func(feature, first_feature) * 0.2
            return loss

    def forward(self, x, *args, **kwargs):
        return self.p_losses(x, *args, **kwargs)
