import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import math
from functools import partial
from collections import namedtuple
from tqdm.auto import tqdm

import ptwt, pywt
from realpdebench.model.model import Model

# constants

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def sigmoid_beta_schedule(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

# wavelet transform helper functions

def find_rescaler(train_dataset, wavelet, pad_mode, dataset_root, dataset_name, device, batch_size=64):
    save_path = os.path.join(dataset_root, dataset_name, f"wdno_rescaler_{wavelet.name}_{pad_mode}.pt")

    if not os.path.exists(save_path):
        assert train_dataset.dataset_type == 'numerical', "Rescaler should be computed on numerical data"
        loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=12)

        rescaler = None
        for inputs, targets in tqdm(loader, desc="Computing rescaler of WDNO"):
            inputs, targets = inputs.to(device), targets.to(device)
            b, f, h, w, c_in = inputs.shape
            c_out = targets.shape[-1] * targets.shape[1] // f
            c = c_in + c_out
            targets_ = targets.reshape(b, -1, f, h, w, targets.shape[-1]).permute(0, 2, 3, 4, 5, 1) # b, f, h, w, c, sub_f
            targets_ = targets_.reshape(b, f, h, w, c_out)
            data = torch.cat([inputs, targets_], dim=-1)
            data = data.permute(0, 4, 1, 2, 3).reshape(-1, f, h, w) # b*c, f, h, w

            wave_coef = ptwt.wavedec3(data, wavelet, mode=pad_mode, level=1)
            wave_coef = coef_to_tensor(wave_coef)
            wave_coef = wave_coef.reshape(b, c, 8, -1).reshape(b, c*8, -1)
            wave_coef = wave_coef.permute(1, 0, 2).reshape(c*8, -1)

            if rescaler is None:
                rescaler = wave_coef.abs().max(dim=1)[0]
            else:
                rescaler = torch.max(rescaler, wave_coef.abs().max(dim=1)[0])

        rescaler[rescaler == 0] = 1
        rescaler = rescaler.reshape(1, 1, 1, 1, -1)
        torch.save(rescaler.cpu(), save_path)

    else:
        rescaler = torch.load(save_path)

    return (rescaler * 1.4).to(device) # set a larger rescaler

def tensor_to_coef(coef_tensor, shape=None):
    '''
    input: [N, C*8, padded, padded, padded]
    output: list [coarse, detail]
    '''
    if shape is None:
        shape = coef_tensor.shape[-3:]

    Yls = []
    Yhs = []
    for i in range(int(coef_tensor.shape[1]/8)):
        Yl = coef_tensor[:, i*8:i*8+1, :shape[-3], :shape[-2], :shape[-1]]
        Yh = coef_tensor[:, i*8+1:(i+1)*8, :shape[-3], :shape[-2], :shape[-1]]
        Yls.append(Yl)
        Yhs.append(Yh)

    Yl = torch.cat(Yls, dim=1).reshape(-1, shape[-3], shape[-2], shape[-1])
    Yh_ = torch.cat(Yhs, dim=1).reshape(-1, 7, shape[-3], shape[-2], shape[-1])
    Yh = {}
    Yh["aad"] = Yh_[:, 0]
    Yh["ada"] = Yh_[:, 1]
    Yh["add"] = Yh_[:, 2]
    Yh["daa"] = Yh_[:, 3]
    Yh["dad"] = Yh_[:, 4]
    Yh["dda"] = Yh_[:, 5]
    Yh["ddd"] = Yh_[:, 6]
    return Yl, Yh
        
def coef_to_tensor(coef):
    Yl, Yh = coef[0], coef[1]
    Yh = torch.stack(list(Yh.values()), dim=1)
    return torch.cat((Yl[:, None], Yh), dim=1) #, N, 8, T, H, W

class WDNO(Model):
    def __init__(
        self,
        model,
        train_dataset,
        dataset_root,
        dataset_name,
        # wavelet
        wave_type='bior1.3',
        pad_mode='zero',
        # training
        loss_type = 'l2',
        timesteps = 1000,
        beta_schedule = 'sigmoid',
        schedule_fn_kwargs = dict(),
        # sampling
        sampling_timesteps = None,
        ddim_sampling_eta = 0.,
    ):
        super().__init__()

        self.model = model
        self.channels = self.model.channels

        input, target = train_dataset[0]
        self.input_shape = input.shape
        self.output_shape = target.shape
        self.image_size = self.input_shape[1]
        self.frames = self.input_shape[0]

        self.wave_type = wave_type
        self.wavelet = pywt.Wavelet(wave_type)
        self.pad_mode = pad_mode
        sample_data = torch.randn(self.input_shape).permute(3, 0, 1, 2) # C, T, S, S
        sample_wave_data = ptwt.wavedec3(sample_data, self.wavelet, mode=pad_mode, level=1)
        self.coef_shape = coef_to_tensor(sample_wave_data).shape[2:]
        self.rescaler = find_rescaler(train_dataset, self.wavelet, self.pad_mode, dataset_root, dataset_name, 
                                    self.model.time_mlp[1].weight.device)

        # pad because of unet's downsampling
        pad_factor = 2 ** len(self.model.downs)
        self.padded_shape = tuple(
            ((d + pad_factor - 1) // pad_factor) * pad_factor for d in self.coef_shape
        )
        self.pad_x = self.padded_shape[1] - self.coef_shape[1]
        self.pad_t = self.padded_shape[0] - self.coef_shape[0]

        if beta_schedule == 'linear':
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == 'cosine':
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == 'sigmoid':
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)
        
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    
    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def model_predictions(self, x, t, clip_x_start = True, rederive_pred_noise = False):

        model_output = self.model(x, t)
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        pred_noise = model_output
        x_start = self.predict_start_from_noise(x, t, pred_noise)
        x_start = maybe_clip(x_start)

        if clip_x_start and rederive_pred_noise:
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)
    
    def p_mean_variance(self, x, t, clip_denoised = True):
        preds = self.model_predictions(x, t)
        x_start = preds.pred_x_start
        if clip_denoised:
            x_start.clamp_(-1., 1.) 
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        
        return model_mean, posterior_variance, posterior_log_variance, x_start

    def sample_noise(self, shape, device):
        return torch.randn(shape, device = device)

    @torch.no_grad()
    def p_sample(self, x, t: int, clip_denoised = True):
        """
        Different design_guidance follows the paper "Universal Guidance for Diffusion Models"
        """
        b, *_, device = *x.shape, x.device 
        batched_times = torch.full((b,), t, device = x.device, dtype = torch.long)
        p_mean_variance = self.p_mean_variance(x = x, t = batched_times, clip_denoised = clip_denoised)

        model_mean, _, model_log_variance, x_start = p_mean_variance
        noise = self.sample_noise(model_mean.shape, device) if t > 0 else 0
        pred = model_mean + (0.5 * model_log_variance).exp() * noise

        return pred, x_start
            

    @torch.no_grad()
    def p_sample_loop(self, input, device=None):
        batch, device = input.shape[0], self.betas.device
        coef_shape, input_shape, padded_shape, output_shape \
                    = self.coef_shape, self.input_shape, self.padded_shape, self.output_shape
        shape = (batch, *padded_shape, self.channels)

        # self.betas = self.betas.to(device)
        noise_state = self.sample_noise(shape, device) 
        
        # condition
        input = input.to(device)
        ori_input = input.permute(0, 4, 1, 2, 3).reshape(-1, *input_shape[:-1])
        wave_coef_input = ptwt.wavedec3(ori_input, self.wavelet, mode=self.pad_mode, level=1)
        wave_coef_input = coef_to_tensor(wave_coef_input).reshape(batch, input_shape[-1], 8, *coef_shape)
        wave_coef_input = wave_coef_input.reshape(batch, -1, *coef_shape)
        wave_coef_input = F.pad(wave_coef_input, (0, self.pad_x, 0, self.pad_x, 0, self.pad_t),\
                        'constant', 0).permute(0, 2, 3, 4, 1)
        wave_coef_input = wave_coef_input / self.rescaler[..., :wave_coef_input.shape[-1]]

        noise_state = self.set_input_condition(noise_state, wave_coef_input)
        noise_state = self.set_pad_condition(noise_state)

        x = noise_state
        x_start = None
        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            x, x_start = self.p_sample(x, t)
            x = self.set_input_condition(x, wave_coef_input)
            x = self.set_pad_condition(x)
            final_result = x

        final_result = final_result * self.rescaler
        coef = tensor_to_coef(final_result.permute(0, 4, 1, 2, 3), coef_shape)

        ori_data = ptwt.waverec3(coef, self.wavelet)
        ori_data = ori_data.reshape(batch, -1, *ori_data.shape[1:])
        pred = ori_data[:, input_shape[3]:, :input_shape[0], :input_shape[1], :input_shape[2]]
        pred = pred.reshape(batch, output_shape[3], -1, *input_shape[:-1]) # b, c, sub_f, f, h, w
        pred = pred.permute(0, 2, 3, 4, 5, 1) # b, sub_f, f, h, w, c
        ret = pred.reshape(batch, *output_shape)

        return ret

    @torch.no_grad()
    def ddim_sample(self, input, device=None):
        batch = input.shape[0]
        device, total_timesteps, sampling_timesteps, eta = self.betas.device, self.num_timesteps, \
                                                        self.sampling_timesteps, self.ddim_sampling_eta
        padded_shape, coef_shape, input_shape, output_shape \
                    = self.padded_shape, self.coef_shape, self.input_shape, self.output_shape
        shape = (batch, *padded_shape, self.channels)
            
        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   
        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device = device)
        img = torch.randn(shape, device = device)
        
        # condition 
        input = input.to(device)
        ori_input = input.permute(0, 4, 1, 2, 3).reshape(-1, *input_shape[:-1])
        wave_coef_input = ptwt.wavedec3(ori_input, self.wavelet, mode=self.pad_mode, level=1)
        wave_coef_input = coef_to_tensor(wave_coef_input).reshape(batch, input_shape[-1], 8, *coef_shape)
        wave_coef_input = wave_coef_input.reshape(batch, -1, *coef_shape)
        wave_coef_input = F.pad(wave_coef_input, (0, self.pad_x, 0, self.pad_x, 0, self.pad_t),\
                        'constant', 0).permute(0, 2, 3, 4, 1)
        wave_coef_input = wave_coef_input / self.rescaler[..., :wave_coef_input.shape[-1]]

        x_start = None
        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            img = self.set_input_condition(img, wave_coef_input)
            img = self.set_pad_condition(img)

            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, clip_x_start = True, rederive_pred_noise = True)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

        img = self.set_input_condition(img, wave_coef_input)
        img = self.set_pad_condition(img)

        img = img * self.rescaler
        coef = tensor_to_coef(img.permute(0, 4, 1, 2, 3), coef_shape)

        ori_data = ptwt.waverec3(coef, self.wavelet)
        ori_data = ori_data.reshape(batch, -1, *ori_data.shape[1:])
        pred = ori_data[:, input_shape[3]:, :input_shape[0], :input_shape[1], :input_shape[2]]
        pred = pred.reshape(batch, output_shape[3], -1, *input_shape[:-1]) # b, c, sub_f, f, h, w
        pred = pred.permute(0, 2, 3, 4, 5, 1) # b, sub_f, f, h, w, c
        ret = pred.reshape(batch, *output_shape)

        return ret

    @torch.no_grad()
    def forward(self, input = None, device = None):
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample

        return sample_fn(input = input, device = device)

    @torch.no_grad()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.full((b,), t, device = device)
        xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2

        x_start = None

        for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
            img, x_start = self.p_sample(img, i)

        return img

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )        

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def set_pad_condition(self, state):
        coef_shape = self.coef_shape
        state[:, coef_shape[0]:, :, :, :] = 0
        state[:, :, coef_shape[1]:] = 0
        state[:, :, :, coef_shape[2]:] = 0
        return state

    def set_input_condition(self, state, input):
        c_in = input.shape[-1]
        state[..., :c_in] = input[..., :c_in]
        return state

    def p_losses(self, input, target, noise = None):
        b, f, h, w, c_in = input.shape
        t = torch.randint(0, self.num_timesteps, (b,), device=input.device).long()
        coef_shape = self.coef_shape

        c_out = target.shape[-1] * target.shape[1] // f # c*sub_f
        target_ = target.reshape(b, -1, f, h, w, target.shape[-1]).permute(0, 2, 3, 4, 5, 1) # b, f, h, w, c, sub_f
        target_ = target_.reshape(b, f, h, w, c_out)
        ori_data = torch.cat([input, target_], dim=-1)
        ori_data = ori_data.permute(0, 4, 1, 2, 3).reshape(-1, f, h, w) # b*(c_in+c_out), f, h, w

        wave_coef = ptwt.wavedec3(ori_data, self.wavelet, mode=self.pad_mode, level=1)
        wave_coef_tensor = coef_to_tensor(wave_coef).reshape(b, c_in+c_out, 8, *coef_shape)
        wave_coef_tensor = wave_coef_tensor.reshape(b, -1, *coef_shape)
        wave_coef_tensor = F.pad(wave_coef_tensor, (0, self.pad_x, 0, self.pad_x, 0, self.pad_t),\
                        'constant', 0).permute(0, 2, 3, 4, 1)
        state_start = wave_coef_tensor / self.rescaler
        wave_coef_input = state_start[..., :8*c_in]

        noise_state = default(noise, lambda: torch.randn_like(state_start))
        
        # noisy sample
        state = self.q_sample(x_start = state_start, t = t, noise = noise_state)
        
        # condition on input
        # print("conditional ... ")
        state = self.set_input_condition(state, wave_coef_input)
        noise_state = self.set_input_condition(noise_state, torch.zeros_like(wave_coef_input))
        # condition on padding
        state = self.set_pad_condition(state)
        noise_state = self.set_pad_condition(noise_state)

        model_out = self.model(state, t)
        
        loss = self.loss_fn(model_out, noise_state, reduction = 'none')

        return loss

    def train_loss(self, input, target):
        return self.p_losses(input, target)
