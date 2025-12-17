import os
import math
import numpy as np
import jittor as jt
from jittor import nn
from jittor.einops import repeat

# -------------------------
# schedules
# -------------------------

def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == "linear":
        betas = (np.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=np.float64) ** 2)

    elif schedule == "cosine":
        timesteps = (np.arange(n_timestep + 1, dtype=np.float64) / n_timestep + cosine_s)
        alphas = timesteps / (1 + cosine_s) * np.pi / 2
        alphas = np.cos(alphas) ** 2
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = np.clip(betas, a_min=0, a_max=0.999)

    elif schedule == "sqrt_linear":
        betas = np.linspace(linear_start, linear_end, n_timestep, dtype=np.float64)

    elif schedule == "sqrt":
        betas = np.sqrt(np.linspace(linear_start, linear_end, n_timestep, dtype=np.float64))

    else:
        raise ValueError(f"schedule '{schedule}' unknown.")

    return betas


def make_ddim_timesteps(ddim_discr_method, num_ddim_timesteps, num_ddpm_timesteps, verbose=True):
    if ddim_discr_method == 'uniform':
        c = num_ddpm_timesteps // num_ddim_timesteps
        ddim_timesteps = np.asarray(list(range(0, num_ddpm_timesteps, c)))
    elif ddim_discr_method == 'quad':
        ddim_timesteps = ((np.linspace(0, np.sqrt(num_ddpm_timesteps * .8), num_ddim_timesteps)) ** 2).astype(int)
    else:
        raise NotImplementedError(f'There is no ddim discretization method called "{ddim_discr_method}"')

    # assert ddim_timesteps.shape[0] == num_ddim_timesteps
    # add one to get the final alpha values right (the ones from first scale to data during sampling)
    steps_out = ddim_timesteps + 1
    if verbose:
        print(f'Selected timesteps for ddim sampler: {steps_out}')
    return steps_out


def make_ddim_sampling_parameters(alphacums, ddim_timesteps, eta, verbose=True):
    # select alphas for computing the variance schedule
    alphas = alphacums[ddim_timesteps]
    alphas_prev = np.asarray([alphacums[0]] + alphacums[ddim_timesteps[:-1]].tolist())
    
    # according the the formula provided in https://arxiv.org/abs/2010.02502
    sigmas = eta * np.sqrt((1 - alphas_prev) / (1 - alphas) * (1 - alphas / alphas_prev))
    if verbose:
        print(f'Selected alphas for ddim sampler: a_t: {alphas}; a_(t-1): {alphas_prev}')
        print(f'For eta={eta}, sigma_t schedule: {sigmas}')
    return sigmas, alphas, alphas_prev


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas, dtype=np.float64)


# -------------------------
# tensor helpers (jittor)
# -------------------------

def extract_into_tensor(a, t, x_shape):
    """
    a: 1-D jt.Var of shape [T]
    t: jt.Var int indices, shape [b]
    return: jt.Var shape [b, 1, 1, ...] broadcast-friendly
    """
    if t.dtype != jt.int32:
        t = t.int32()

    if len(a.shape) == 1:
        out = a[t]
    else:
        out = jt.gather(a, -1, t)

    b = t.shape[0]
    reshape_to = [b] + [1] * (len(x_shape) - 1)
    return out.reshape(reshape_to)


# -------------------------
# checkpoint (fallback)
# -------------------------

def checkpoint(func, inputs, params, flag):
    """
    todo (jittor)
    """
    return func(*inputs)


# -------------------------
# timestep embedding (jittor)
# -------------------------

def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    if not repeat_only:
        half = dim // 2
        # freqs: [half]
        freqs = jt.exp(
            -math.log(max_period) * jt.arange(0, half, dtype=jt.float32) / float(half)
        )
        # args: [b, half]
        args = timesteps.reshape([-1, 1]).float() * freqs.reshape([1, -1])
        embedding = jt.concat([jt.cos(args), jt.sin(args)], dim=-1)  # [b, 2*half]
        if dim % 2 == 1: 
            embedding = jt.concat([embedding, jt.zeros([embedding.shape[0], 1], dtype=embedding.dtype)], dim=-1)
    else:
        embedding = repeat(timesteps, 'b -> b d', d=dim)
    return embedding


# -------------------------
# module init helpers (jittor)
# -------------------------

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.assign(jt.zeros_like(p))
    return module


def scale_module(module, scale):
    """
    Scale the parameters of a module and return it.
    """
    for p in module.parameters():
        p.assign(p * scale)
    return module


def mean_flat(tensor):
    """
    mean over all non-batch dims
    """
    if len(tensor.shape) <= 1:
        return tensor
    dims = list(range(1, len(tensor.shape)))
    return jt.mean(tensor, dims=dims)


def normalization(channels):
    return GroupNorm32(32, channels)


class SiLU(nn.Module):
    def execute(self, x):
        return x * jt.sigmoid(x)


class GroupNorm32(nn.GroupNorm):
    def execute(self, x):
        return super().execute(x)


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args, **kwargs):
    return nn.Linear(*args, **kwargs)

# -------------------------
# implement AvgPool1d (jittor)
# -------------------------

def _to_1tuple(v):
    if isinstance(v, (list, tuple)):
        assert len(v) == 1
        return int(v[0])
    return int(v)

class AvgPool1d(nn.Module):
    """
    x: [B, C, L] -> reshape [B, C, L, 1] -> AvgPool2d -> [B, C, L', 1] -> squeeze -> [B, C, L']
    """
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True):
        super().__init__()
        k = _to_1tuple(kernel_size)
        s = _to_1tuple(stride) if stride is not None else k
        p = _to_1tuple(padding)

        self.pool2d = nn.AvgPool2d(
            kernel_size=(k, 1),
            stride=(s, 1),
            padding=(p, 0),
        )
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad

    def execute(self, x):
        x = x.unsqueeze(-1)     # [B, C, L, 1]
        x = self.pool2d(x)
        x = x.squeeze(-1)       # [B, C, L']
        return x

def avg_pool_nd(dims, *args, **kwargs):
    if dims == 1:
        return AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


# -------------------------
# conditioner
# -------------------------

class HybridConditioner(nn.Module):
    """
    Unknown and not used.
    """
    def __init__(self, c_concat_config, c_crossattn_config):
        super().__init__()
        self.concat_conditioner = instantiate_from_config(c_concat_config)
        self.crossattn_conditioner = instantiate_from_config(c_crossattn_config)

    def execute(self, c_concat, c_crossattn):
        c_concat = self.concat_conditioner(c_concat)
        c_crossattn = self.crossattn_conditioner(c_crossattn)
        return {'c_concat': [c_concat], 'c_crossattn': [c_crossattn]}


def noise_like(x, repeat=False):
    noise = jt.randn_like(x)
    if repeat:
        bs = x.shape[0]
        n0 = noise[0:1]
        reps = [bs] + [1] * (len(x.shape) - 1)
        noise = jt.broadcast(n0, reps)
    return noise


# -------------------------
# params count
# -------------------------

def count_params(model, verbose=False):
    total_params = 0
    for p in model.parameters():
        total_params += int(np.prod(p.shape))
    if verbose:
        print(f"{model.__class__.__name__} has {total_params*1.e-6:.2f} M params.")
    return total_params
