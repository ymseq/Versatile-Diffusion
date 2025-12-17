"""SAMPLING ONLY. (Jittor version)
"""

import numpy as np
from tqdm import tqdm
import jittor as jt

from .diffusion_utils import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like


# -------------------------
# helpers
# -------------------------

def _normalize_jt_dtype(dtype):
    if dtype is None:
        return None
    if dtype in (jt.float16, jt.float32, jt.float64, jt.int32, jt.int64):
        return dtype

    # numpy dtype
    try:
        import numpy as _np
        if dtype in (_np.float16, _np.dtype("float16")):
            return jt.float16
        if dtype in (_np.float32, _np.dtype("float32")):
            return jt.float32
        if dtype in (_np.float64, _np.dtype("float64")):
            return jt.float64
        if dtype in (_np.int32, _np.dtype("int32")):
            return jt.int32
        if dtype in (_np.int64, _np.dtype("int64")):
            return jt.int64
    except Exception:
        pass

    # torch dtype / string
    s = str(dtype).lower()
    if "float16" in s or "half" in s:
        return jt.float16
    if "float32" in s or ("float" in s and "float16" not in s and "float64" not in s):
        return jt.float32
    if "float64" in s or "double" in s:
        return jt.float64
    if "int32" in s:
        return jt.int32
    if "int64" in s or "long" in s:
        return jt.int64

    return jt.float32


def _maybe_cuda(v, cuda=True):
    if cuda and jt.flags.use_cuda:
        return v.cuda()
    return v


def _to_numpy_any(x):
    """Convert anything -> numpy (jt.Var / numpy / python / torch.Tensor duck-typed)."""
    if isinstance(x, jt.Var):
        return x.numpy()
    if isinstance(x, np.ndarray):
        return x
    # duck-typing for torch.Tensor
    if hasattr(x, "detach") and hasattr(x, "cpu") and hasattr(x, "numpy"):
        try:
            return x.detach().cpu().numpy()
        except Exception:
            pass
    return np.array(x)


def _to_jt(x, dtype=jt.float32, cuda=True, stop_grad=True):
    """Convert numpy / python / jt.Var / torch.Tensor(duck-typed) -> jt.Var."""
    if isinstance(x, jt.Var):
        v = x
    else:
        v = jt.array(_to_numpy_any(x))
    if stop_grad:
        v = v.stop_grad()
    if dtype is not None:
        dtype = _normalize_jt_dtype(dtype)
        v = v.astype(dtype) if hasattr(v, "astype") else v.cast(dtype)
    v = _maybe_cuda(v, cuda=cuda)
    return v


def _ensure_c_info_jt(c_info, dtype):
    """In-place ensure conditioning tensors are jt.Var."""
    dtype = _normalize_jt_dtype(dtype)
    if "conditioning" in c_info and c_info["conditioning"] is not None:
        c_info["conditioning"] = _to_jt(c_info["conditioning"], dtype=dtype)
    if "unconditional_conditioning" in c_info and c_info["unconditional_conditioning"] is not None:
        c_info["unconditional_conditioning"] = _to_jt(c_info["unconditional_conditioning"], dtype=dtype)
    return c_info


def _get_model_dtype(model, fallback=jt.float32):
    dt = getattr(model, "dtype", None)
    return _normalize_jt_dtype(dt) if dt is not None else fallback


def _scalar_at(vec, index, dtype):
    """
    vec: jt.Var (1D) or numpy array/list
    return: jt.Var scalar (0D/1D[1]) casted to dtype
    """
    if isinstance(vec, jt.Var):
        s = vec[index]
        return s.astype(dtype) if hasattr(s, "astype") else s.cast(dtype)
    else:
        return _to_jt(vec[index], dtype=dtype, stop_grad=True)


def _broadcast_scalar(scalar, shape, dtype, cuda=True):
    """
    scalar: jt.Var (scalar)
    shape: list like [b,1,1,1]
    """
    one = jt.ones(shape, dtype=dtype)
    one = _maybe_cuda(one, cuda=cuda)
    # scalar already jt.Var; broadcast multiply
    return one * scalar


# -------------------------
# DDIM Sampler
# -------------------------

class DDIMSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = int(model.num_timesteps)
        self.schedule = schedule

    def register_buffer(self, name, attr, dtype=jt.float32):
        """
        Long-term plan: always store buffers as jt.Var (1D where applicable), stop_grad.
        """
        v = _to_jt(attr, dtype=dtype, stop_grad=True, cuda=True)
        if hasattr(self, name):
            raise ValueError(f"Trying to register {name} but {name} already exists!")
        setattr(self, name, v)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        # keep timesteps as numpy for Python loop convenience
        self.ddim_timesteps = make_ddim_timesteps(
            ddim_discr_method=ddim_discretize,
            num_ddim_timesteps=ddim_num_steps,
            num_ddpm_timesteps=self.ddpm_num_timesteps,
            verbose=verbose
        )

        alphas_cumprod = self.model.alphas_cumprod
        assert int(alphas_cumprod.shape[0]) == self.ddpm_num_timesteps, \
            "alphas have to be defined for each timestep"

        # model buffers (jt.Var)
        self.register_buffer("betas", self.model.betas, dtype=jt.float32)
        self.register_buffer("alphas_cumprod", alphas_cumprod, dtype=jt.float32)
        self.register_buffer("alphas_cumprod_prev", self.model.alphas_cumprod_prev, dtype=jt.float32)

        # numpy copy for parameter derivation (one-time cost)
        ac_np = np.array(alphas_cumprod.numpy(), dtype=np.float64)

        # store sqrt/log buffers as jt.Var(1D)
        self.register_buffer("sqrt_alphas_cumprod", np.sqrt(ac_np).astype(np.float32), dtype=jt.float32)
        self.register_buffer("sqrt_one_minus_alphas_cumprod", np.sqrt(1. - ac_np).astype(np.float32), dtype=jt.float32)
        self.register_buffer("log_one_minus_alphas_cumprod", np.log(1. - ac_np).astype(np.float32), dtype=jt.float32)
        self.register_buffer("sqrt_recip_alphas_cumprod", np.sqrt(1. / ac_np).astype(np.float32), dtype=jt.float32)
        self.register_buffer("sqrt_recipm1_alphas_cumprod", np.sqrt(1. / ac_np - 1.).astype(np.float32), dtype=jt.float32)

        # DDIM sampling parameters from numpy -> jt.Var
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(
            alphacums=ac_np,
            ddim_timesteps=self.ddim_timesteps,
            eta=ddim_eta,
            verbose=verbose
        )

        ddim_sigmas = np.array(ddim_sigmas, dtype=np.float32)
        ddim_alphas = np.array(ddim_alphas, dtype=np.float32)
        ddim_alphas_prev = np.array(ddim_alphas_prev, dtype=np.float32)
        ddim_sqrt_oma = np.sqrt(1. - ddim_alphas).astype(np.float32)

        self.register_buffer("ddim_sigmas", ddim_sigmas, dtype=jt.float32)
        self.register_buffer("ddim_alphas", ddim_alphas, dtype=jt.float32)
        self.register_buffer("ddim_alphas_prev", ddim_alphas_prev, dtype=jt.float32)
        self.register_buffer("ddim_sqrt_one_minus_alphas", ddim_sqrt_oma, dtype=jt.float32)

        # original DDPM steps sigmas (jt.Var 1D)
        # note: this is length = ddpm_num_timesteps
        sigmas_ddpm = float(ddim_eta) * jt.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) *
            (1 - self.alphas_cumprod / self.alphas_cumprod_prev)
        )
        self.register_buffer("ddim_sigmas_for_original_num_steps", sigmas_ddpm, dtype=jt.float32)

    def sample(self,
               steps,
               shape,
               x_info,
               c_info,
               eta=0.,
               temperature=1.,
               noise_dropout=0.,
               verbose=True,
               log_every_t=100):
        with jt.no_grad():
            self.make_schedule(ddim_num_steps=steps, ddim_eta=eta, verbose=verbose)
            print(f"Data shape for DDIM sampling is {shape}, eta {eta}")
            return self.ddim_sampling(
                shape,
                x_info=x_info,
                c_info=c_info,
                noise_dropout=noise_dropout,
                temperature=temperature,
                log_every_t=log_every_t
            )

    def ddim_sampling(self,
                      shape,
                      x_info,
                      c_info,
                      noise_dropout=0.,
                      temperature=1.,
                      log_every_t=100):
        with jt.no_grad():
            dtype = _get_model_dtype(self.model, fallback=jt.float32)
            _ensure_c_info_jt(c_info, dtype=dtype)

            bs = shape[0]
            timesteps = self.ddim_timesteps  # numpy array, for loop

            # init x
            if ("xt" in x_info) and (x_info["xt"] is not None):
                x_info["x"] = _to_jt(x_info["xt"], dtype=dtype)
            elif ("x0" in x_info) and (x_info["x0"] is not None):
                x0 = _to_jt(x_info["x0"], dtype=dtype)
                fwd = int(x_info["x0_forward_timesteps"])
                ts_np = np.repeat(timesteps[fwd], bs).astype(np.int32)
                ts = _to_jt(ts_np, dtype=jt.int32)
                timesteps = timesteps[:fwd]
                x_info["x"] = self.model.q_sample(x0, ts)
            else:
                x_info["x"] = _maybe_cuda(jt.randn(shape, dtype=dtype), cuda=True)

            intermediates = {"pred_xt": [], "pred_x0": []}
            time_range = np.flip(timesteps)
            total_steps = timesteps.shape[0]

            iterator = tqdm(time_range, desc="DDIM Sampler", total=total_steps)

            pred_xt = None
            pred_x0 = None
            for i, step in enumerate(iterator):
                index = total_steps - i - 1
                ts = _maybe_cuda(jt.full([bs], int(step), dtype=jt.int32), cuda=True)
                pred_xt, pred_x0 = self.p_sample_ddim(
                    x_info, c_info, ts, index,
                    noise_dropout=noise_dropout,
                    temperature=temperature
                )
                x_info["x"] = pred_xt

                if index % log_every_t == 0 or index == total_steps - 1:
                    intermediates["pred_xt"].append(pred_xt)
                    intermediates["pred_x0"].append(pred_x0)

            return pred_xt, intermediates

    def p_sample_ddim(self, x_info, c_info, t, index,
                      repeat_noise=False,
                      use_original_steps=False,
                      noise_dropout=0.,
                      temperature=1.):
        with jt.no_grad():
            dtype = _get_model_dtype(self.model, fallback=x_info["x"].dtype)
            _ensure_c_info_jt(c_info, dtype=dtype)

            x = x_info["x"]
            b = x.shape[0]
            ugs = float(c_info["unconditional_guidance_scale"])

            if ugs == 1.0:
                c_info["c"] = c_info["conditioning"]
                e_t = self.model.apply_model(x_info, t, c_info)
            else:
                x_in = jt.concat([x, x], dim=0)
                t_in = jt.concat([t, t], dim=0)
                c_in = jt.concat([c_info["unconditional_conditioning"], c_info["conditioning"]], dim=0)

                x_info["x"] = x_in
                c_info["c"] = c_in
                out = self.model.apply_model(x_info, t_in, c_info)
                e_t_uncond, e_t = jt.chunk(out, 2, dim=0)
                e_t = e_t_uncond + ugs * (e_t - e_t_uncond)

                x_info["x"] = x  # restore

            # pick 1D schedule buffers (jt.Var)
            alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
            alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
            sqrt_oma = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
            sigmas = self.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas

            # scalar jt.Var (no item / float)
            a  = _scalar_at(alphas, index, dtype=x.dtype)
            ap = _scalar_at(alphas_prev, index, dtype=x.dtype)
            soa = _scalar_at(sqrt_oma, index, dtype=x.dtype)
            sg = _scalar_at(sigmas, index, dtype=x.dtype)

            ext = [b] + [1] * (len(e_t.shape) - 1)
            a_t = _broadcast_scalar(a, ext, dtype=x.dtype, cuda=True)
            a_prev = _broadcast_scalar(ap, ext, dtype=x.dtype, cuda=True)
            sqrt_one_minus_at = _broadcast_scalar(soa, ext, dtype=x.dtype, cuda=True)
            sigma_t = _broadcast_scalar(sg, ext, dtype=x.dtype, cuda=True)

            pred_x0 = (x - sqrt_one_minus_at * e_t) / jt.sqrt(a_t)
            dir_xt = jt.sqrt(1.0 - a_prev - sigma_t ** 2) * e_t

            noise = sigma_t * noise_like(x, repeat_noise) * float(temperature)

            if noise_dropout > 0.0:
                p = float(noise_dropout)
                mask = (jt.rand(noise.shape) >= p).astype(noise.dtype)
                noise = noise * mask / (1.0 - p)

            x_prev = jt.sqrt(a_prev) * pred_x0 + dir_xt + noise
            return x_prev, pred_x0

    # ---------------- multicontext ----------------

    def sample_multicontext(self,
                            steps,
                            shape,
                            x_info,
                            c_info_list,
                            eta=0.,
                            temperature=1.,
                            noise_dropout=0.,
                            verbose=True,
                            log_every_t=100):
        with jt.no_grad():
            self.make_schedule(ddim_num_steps=steps, ddim_eta=eta, verbose=verbose)
            print(f"Data shape for DDIM sampling is {shape}, eta {eta}")
            return self.ddim_sampling_multicontext(
                shape,
                x_info=x_info,
                c_info_list=c_info_list,
                noise_dropout=noise_dropout,
                temperature=temperature,
                log_every_t=log_every_t
            )

    def ddim_sampling_multicontext(self,
                                   shape,
                                   x_info,
                                   c_info_list,
                                   noise_dropout=0.,
                                   temperature=1.,
                                   log_every_t=100):
        with jt.no_grad():
            dtype = _get_model_dtype(self.model, fallback=jt.float32)
            for ci in c_info_list:
                _ensure_c_info_jt(ci, dtype=dtype)

            bs = shape[0]
            timesteps = self.ddim_timesteps

            if ("xt" in x_info) and (x_info["xt"] is not None):
                x_info["x"] = _to_jt(x_info["xt"], dtype=dtype)
            elif ("x0" in x_info) and (x_info["x0"] is not None):
                x0 = _to_jt(x_info["x0"], dtype=dtype)
                fwd = int(x_info["x0_forward_timesteps"])
                ts_np = np.repeat(timesteps[fwd], bs).astype(np.int32)
                ts = _to_jt(ts_np, dtype=jt.int32)
                timesteps = timesteps[:fwd]
                x_info["x"] = self.model.q_sample(x0, ts)
            else:
                x_info["x"] = _maybe_cuda(jt.randn(shape, dtype=dtype), cuda=True)

            intermediates = {"pred_xt": [], "pred_x0": []}
            time_range = np.flip(timesteps)
            total_steps = timesteps.shape[0]
            iterator = tqdm(time_range, desc="DDIM Sampler", total=total_steps)

            pred_xt = None
            pred_x0 = None
            for i, step in enumerate(iterator):
                index = total_steps - i - 1
                ts = _maybe_cuda(jt.full([bs], int(step), dtype=jt.int32), cuda=True)

                pred_xt, pred_x0 = self.p_sample_ddim_multicontext(
                    x_info, c_info_list, ts, index,
                    noise_dropout=noise_dropout,
                    temperature=temperature
                )
                x_info["x"] = pred_xt

                if index % log_every_t == 0 or index == total_steps - 1:
                    intermediates["pred_xt"].append(pred_xt)
                    intermediates["pred_x0"].append(pred_x0)

            return pred_xt, intermediates

    def p_sample_ddim_multicontext(self, x_info, c_info_list, t, index,
                                   repeat_noise=False,
                                   use_original_steps=False,
                                   noise_dropout=0.,
                                   temperature=1.):
        with jt.no_grad():
            x = x_info["x"]
            b = x.shape[0]
            dtype = _get_model_dtype(self.model, fallback=x.dtype)

            ugs_ref = None
            for ci in c_info_list:
                _ensure_c_info_jt(ci, dtype=dtype)
                ugs = float(ci["unconditional_guidance_scale"])
                if ugs_ref is None:
                    ugs_ref = ugs
                else:
                    assert ugs_ref == ugs, "Different unconditional guidance scale between contexts is not allowed!"

                if ugs_ref == 1.0:
                    ci["c"] = ci["conditioning"]
                else:
                    ci["c"] = jt.concat([ci["unconditional_conditioning"], ci["conditioning"]], dim=0)

            if ugs_ref == 1.0:
                e_t = self.model.apply_model_multicontext(x_info, t, c_info_list)
            else:
                x_in = jt.concat([x, x], dim=0)
                t_in = jt.concat([t, t], dim=0)
                x_info["x"] = x_in

                out = self.model.apply_model_multicontext(x_info, t_in, c_info_list)
                e_t_uncond, e_t = jt.chunk(out, 2, dim=0)
                e_t = e_t_uncond + ugs_ref * (e_t - e_t_uncond)

                x_info["x"] = x  # restore

            alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
            alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
            sqrt_oma = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
            sigmas = self.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas

            a  = _scalar_at(alphas, index, dtype=x.dtype)
            ap = _scalar_at(alphas_prev, index, dtype=x.dtype)
            soa = _scalar_at(sqrt_oma, index, dtype=x.dtype)
            sg = _scalar_at(sigmas, index, dtype=x.dtype)

            ext = [b] + [1] * (len(e_t.shape) - 1)
            a_t = _broadcast_scalar(a, ext, dtype=x.dtype, cuda=True)
            a_prev = _broadcast_scalar(ap, ext, dtype=x.dtype, cuda=True)
            sqrt_one_minus_at = _broadcast_scalar(soa, ext, dtype=x.dtype, cuda=True)
            sigma_t = _broadcast_scalar(sg, ext, dtype=x.dtype, cuda=True)

            pred_x0 = (x - sqrt_one_minus_at * e_t) / jt.sqrt(a_t)
            dir_xt = jt.sqrt(1.0 - a_prev - sigma_t ** 2) * e_t

            noise = sigma_t * noise_like(x, repeat_noise) * float(temperature)
            if noise_dropout > 0.0:
                p = float(noise_dropout)
                mask = (jt.rand(noise.shape) >= p).astype(noise.dtype)
                noise = noise * mask / (1.0 - p)

            x_prev = jt.sqrt(a_prev) * pred_x0 + dir_xt + noise
            return x_prev, pred_x0
