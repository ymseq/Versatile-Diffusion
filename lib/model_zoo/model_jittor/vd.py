import math
import numpy as np
import numpy.random as npr
from contextlib import contextmanager

import jittor as jt
import jittor.nn as nn

from lib.model_zoo.common.get_model import get_model, register
from lib.log_service import print_log

from ..model_torch.torch_hub import TorchVAEContextHub

from .diffusion_utils import make_beta_schedule, extract_into_tensor, timestep_embedding
from .ema import LitEma
from .nn_compat import ModuleDict

symbol = 'vd'


def highlight_print(info):
    print_log('')
    print_log(''.join(['#'] * (len(info) + 4)))
    print_log('# ' + info + ' #')
    print_log(''.join(['#'] * (len(info) + 4)))
    print_log('')


def _strip_module_prefix(sd: dict):
    def strip(k):
        return k[7:] if isinstance(k, str) and k.startswith("module.") else k
    return {strip(k): v for k, v in sd.items()}


def _to_numpy_any(x):
    """兼容 torch.Tensor / jt.Var / np.ndarray / 标量，不在本文件 import torch，走 duck-typing。"""
    if isinstance(x, np.ndarray):
        return x
    # jittor Var
    if hasattr(x, "numpy") and callable(getattr(x, "numpy")):
        try:
            return x.numpy()
        except Exception:
            pass
    # torch Tensor (duck typing)
    if hasattr(x, "detach") and callable(getattr(x, "detach")):
        x = x.detach()
    if hasattr(x, "cpu") and callable(getattr(x, "cpu")):
        try:
            x = x.cpu()
        except Exception:
            pass
    if hasattr(x, "numpy") and callable(getattr(x, "numpy")):
        try:
            return x.numpy()
        except Exception:
            pass
    return np.array(x)


def _to_jt_var(x, dtype=None):
    if isinstance(x, jt.Var):
        return x if dtype is None else x.astype(dtype)
    arr = _to_numpy_any(x)
    v = jt.array(arr)
    return v if dtype is None else v.astype(dtype)


class String_Reg_Buffer(nn.Module):
    def __init__(self, output_string):
        super().__init__()
        self.output_string = str(output_string)

    def execute(self, *args, **kwargs):
        return self.output_string


@register('vd_v2_0')
class VD_v2_0(nn.Module):

    _BUFFER_NAMES = [
        "betas",
        "alphas_cumprod",
        "alphas_cumprod_prev",
        "sqrt_alphas_cumprod",
        "sqrt_one_minus_alphas_cumprod",
        "log_one_minus_alphas_cumprod",
        "sqrt_recip_alphas_cumprod",
        "sqrt_recipm1_alphas_cumprod",
        "posterior_variance",
        "posterior_log_variance_clipped",
        "posterior_mean_coef1",
        "posterior_mean_coef2",
        "lvlb_weights",
        "logvar",
    ]

    def __init__(
        self,
        vae_cfg_list,
        ctx_cfg_list,
        diffuser_cfg_list,
        global_layer_ptr=None,

        parameterization="eps",
        timesteps=1000,
        use_ema=False,

        beta_schedule="linear",
        beta_linear_start=1e-4,
        beta_linear_end=2e-2,
        given_betas=None,
        cosine_s=8e-3,

        loss_type="l2",
        l_simple_weight=1.,
        l_elbo_weight=0.,

        v_posterior=0.,
        learn_logvar=False,
        logvar_init=0,

        latent_scale_factor=None,
    ):
        super().__init__()
        assert parameterization in ["eps", "x0"], 'currently only supporting "eps" and "x0"'
        self.parameterization = parameterization
        highlight_print(f"Running in {self.parameterization} mode")

        # ---- torch hub: vae + context (inference only) ----
        self.torch_hub = TorchVAEContextHub(
            vae_cfg_list=vae_cfg_list,
            ctx_cfg_list=ctx_cfg_list,
        )

        self.vae = self.torch_hub.vae
        self.ctx = self.torch_hub.ctx

        # ---- jittor diffusion modules ----
        self.diffuser = self.get_model_list(diffuser_cfg_list)
        self.global_layer_ptr = global_layer_ptr

        assert self.check_diffuser(), "diffuser layers are not aligned!"

        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self)
            print_log(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        self.loss_type = loss_type
        self.l_simple_weight = float(l_simple_weight)
        self.l_elbo_weight = float(l_elbo_weight)
        self.v_posterior = float(v_posterior)
        self.dtype = jt.float32
        self.device = "cpu"

        self.register_schedule(
            given_betas=given_betas,
            beta_schedule=beta_schedule,
            timesteps=timesteps,
            linear_start=beta_linear_start,
            linear_end=beta_linear_end,
            cosine_s=cosine_s,
        )

        self.learn_logvar = bool(learn_logvar)
        self.logvar = jt.ones([self.num_timesteps], dtype=jt.float32) * float(logvar_init)
        if not self.learn_logvar:
            self.logvar.stop_grad()

        self.latent_scale_factor = {} if latent_scale_factor is None else latent_scale_factor

        # 兼容你原先的 parameter_group 结构（供外部 optimizer 构建）
        self.parameter_group = {}
        for namei, diffuseri in self.diffuser.items():
            if hasattr(diffuseri, "parameter_group"):
                for pgni, pgi in diffuseri.parameter_group.items():
                    self.parameter_group[f"diffuser_{namei}_{pgni}"] = pgi


    def execute(self, x_info, c_info):
        return self.forward(x_info, c_info)

    def forward(self, x_info, c_info):
        x = _to_jt_var(x_info["x"], dtype=self.dtype)
        bs = x.shape[0]
        t = jt.randint(0, self.num_timesteps, [bs]).int32()
        return self.p_losses({"type": x_info["type"], "x": x}, t, c_info)

    # -----------------------
    # device / dtype helpers
    # -----------------------
    def to(self, device):
        self.device = str(device)
        if "cuda" in self.device:
            jt.flags.use_cuda = 1
        else:
            jt.flags.use_cuda = 0

        if hasattr(self.torch_hub, "to"):
            self.torch_hub.to(device)
        return self

    def half(self):
        self.dtype = jt.float16

        for n in self._BUFFER_NAMES:
            if hasattr(self, n):
                v = getattr(self, n)
                if isinstance(v, jt.Var):
                    vv = v.astype(jt.float16)
                    if (n != "logvar") or (not self.learn_logvar):
                        vv.stop_grad()
                    setattr(self, n, vv)

        for _, dmod in self.diffuser.items():
            if hasattr(dmod, "half") and callable(getattr(dmod, "half")):
                dmod.half()

        if hasattr(self.torch_hub, "half"):
            self.torch_hub.half()

        return self

    def float(self):
        self.dtype = jt.float32

        for n in self._BUFFER_NAMES:
            if hasattr(self, n):
                v = getattr(self, n)
                if isinstance(v, jt.Var):
                    vv = v.astype(jt.float32)
                    if (n != "logvar") or (not self.learn_logvar):
                        vv.stop_grad()
                    setattr(self, n, vv)
        for _, dmod in self.diffuser.items():
            if hasattr(dmod, "float") and callable(getattr(dmod, "float")):
                dmod.float()

        if hasattr(self.torch_hub, "float"):
            self.torch_hub.float()

        return self

    # -----------------------
    # model list
    # -----------------------

    def get_model_list(self, cfg_list):
        net = ModuleDict()
        for name, cfg in cfg_list:
            if not isinstance(cfg, str):
                net[name] = get_model()(cfg)
            else:
                net[name] = String_Reg_Buffer(cfg)
        return net

    # -----------------------
    # diffusion schedule
    # -----------------------
    def register_schedule(
        self,
        given_betas=None,
        beta_schedule="linear",
        timesteps=1000,
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
    ):
        if given_betas is not None:
            betas = np.array(given_betas, dtype=np.float32)
        else:
            betas = make_beta_schedule(
                beta_schedule,
                timesteps,
                linear_start=linear_start,
                linear_end=linear_end,
                cosine_s=cosine_s,
            ).astype(np.float32)

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0).astype(np.float32)
        alphas_cumprod_prev = np.append(np.array([1.0], dtype=np.float32), alphas_cumprod[:-1]).astype(np.float32)

        self.num_timesteps = int(betas.shape[0])
        self.linear_start = float(linear_start)
        self.linear_end = float(linear_end)

        def to_jt_buf(x):
            v = jt.array(x).astype(jt.float32)
            v.stop_grad()
            return v

        self.betas = to_jt_buf(betas)
        self.alphas_cumprod = to_jt_buf(alphas_cumprod)
        self.alphas_cumprod_prev = to_jt_buf(alphas_cumprod_prev)

        self.sqrt_alphas_cumprod = to_jt_buf(np.sqrt(alphas_cumprod))
        self.sqrt_one_minus_alphas_cumprod = to_jt_buf(np.sqrt(1.0 - alphas_cumprod))
        self.log_one_minus_alphas_cumprod = to_jt_buf(np.log(1.0 - alphas_cumprod + 1e-20))
        self.sqrt_recip_alphas_cumprod = to_jt_buf(np.sqrt(1.0 / alphas_cumprod))
        self.sqrt_recipm1_alphas_cumprod = to_jt_buf(np.sqrt(1.0 / alphas_cumprod - 1.0))

        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod) \
                             + self.v_posterior * betas
        posterior_variance = posterior_variance.astype(np.float32)

        self.posterior_variance = to_jt_buf(posterior_variance)
        self.posterior_log_variance_clipped = to_jt_buf(np.log(np.maximum(posterior_variance, 1e-20)))

        self.posterior_mean_coef1 = to_jt_buf(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)
        )
        self.posterior_mean_coef2 = to_jt_buf(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)
        )

        # lvlb weights
        if self.parameterization == "eps":
            eps = 1e-20
            den = 2 * np.maximum(posterior_variance, eps) * alphas * np.maximum(1.0 - alphas_cumprod, eps)
            lvlb_weights = (betas ** 2) / den
        elif self.parameterization == "x0":
            lvlb_weights = 0.5 * np.sqrt(alphas_cumprod) / (2. * 1 - alphas_cumprod)
        else:
            raise NotImplementedError

        lvlb_weights = lvlb_weights.astype(np.float32)
        if lvlb_weights.shape[0] > 1:
            lvlb_weights[0] = lvlb_weights[1]
        self.lvlb_weights = to_jt_buf(lvlb_weights)

    # -----------------------
    # losses
    # -----------------------
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = jt.randn(x_start.shape).astype(self.dtype)
        return extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + \
               extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise

    def get_loss(self, pred, target, mean=True):
        if self.loss_type == "l1":
            loss = jt.abs(target - pred)
            return loss.mean() if mean else loss
        elif self.loss_type == "l2":
            loss = (target - pred) ** 2
            return loss.mean() if mean else loss
        else:
            raise NotImplementedError(f"unknown loss type '{self.loss_type}'")

    def p_losses(self, x_info, t, c_info, noise=None):
        x = _to_jt_var(x_info["x"], dtype=self.dtype)
        if noise is None:
            noise = jt.randn(x.shape).astype(self.dtype)

        x_noisy = self.q_sample(x_start=x, t=t, noise=noise)
        x_info = {"type": x_info["type"], "x": x_noisy}

        model_output = self.apply_model(x_info, t, c_info)

        if self.parameterization == "x0":
            target = x
        elif self.parameterization == "eps":
            target = noise
        else:
            raise NotImplementedError

        bs = model_output.shape[0]
        loss_simple = self.get_loss(model_output, target, mean=False).reshape([bs, -1]).mean(-1)

        logvar_t = extract_into_tensor(self.logvar, t, [bs]).astype(self.dtype)
        loss = loss_simple / jt.exp(logvar_t) + logvar_t

        loss_dict = {}
        loss_dict["loss_simple"] = loss_simple.mean()

        if self.learn_logvar:
            loss_dict["loss_gamma"] = loss.mean()
            loss_dict["logvar"] = self.logvar.mean()

        loss = self.l_simple_weight * loss.mean()

        loss_vlb = self.get_loss(model_output, target, mean=False).reshape([bs, -1]).mean(-1)
        lvlb_t = extract_into_tensor(self.lvlb_weights, t, [bs]).astype(self.dtype)
        loss_vlb = (lvlb_t * loss_vlb).mean()

        loss_dict["loss_vlb"] = loss_vlb
        loss_dict["Loss"] = loss
        return loss, loss_dict

    # -----------------------
    # torch hub adapters (inference-only)
    # -----------------------
    def vae_encode(self, x, which, **kwargs):
        z = self.torch_hub.vae_encode(x, which, **kwargs)
        if self.latent_scale_factor is not None:
            if self.latent_scale_factor.get(which, None) is not None:
                scale = self.latent_scale_factor[which]
                return scale * z
        return z

    def vae_decode(self, z, which, **kwargs):
        if self.latent_scale_factor is not None:
            if self.latent_scale_factor.get(which, None) is not None:
                scale = self.latent_scale_factor[which]
                z = 1./scale * z
        x = self.torch_hub.vae_decode(z, which, **kwargs)
        return x

    def ctx_encode(self, x, which, **kwargs):
        if which.find('vae_') == 0:
            return self.vae_encode(x, which[4:], **kwargs)
        else:
            return self.torch_hub.ctx_encode(x, which, **kwargs)


    # -----------------------
    # diffuser apply
    # -----------------------
    def check_diffuser(self):
        order = None
        for idx, (_, diffuseri) in enumerate(self.diffuser.items()):
            if not hasattr(diffuseri, "layer_order"):
                continue
            if idx == 0:
                order = diffuseri.layer_order
            else:
                if order != diffuseri.layer_order:
                    return False
        return True

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self.parameters())
            if context is not None:
                print_log(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.parameters())
                if context is not None:
                    print_log(f"{context}: Restored training weights")

    def on_train_batch_start(self, x):
        return

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self)

    def apply_model(self, x_info, timesteps, c_info):
        x_type, x = x_info["type"], _to_jt_var(x_info["x"], dtype=self.dtype)
        c_type, c = c_info["type"], _to_jt_var(c_info["c"], dtype=self.dtype)

        hs = []

        glayer_ptr = x_type if self.global_layer_ptr is None else self.global_layer_ptr
        model_channels = self.diffuser[glayer_ptr].model_channels

        t_emb = timestep_embedding(timesteps, model_channels, repeat_only=False).astype(self.dtype)
        emb = self.diffuser[glayer_ptr].time_embed(t_emb)

        d_iter = iter(self.diffuser[x_type].data_blocks)
        c_iter = iter(self.diffuser[c_type].context_blocks)

        i_order = self.diffuser[x_type].i_order
        m_order = self.diffuser[x_type].m_order
        o_order = self.diffuser[x_type].o_order

        h = x
        for ltype in i_order:
            if ltype == "d":
                module = next(d_iter)
                h = module(h, emb, None)
            elif ltype == "c":
                module = next(c_iter)
                h = module(h, emb, c)
            elif ltype == "save_hidden_feature":
                hs.append(h)

        for ltype in m_order:
            if ltype == "d":
                module = next(d_iter)
                h = module(h, emb, None)
            elif ltype == "c":
                module = next(c_iter)
                h = module(h, emb, c)

        for ltype in o_order:
            if ltype == "load_hidden_feature":
                h = jt.concat([h, hs.pop()], dim=1)
            elif ltype == "d":
                module = next(d_iter)
                h = module(h, emb, None)
            elif ltype == "c":
                module = next(c_iter)
                h = module(h, emb, c)

        return h

    def context_mixing(self, x, emb, context_module_list, context_info_list, mixing_type):
        nm = len(context_module_list)
        nc = len(context_info_list)
        assert nm == nc
        context = [_to_jt_var(ci["c"], dtype=self.dtype) for ci in context_info_list]
        cratio = np.array([ci["ratio"] for ci in context_info_list], dtype=np.float32)
        cratio = cratio / (cratio.sum() + 1e-12)

        if mixing_type == "attention":
            h = None
            for module, c, r in zip(context_module_list, context, cratio):
                hi = module(x, emb, c) * float(r)
                h = hi if h is None else (h + hi)
            return h
        elif mixing_type == "layer":
            ni = npr.choice(nm, p=cratio)
            module = context_module_list[int(ni)]
            c = context[int(ni)]
            return module(x, emb, c)
        else:
            raise ValueError(f"Unknown mixing_type: {mixing_type}")

    def apply_model_multicontext(self, x_info, timesteps, c_info_list, mixing_type="attention"):
        x_type, x = x_info["type"], _to_jt_var(x_info["x"], dtype=self.dtype)
        hs = []

        model_channels = self.diffuser[x_type].model_channels
        t_emb = timestep_embedding(timesteps, model_channels, repeat_only=False).astype(self.dtype)
        emb = self.diffuser[x_type].time_embed(t_emb)

        d_iter = iter(self.diffuser[x_type].data_blocks)
        c_iter_list = [iter(self.diffuser[cinfo["type"]].context_blocks) for cinfo in c_info_list]

        i_order = self.diffuser[x_type].i_order
        m_order = self.diffuser[x_type].m_order
        o_order = self.diffuser[x_type].o_order

        h = x
        for ltype in i_order:
            if ltype == "d":
                module = next(d_iter)
                h = module(h, emb, None)
            elif ltype == "c":
                module_list = [next(ci) for ci in c_iter_list]
                h = self.context_mixing(h, emb, module_list, c_info_list, mixing_type)
            elif ltype == "save_hidden_feature":
                hs.append(h)

        for ltype in m_order:
            if ltype == "d":
                module = next(d_iter)
                h = module(h, emb, None)
            elif ltype == "c":
                module_list = [next(ci) for ci in c_iter_list]
                h = self.context_mixing(h, emb, module_list, c_info_list, mixing_type)

        for ltype in o_order:
            if ltype == "load_hidden_feature":
                h = jt.concat([h, hs.pop()], dim=1)
            elif ltype == "d":
                module = next(d_iter)
                h = module(h, emb, None)
            elif ltype == "c":
                module_list = [next(ci) for ci in c_iter_list]
                h = self.context_mixing(h, emb, module_list, c_info_list, mixing_type)

        return h

    # -----------------------
    # state_dict / load_state_dict
    # -----------------------
    def state_dict(self):
        sd = {}

        # buffers
        for n in self._BUFFER_NAMES:
            if hasattr(self, n):
                v = getattr(self, n)
                if isinstance(v, jt.Var):
                    sd[n] = v.numpy()
        # diffuser params
        for dname, dmod in self.diffuser.items():
            if hasattr(dmod, "state_dict"):
                sub = dmod.state_dict()
                for k, v in sub.items():
                    sd[f"diffuser.{dname}.{k}"] = _to_numpy_any(v)
        return sd

    def save_state_dict(self):
        return self.state_dict()

    def load_state_dict(self, state_dict, strict=False):
        sd = state_dict
        if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
            sd = sd["state_dict"]
        sd = _strip_module_prefix(sd)

        loaded = set()
        missing = []
        unexpected = []

        # # 1) buffers
        # for n in self._BUFFER_NAMES:
        #     if n in sd:
        #         v = jt.array(_to_numpy_any(sd[n])).astype(self.dtype)
        #         if (n != "logvar") or (not self.learn_logvar):
        #             v.stop_grad()
        #         setattr(self, n, v)
        #         loaded.add(n)
        #     else:
        #         if strict:
        #             missing.append(n)

        # 2) diffuser params by prefix
        for dname, dmod in self.diffuser.items():
            prefix = f"diffuser.{dname}."
            sub = {}
            for k, v in sd.items():
                if isinstance(k, str) and k.startswith(prefix):
                    sub[k[len(prefix):]] = v
            if len(sub) == 0:
                continue
            if hasattr(dmod, "load_state_dict"):
                dmod.load_state_dict(sub, strict=strict) if "strict" in dmod.load_state_dict.__code__.co_varnames else dmod.load_state_dict(sub)
                for k in sub.keys():
                    loaded.add(prefix + k)
            else:
                if strict:
                    missing.append(f"diffuser.{dname}.* (no load_state_dict in submodule)")

        # 3) unexpected keys
        for k in sd.keys():
            if k not in loaded and not any(str(k).startswith(f"diffuser.{dn}.") for dn in self.diffuser.keys()):
                unexpected.append(k)
                
        print_log({"missing_keys": missing, "unexpected_keys": unexpected})
        
        return {"missing_keys": missing, "unexpected_keys": unexpected}
