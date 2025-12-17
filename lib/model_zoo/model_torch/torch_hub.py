# Keep using troch
import os
import json
import hashlib
import numpy as np

import torch
import torch.nn as nn

from lib.model_zoo.common.get_model import get_model


class TorchVAEContextHub:
    """
    Torch VAE + Context Hub
    """

    def __init__(
        self,
        vae_cfg_list=None,
        ctx_cfg_list=None,
        cache_dir=None,
    ):
        self.cache_dir = cache_dir

        self.vae = self._build_moduledict(vae_cfg_list or [])
        self.ctx = self._build_moduledict(ctx_cfg_list or [])

        self._freeze_all(self.vae)
        self._freeze_all(self.ctx)

    def _build_moduledict(self, cfg_list):
        md = nn.ModuleDict()
        for name, cfg in cfg_list:
            md[name] = get_model()(cfg)
        return md

    def _freeze_all(self, md):
        for _, m in md.items():
            if isinstance(m, nn.Module):
                m.eval()
                for p in m.parameters():
                    p.requires_grad_(False)

    def to(self, device):
        self.device = device
        for _, m in self.vae.items():
            m.to(device)
        for _, m in self.ctx.items():
            m.to(device)
        return self

    def half(self):
        self.fp16 = True
        for _, m in self.vae.items():
            if hasattr(m, "fp16"):
                m.fp16 = True
            m.half()
        for _, m in self.ctx.items():
            if hasattr(m, "fp16"):
                m.fp16 = True
            m.half()
        return self

    def float(self):
        self.fp16 = False
        for _, m in self.vae.items():
            if hasattr(m, "fp16"):
                m.fp16 = False
            m.float()
        for _, m in self.ctx.items():
            if hasattr(m, "fp16"):
                m.fp16 = False
            m.float()
        return self

    # ---------------- cache helpers ----------------
    def _hash_key(self, payload: dict) -> str:
        s = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(s.encode("utf-8")).hexdigest()

    def _cache_path(self, kind: str, key: str) -> str:
        os.makedirs(self.cache_dir, exist_ok=True)
        return os.path.join(self.cache_dir, f"{kind}-{key}.npy")

    def _maybe_load_cache(self, kind: str, payload: dict):
        if self.cache_dir is None:
            return None, None
        key = self._hash_key(payload)
        path = self._cache_path(kind, key)
        if os.path.exists(path):
            return np.load(path), key
        return None, key

    def _maybe_save_cache(self, kind: str, key: str, arr: np.ndarray):
        if self.cache_dir is None:
            return
        np.save(self._cache_path(kind, key), arr)

    # ---------------- conversion ----------------
    def _as_torch(self, x):
        if torch.is_tensor(x):
            return x
        if hasattr(x, "numpy"):
            x = x.numpy()
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x)
        return torch.tensor(x)

    # ---------------- main apis ----------------
    @torch.inference_mode()
    def vae_encode(self, x, which, cache_id=None, **kwargs) -> np.ndarray:
        payload = {"kind": "vae_encode", "which": which, "cache_id": cache_id, "fp16": self.fp16}
        cached, key = self._maybe_load_cache("z", payload) if cache_id is not None else (None, None)
        if cached is not None:
            return cached

        xt = self._as_torch(x).to(self.device, non_blocking=True)
        if self.fp16:
            xt = xt.half()
        z = self.vae[which].encode(xt, **kwargs)
        z = z.detach().to("cpu", non_blocking=True).contiguous()

        if cache_id is not None:
            self._maybe_save_cache("z", key, z)
        return z

    @torch.inference_mode()
    def vae_decode(self, z, which, **kwargs):
        zt = self._as_torch(z).to(self.device, non_blocking=True)
        if self.fp16:
            zt = zt.half()
        return self.vae[which].decode(zt, **kwargs)

    @torch.inference_mode()
    def ctx_encode(self, x, which, cache_id=None, **kwargs) -> np.ndarray:
        payload = {"kind": "ctx_encode", "which": which, "cache_id": cache_id, "fp16": self.fp16}
        cached, key = self._maybe_load_cache(f"c_{which}", payload) if cache_id is not None else (None, None)
        if cached is not None:
            return cached

        out = self.ctx[which].encode(x, **kwargs)
        if torch.is_tensor(out):
            out = out.detach().to("cpu", non_blocking=True).contiguous()
        else:
            out = np.asarray(out)

        if cache_id is not None:
            self._maybe_save_cache(f"c_{which}", key, out)
        return out

