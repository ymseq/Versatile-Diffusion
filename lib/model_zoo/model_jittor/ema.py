import jittor as jt
from jittor import nn


class LitEma(nn.Module):
    """
    Exponential Moving Average for Jittor.

    Key points:
    - No .item() (avoid cupy / sync issues)
    - shadow stored in fp32 by default for stability
    - in-place assign to reduce allocations
    """

    def __init__(self, model, decay=0.9999, use_num_updates=True, shadow_dtype=jt.float32):
        super().__init__()
        if not (0.0 <= float(decay) <= 1.0):
            raise ValueError("Decay must be between 0 and 1")

        self.decay = float(decay)
        self.use_num_updates = bool(use_num_updates)
        self.num_updates = 0  # python int, no device sync
        self.shadow_dtype = shadow_dtype

        # name mapping & shadow weights
        self.m_name2s_name = {}
        self.shadow = {}

        for name, p in model.named_parameters():
            if getattr(p, "requires_grad", True):
                # safer than removing dots entirely
                s_name = name.replace(".", "__")
                if s_name in self.shadow:
                    raise ValueError(f"EMA shadow name collision: {name} -> {s_name}")
                self.m_name2s_name[name] = s_name

                # store fp32 shadow by default, stop_grad
                sp = p.detach()
                if shadow_dtype is not None:
                    sp = sp.cast(shadow_dtype)
                self.shadow[s_name] = sp.clone().stop_grad()

        self.collected_params = []

    def _get_decay(self):
        """Warmup decay schedule (python scalar)."""
        if not self.use_num_updates:
            return self.decay
        self.num_updates += 1
        nu = float(self.num_updates)
        warm = (1.0 + nu) / (10.0 + nu)
        return min(self.decay, warm)

    def execute(self, model):
        d = self._get_decay()
        one_minus = 1.0 - d

        for name, p in model.named_parameters():
            if not getattr(p, "requires_grad", True):
                continue

            sname = self.m_name2s_name[name]
            sp = self.shadow[sname]

            # detach current param
            cur = p.detach()
            if self.shadow_dtype is not None and cur.dtype != self.shadow_dtype:
                cur = cur.cast(self.shadow_dtype)

            # in-place: sp = d*sp + (1-d)*cur
            # (avoid creating many temporary vars)
            sp.assign(sp * d + cur * one_minus)
            sp.stop_grad()
            # dict already holds reference; no need to reassign

    def copy_to(self, model):
        """Copy EMA weights to model parameters (cast to param dtype)."""
        for name, p in model.named_parameters():
            if not getattr(p, "requires_grad", True):
                continue
            sname = self.m_name2s_name[name]
            sp = self.shadow[sname]
            if sp.dtype != p.dtype:
                p.assign(sp.cast(p.dtype))
            else:
                p.assign(sp)

    def store(self, parameters):
        """Store current parameters for later restore."""
        self.collected_params = [p.detach().clone().stop_grad() for p in parameters]

    def restore(self, parameters):
        """Restore parameters previously stored by store()."""
        for c_param, p in zip(self.collected_params, parameters):
            p.assign(c_param)
