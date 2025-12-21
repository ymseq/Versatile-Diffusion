import os
import os.path as osp
import json
import numpy as np

EVAL_REGISTRY = {}

def register_evaluator(name: str):
    def deco(cls):
        EVAL_REGISTRY[name] = cls
        return cls
    return deco


def get_evaluator():
    def _builder(cfg):
        def _get(obj, key, default=None):
            if obj is None:
                return default
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        name = _get(cfg, "name", None)
        if name is None:
            raise KeyError("evaluator cfg must have one of fields: name/type/symbol")

        if name not in EVAL_REGISTRY:
            raise KeyError(f"Evaluator '{name}' not registered. Available: {list(EVAL_REGISTRY.keys())}")

        cls = EVAL_REGISTRY[name]
        return cls(cfg)

    return _builder


class BaseEvaluator:
    def __init__(self, cfg=None):
        self.cfg = cfg
        self.sample_n = None
        self._summary = {}
        self.clear_data()

    def clear_data(self):

        self._sum = {}
        self._cnt = {}
        self._summary = {}

    def set_sample_n(self, sample_n: int):
        self.sample_n = int(sample_n) if sample_n is not None else None

    def add_batch(self, **rv):
        raise NotImplementedError

    def compute(self):
        raise NotImplementedError

    def one_line_summary(self):
        msg = " | ".join([f"{k}={v:.6f}" if isinstance(v, (int, float)) else f"{k}={v}"
                          for k, v in self._summary.items()])
        print(msg)

    def save(self, log_dir: str):
        os.makedirs(log_dir, exist_ok=True)
        path = osp.join(log_dir, "eval_summary.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._summary, f, ensure_ascii=False, indent=2)


    @staticmethod
    def _to_float(x):
        try:
            import jittor as jt
            if isinstance(x, jt.Var):
                return float(x.detach().mean().item())
        except Exception:
            pass

        try:
            import torch
            if isinstance(x, torch.Tensor):
                return float(x.detach().float().mean().item())
        except Exception:
            pass
        # numpy
        if isinstance(x, np.ndarray):
            return float(np.mean(x))
        # python number
        if isinstance(x, (int, float)):
            return float(x)

        raise TypeError(f"Cannot convert type {type(x)} to float")

@register_evaluator("avg_loss")
class AvgLossEvaluator(BaseEvaluator):
    def __init__(self, cfg=None):
        super().__init__(cfg=cfg)
        self.key_loss = getattr(cfg, "key_loss", "loss") if cfg is not None else "loss"
        self.use_neg = getattr(cfg, "use_negative", True) if cfg is not None else True

    def clear_data(self):
        super().clear_data()
        self._loss_sum = 0.0
        self._loss_cnt = 0
        self._correct_sum = 0
        self._total_sum = 0

    def add_batch(self, **rv):
        if self.key_loss not in rv:
            raise KeyError(f"AvgLossEvaluator expects '{self.key_loss}' in rv keys={list(rv.keys())}")

        loss_v = self._to_float(rv[self.key_loss])


        bs = rv.get("bs", None)
        if bs is None:
            bs = rv.get("n", rv.get("num", 1))
        bs = int(bs)

        self._loss_sum += loss_v * bs
        self._loss_cnt += bs

        if "correct" in rv and "total" in rv:
            self._correct_sum += int(rv["correct"])
            self._total_sum += int(rv["total"])

    def compute(self):
        denom = self._loss_cnt if self._loss_cnt > 0 else 1
        avg_loss = self._loss_sum / denom

        self._summary = {"avg_loss": float(avg_loss)}

        if self._total_sum > 0:
            acc = self._correct_sum / max(1, self._total_sum)
            self._summary["acc"] = float(acc)

        eval_rv = -avg_loss if self.use_neg else avg_loss
        self._summary["eval_rv"] = float(eval_rv)
        return float(eval_rv)
