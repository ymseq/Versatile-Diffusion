import os
from typing import Any, Callable, Dict

import torch


EVALUATOR_REGISTRY: Dict[str, Callable] = {}


def register_evaluator(name: str) -> Callable:
    def decorator(cls: Callable) -> Callable:
        EVALUATOR_REGISTRY[name] = cls
        return cls

    return decorator


class BaseEvaluator(object):
    def __init__(self, **kwargs) -> None:
        self.results = []
        self.sample_n = 0

    def add_batch(self, **kwargs):
        self.results.append(kwargs)

    def set_sample_n(self, n: int):
        self.sample_n = n

    def compute(self) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {}
        if not self.results:
            return metrics
        for batch in self.results:
            for key, value in batch.items():
                if isinstance(value, (float, int, torch.Tensor)):
                    if isinstance(value, torch.Tensor):
                        value = value.detach().float().mean().item()
                    metrics.setdefault(key, []).append(float(value))
        for key, values in metrics.items():
            metrics[key] = float(sum(values) / max(len(values), 1))
        return metrics

    def one_line_summary(self):
        summary = ", ".join([f"{k}: {v}" for k, v in self.compute().items()])
        print(summary)

    def clear_data(self):
        self.results.clear()

    def save(self, log_dir: str):
        os.makedirs(log_dir, exist_ok=True)
        path = os.path.join(log_dir, "metrics.pt")
        torch.save(self.compute(), path)


@register_evaluator("t2i")
class TextToImageEvaluator(BaseEvaluator):
    def __init__(self, fid_stats_path: str = None, clip_model_name: str = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.fid_stats_path = fid_stats_path
        self.clip_model_name = clip_model_name
        if fid_stats_path is not None and not os.path.exists(fid_stats_path):
            raise FileNotFoundError(f"FID stats not found: {fid_stats_path}")

    def compute(self) -> Dict[str, Any]:
        metrics = {"fid": 0.0, "clip_score": 0.0}
        metrics.update(super().compute())
        return metrics


@register_evaluator("image_variation")
class ImageVariationEvaluator(BaseEvaluator):
    def __init__(self, lpips_weight: str = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.lpips_weight = lpips_weight
        if lpips_weight is not None and not os.path.exists(lpips_weight):
            raise FileNotFoundError(f"LPIPS weight not found: {lpips_weight}")

    def compute(self) -> Dict[str, Any]:
        metrics = {"fid": 0.0, "lpips": 0.0, "clip_i2i": 0.0}
        metrics.update(super().compute())
        return metrics


@register_evaluator("i2t")
class ImageToTextEvaluator(BaseEvaluator):
    def __init__(self, reference_path: str = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.reference_path = reference_path
        if reference_path is not None and not os.path.exists(reference_path):
            raise FileNotFoundError(f"Reference captions not found: {reference_path}")

    def compute(self) -> Dict[str, Any]:
        metrics = {"bleu": 0.0, "cider": 0.0, "meteor": 0.0}
        metrics.update(super().compute())
        return metrics


def get_evaluator(name_or_cfg: Any):
    if isinstance(name_or_cfg, str):
        cfg = {"type": name_or_cfg}
    else:
        cfg = name_or_cfg
    eval_type = cfg.get("type") or cfg.get("name")
    if eval_type is None:
        raise ValueError("Evaluator config must include a type or name")
    evaluator_cls = EVALUATOR_REGISTRY.get(eval_type)
    if evaluator_cls is None:
        raise KeyError(f"Evaluator {eval_type} is not registered")
    params = {k: v for k, v in cfg.items() if k not in {"type", "name"}}
    return evaluator_cls(**params)


def get_evaluator_builder() -> Callable:
    def _builder(name_or_cfg: Any):
        return get_evaluator(name_or_cfg)

    return _builder
