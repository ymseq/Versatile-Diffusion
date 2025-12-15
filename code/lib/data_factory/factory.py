import os
import json
from typing import Any, Callable, Dict, List, Optional

import torch
from torch.utils.data import Dataset
from torch.utils.data import distributed, sampler
from torchvision import transforms as T
from PIL import Image


DATASET_REGISTRY: Dict[str, Callable] = {}
TRANSFORM_REGISTRY: Dict[str, Callable] = {}
COLLATE_REGISTRY: Dict[str, Callable] = {}
SAMPLER_REGISTRY: Dict[str, Callable] = {}


def register(registry: Dict[str, Callable], name: str) -> Callable:
    def decorator(obj: Callable) -> Callable:
        registry[name] = obj
        return obj

    return decorator


def register_dataset(name: str) -> Callable:
    return register(DATASET_REGISTRY, name)


def register_transform(name: str) -> Callable:
    return register(TRANSFORM_REGISTRY, name)


def register_collate(name: str) -> Callable:
    return register(COLLATE_REGISTRY, name)


def register_sampler(name: str) -> Callable:
    return register(SAMPLER_REGISTRY, name)


def load_meta_file(meta_path: str) -> List[Dict[str, Any]]:
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"Meta/annotation file not found: {meta_path}")
    if meta_path.endswith(".json"):
        with open(meta_path, "r") as f:
            return json.load(f)
    records: List[Dict[str, Any]] = []
    with open(meta_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            if "\t" in line:
                path, caption = line.strip().split("\t", 1)
                records.append({"image": path, "caption": caption})
            else:
                records.append(json.loads(line))
    return records


class BaseImageTextDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        meta_path: str,
        transform: Optional[Callable] = None,
        use_text: bool = True,
        tokenizer: Optional[Callable[[str], Dict[str, Any]]] = None,
        image_key: str = "image",
        text_key: str = "caption",
        **kwargs,
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.meta_path = meta_path
        self.transform = transform
        self.use_text = use_text
        self.tokenizer = tokenizer
        self.image_key = image_key
        self.text_key = text_key
        self.records = load_meta_file(meta_path)

    def __len__(self) -> int:
        return len(self.records)

    def _resolve_path(self, record: Dict[str, Any]) -> str:
        img_path = record.get(self.image_key)
        if img_path is None:
            raise KeyError(f"'{self.image_key}' not found in record")
        if os.path.isabs(img_path):
            return img_path
        return os.path.join(self.data_root, img_path)

    def _load_image(self, image_path: str) -> torch.Tensor:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img

    def _process_text(self, record: Dict[str, Any]) -> Dict[str, Any]:
        if not self.use_text:
            return {}
        caption = record.get(self.text_key, "")
        if self.tokenizer:
            tokens = self.tokenizer(caption)
            return {"text": caption, **tokens}
        return {"text": caption}

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.records[idx]
        image_path = self._resolve_path(record)
        sample: Dict[str, Any] = {"image": self._load_image(image_path)}
        sample.update(self._process_text(record))
        return sample


@register_dataset("CocoCaptionDataset")
class CocoCaptionDataset(BaseImageTextDataset):
    def __init__(self, *args, ann_path: str = None, **kwargs) -> None:
        meta_path = ann_path or kwargs.pop("meta_path", None)
        if meta_path is None:
            raise ValueError("CocoCaptionDataset requires ann_path or meta_path")
        super().__init__(*args, meta_path=meta_path, **kwargs)


@register_dataset("LaionSubsetDataset")
class LaionSubsetDataset(BaseImageTextDataset):
    pass


@register_dataset("ImageVariationDataset")
class ImageVariationDataset(BaseImageTextDataset):
    def __init__(
        self,
        *args,
        num_variations: int = 1,
        context_mix: str = "image",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.num_variations = num_variations
        self.context_mix = context_mix

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = super().__getitem__(idx)
        sample["num_variations"] = self.num_variations
        sample["context_mix"] = self.context_mix
        return sample


@register_transform("default_train")
def default_train_transform(image_size: int = 256, **kwargs) -> Callable:
    return T.Compose(
        [
            T.Resize(image_size + 32),
            T.RandomCrop(image_size),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
        ]
    )


@register_transform("default_eval")
def default_eval_transform(image_size: int = 256, **kwargs) -> Callable:
    return T.Compose(
        [
            T.Resize(image_size + 32),
            T.CenterCrop(image_size),
            T.ToTensor(),
        ]
    )


def build_transform(cfg: Any) -> Optional[Callable]:
    if cfg is None:
        return None
    if isinstance(cfg, str):
        fn = TRANSFORM_REGISTRY.get(cfg)
        if fn is None:
            raise KeyError(f"Transform {cfg} is not registered")
        return fn()
    if isinstance(cfg, dict):
        name = cfg.get("name") or cfg.get("type")
        params = {k: v for k, v in cfg.items() if k not in {"name", "type"}}
        fn = TRANSFORM_REGISTRY.get(name)
        if fn is None:
            raise KeyError(f"Transform {name} is not registered")
        return fn(**params)
    return cfg


@register_collate("default")
def default_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    images = torch.stack([b["image"] for b in batch], dim=0)
    output: Dict[str, Any] = {"image": images}
    if any("text" in b for b in batch):
        texts = [b.get("text", "") for b in batch]
        output["text"] = texts
    if any("input_ids" in b for b in batch):
        lengths = [len(b.get("input_ids", [])) for b in batch]
        max_len = max(lengths) if lengths else 0
        padded = []
        attention_masks = []
        for b in batch:
            tokens = b.get("input_ids", [])
            padding = [0] * max(0, max_len - len(tokens))
            padded.append(torch.tensor(tokens + padding, dtype=torch.long))
            attention_mask = [1] * len(tokens) + [0] * (max_len - len(tokens))
            attention_masks.append(torch.tensor(attention_mask, dtype=torch.long))
        output["input_ids"] = torch.stack(padded, dim=0)
        output["attention_mask"] = torch.stack(attention_masks, dim=0)
    for key in ["num_variations", "context_mix"]:
        if any(key in b for b in batch):
            output[key] = [b.get(key) for b in batch]
    return output


@register_sampler("default_train")
def distributed_train_sampler(dataset: Dataset, shuffle: bool = True, **kwargs):
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return distributed.DistributedSampler(dataset, shuffle=shuffle)
    return sampler.RandomSampler(dataset) if shuffle else sampler.SequentialSampler(dataset)


@register_sampler("default_eval")
def distributed_eval_sampler(dataset: Dataset, shuffle: bool = False, **kwargs):
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return distributed.DistributedSampler(dataset, shuffle=shuffle)
    return sampler.SequentialSampler(dataset)


@register_sampler("sequential")
def sequential_sampler(dataset: Dataset, **kwargs):
    return sampler.SequentialSampler(dataset)


def get_dataset() -> Callable:
    def _builder(cfg: Any, tokenizer: Optional[Callable] = None):
        ds_type = cfg.get("type") if isinstance(cfg, dict) else getattr(cfg, "type", None)
        if ds_type is None:
            raise ValueError("Dataset config must include a type field")
        ds_cls = DATASET_REGISTRY.get(ds_type)
        if ds_cls is None:
            raise KeyError(f"Dataset {ds_type} is not registered")
        cfg_dict = cfg if isinstance(cfg, dict) else dict(cfg)
        transform = build_transform(cfg_dict.get("transform") or cfg_dict.get("preprocess"))
        cfg_dict = {**cfg_dict, "transform": transform, "tokenizer": tokenizer}
        return ds_cls(**cfg_dict)

    return _builder


def get_sampler() -> Callable:
    def _builder(dataset: Dataset, cfg: Any = "default_eval"):
        if isinstance(cfg, str):
            name = cfg
            params = {}
        else:
            name = cfg.get("name") or cfg.get("type") or cfg.get("sampler") or "default_eval"
            params = {k: v for k, v in cfg.items() if k not in {"name", "type", "sampler"}}
        sampler_fn = SAMPLER_REGISTRY.get(name)
        if sampler_fn is None:
            raise KeyError(f"Sampler {name} is not registered")
        return sampler_fn(dataset=dataset, **params)

    return _builder


def collate(name: str = "default") -> Callable:
    fn = COLLATE_REGISTRY.get(name)
    if fn is None:
        raise KeyError(f"Collate fn {name} is not registered")
    return fn


def get_transform() -> Callable:
    return build_transform


def get_loader() -> Callable:
    def _builder(dataset: Dataset, batch_size: int, sampler_obj: Any, num_workers: int, pin_memory: bool = False, collate_fn: Optional[Callable] = None):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler_obj,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn or default_collate,
        )

    return _builder


def get_estimator():
    def _missing(*args, **kwargs):
        raise NotImplementedError("Estimator factory is not implemented yet.")

    return _missing


def get_formatter():
    def _missing(*args, **kwargs):
        raise NotImplementedError("Formatter factory is not implemented yet.")

    return _missing
