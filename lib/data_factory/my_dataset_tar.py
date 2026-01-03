import os
import io
import re
import glob
import tarfile
import random
from typing import List, Tuple, Optional, Dict

from PIL import Image
from jittor.dataset.dataset import Dataset
import jittor.transform as transform


class MyVDDatasetTar(Dataset):

    IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp")
    TXT_EXTS = (".txt",)

    def __init__(self, cfg):
        super().__init__()
        self.root = cfg.root

        tar_glob = getattr(cfg, "tar_glob", "*.tar")
        pattern = tar_glob if os.path.isabs(tar_glob) else os.path.join(self.root, tar_glob)
        all_tar_paths = glob.glob(pattern)
        if not all_tar_paths:
            raise FileNotFoundError(f"No tar shards found: {pattern}")

        def natural_key(p: str):
            b = os.path.basename(p)
            m = re.search(r"(\d+)", b)
            return int(m.group(1)) if m else b

        all_tar_paths = sorted(all_tar_paths, key=natural_key)

        split = getattr(cfg, "split", "train")
        val_ratio = float(getattr(cfg, "val_ratio", 0.01))
        split_seed = int(getattr(cfg, "split_seed", 42))
        shuffle_shards_before_split = bool(getattr(cfg, "shuffle_shards_before_split", True))

        tar_paths = all_tar_paths[:]
        if shuffle_shards_before_split:
            rng = random.Random(split_seed)
            rng.shuffle(tar_paths)

        n = len(tar_paths)
        if n < 2:
            raise ValueError(f"Need at least 2 tar shards to split train/val, got {n}.")

        n_val = int(n * val_ratio)
        if n_val < 1:
            n_val = 1
        if n_val >= n:
            n_val = n - 1

        n_train = n - n_val

        if split == "train":
            self.tar_paths = tar_paths[:n_train]
        elif split == "val":
            self.tar_paths = tar_paths[n_train:]
        else:
            raise ValueError(f"Unknown split: {split}, expected 'train' or 'val'")

        size = getattr(cfg, "image_size", 512)
        self.transform = transform.Compose([
            transform.Resize(size),
            transform.CenterCrop(size),
            transform.ToTensor(),
            transform.ImageNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        self.index: List[Tuple[str, str, str]] = []
        self._build_index(natural_key)

        shuffle_index = bool(getattr(cfg, "shuffle_index", split == "train"))
        if shuffle_index:
            rng = random.Random(split_seed + 999)
            rng.shuffle(self.index)

        self.cache_tar = bool(getattr(cfg, "cache_tar", True))
        self._cached_tar_path: Optional[str] = None
        self._cached_tar: Optional[tarfile.TarFile] = None

        self.set_attrs(total_len=len(self.index))

    def __len__(self):
        return len(self.index)

    def _build_index(self, natural_key):
        for tar_path in self.tar_paths:
            with tarfile.open(tar_path, "r:*") as tf:
                members = tf.getmembers()

                bucket: Dict[str, Dict[str, str]] = {}
                for m in members:
                    if not m.isfile():
                        continue
                    name = m.name
                    base, ext = os.path.splitext(name)
                    ext = ext.lower()
                    if ext in self.IMG_EXTS:
                        bucket.setdefault(base, {})["img"] = name
                    elif ext in self.TXT_EXTS:
                        bucket.setdefault(base, {})["txt"] = name

                bases = [b for b, parts in bucket.items() if ("img" in parts and "txt" in parts)]
                bases.sort(key=natural_key)

                for b in bases:
                    parts = bucket[b]
                    self.index.append((tar_path, parts["img"], parts["txt"]))

    def _get_tar(self, tar_path: str) -> tarfile.TarFile:
        if not self.cache_tar:
            return tarfile.open(tar_path, "r:*")

        if self._cached_tar is not None and self._cached_tar_path == tar_path:
            return self._cached_tar

        if self._cached_tar is not None:
            try:
                self._cached_tar.close()
            except Exception:
                pass

        self._cached_tar_path = tar_path
        self._cached_tar = tarfile.open(tar_path, "r:*")
        return self._cached_tar

    @staticmethod
    def _read_member_bytes(tf: tarfile.TarFile, member_name: str) -> bytes:
        f = tf.extractfile(member_name)
        if f is None:
            raise FileNotFoundError(f"Member not found in tar: {member_name}")
        data = f.read()
        f.close()
        return data

    def __getitem__(self, idx):
        tar_path, img_name, txt_name = self.index[idx]
        tf = self._get_tar(tar_path)

        try:
            img_bytes = self._read_member_bytes(tf, img_name)
            img = Image.open(io.BytesIO(img_bytes))
            img.load()
            img = img.convert("RGB")
            image = self.transform(img)

            txt_bytes = self._read_member_bytes(tf, txt_name)
            caption = txt_bytes.decode("utf-8", errors="ignore").strip()
        finally:
            if not self.cache_tar:
                try:
                    tf.close()
                except Exception:
                    pass

        return image, caption

    def __del__(self):
        if self._cached_tar is not None:
            try:
                self._cached_tar.close()
            except Exception:
                pass
