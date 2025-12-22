# lib/data_factory/my_vd_dataset_tar.py
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
    """
    读取 img2dataset/WebDataset 生成的 tar shards：
      - 图片：*.jpg / *.jpeg / *.png / *.webp
      - 文本：*.txt（caption 在 txt）

    支持 split: "train" / "val"（按 tar 文件切分）
    返回：(image_tensor, caption_str)
    """

    IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp")
    TXT_EXTS = (".txt",)

    def __init__(self, cfg):
        super().__init__()
        self.root = cfg.root

        # -------- 1) 收集 tar 列表 --------
        tar_glob = getattr(cfg, "tar_glob", "*.tar")
        pattern = tar_glob if os.path.isabs(tar_glob) else os.path.join(self.root, tar_glob)
        all_tar_paths = glob.glob(pattern)
        if not all_tar_paths:
            raise FileNotFoundError(f"No tar shards found: {pattern}")

        # 自然排序（即使名字不是 00000.tar 也不会 2/10 乱序）
        def natural_key(p: str):
            b = os.path.basename(p)
            m = re.search(r"(\d+)", b)
            return int(m.group(1)) if m else b

        all_tar_paths = sorted(all_tar_paths, key=natural_key)

        # -------- 2) 切分 train / val（按 tar 文件切分）--------
        split = getattr(cfg, "split", "train")  # "train" / "val"
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

        # 至少给 val 1 个 shard（避免 val 为空）
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

        # -------- 3) transform（与你原来一致）--------
        size = getattr(cfg, "image_size", 512)
        self.transform = transform.Compose([
            transform.Resize(size),
            transform.CenterCrop(size),
            transform.ToTensor(),
            transform.ImageNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        # -------- 4) 建索引 --------
        # 索引：[(tar_path, img_member, txt_member), ...]
        self.index: List[Tuple[str, str, str]] = []
        self._build_index(natural_key)

        # 训练集是否打乱样本
        shuffle_index = bool(getattr(cfg, "shuffle_index", split == "train"))
        if shuffle_index:
            rng = random.Random(split_seed + 999)
            rng.shuffle(self.index)

        # -------- 5) tar 句柄缓存（单卡、num_workers=0 推荐 True）--------
        self.cache_tar = bool(getattr(cfg, "cache_tar", True))
        self._cached_tar_path: Optional[str] = None
        self._cached_tar: Optional[tarfile.TarFile] = None

        self.set_attrs(total_len=len(self.index))

    def __len__(self):
        return len(self.index)

    def _build_index(self, natural_key):
        """
        扫描 self.tar_paths，把同 base 的 jpg 与 txt 配对：
          000000006.jpg + 000000006.txt -> 一个样本

        并保证 tar 内样本按 key 顺序进入 index（当你不 shuffle 时“顺序可复现”）
        """
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
