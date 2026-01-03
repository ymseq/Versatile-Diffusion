# lib/data_factory/my_dataset.py
import json
import os
from PIL import Image

import jittor as jt
from jittor.dataset.dataset import Dataset
import jittor.transform as transform
from lib.log_service import print_log

class MyVDDataset(Dataset):
    def __init__(self, cfg):
        super().__init__()
        self.root = cfg.root
        self.items = []
        meta_path = os.path.join(self.root, cfg.meta_file)
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                self.items.append(json.loads(line))
        size = getattr(cfg, "image_size", 512)
        self.transform = transform.Compose([
            transform.Resize(size),
            transform.CenterCrop(size),
            transform.ToTensor(),
            transform.ImageNormalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5]),
        ])

        self.set_attrs(total_len=len(self.items))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        img_path = os.path.join(self.root, "images", item["image"])
        img = Image.open(img_path)
        img.load()
        img = img.convert("RGB")
        image = self.transform(img)
        caption = item["caption"]
        return image, caption
