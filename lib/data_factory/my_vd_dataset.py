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
        # print_log(f"Loading dataset from {self.root}...")
        meta_path = os.path.join(self.root, cfg.meta_file)
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                self.items.append(json.loads(line))
        # print_log(f"Loaded {len(self.items)} items.")
        size = getattr(cfg, "image_size", 512)
        self.transform = transform.Compose([
            transform.Resize(size),
            transform.CenterCrop(size),
            transform.ToTensor(),
            transform.ImageNormalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5]),
        ])

        # jittor Dataset 需要设置 total_len
        self.set_attrs(total_len=len(self.items))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        #print_log(f"Fetching item {idx}...")
        item = self.items[idx]
        img_path = os.path.join(self.root, "images", item["image"])
        img = Image.open(img_path)
        # print_log(f"Opened image {img_path}, mode: {img.mode}")
        img.load()
        # print_log(f"Image {img_path} loaded.")
        img = img.convert("RGB")
        # print_log(f"Converted image {img_path} to RGB.")
        image = self.transform(img)
        # print_log(f"Transformed image {img_path}, shape: {image.shape}")
        caption = item["caption"]
        # print_log(f"Loaded item {idx}: image shape {image.shape}, caption length {len(caption)}")
        return image, caption
