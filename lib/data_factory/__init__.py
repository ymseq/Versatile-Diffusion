from lib.data_factory.my_vd_dataset import MyVDDataset
from lib.data_factory.my_dataset_tar import MyVDDatasetTar
from torch.utils.data.dataloader import default_collate
from torch.utils.data import RandomSampler, SequentialSampler



def get_dataset():
    def _builder(ds_cfg):
        name = getattr(ds_cfg, "name", "my_vd_dataset")
        if name == "my_vd_dataset":
            return MyVDDataset(ds_cfg)
        if name in ("my_vd_dataset_tar", "my_vd_tar"):
            return MyVDDatasetTar(ds_cfg)
        else:
            raise ValueError(f"Unknown dataset name: {name}")

    return _builder


def collate():
    return default_collate


def get_sampler():
    def _builder(dataset, cfg="default_train"):
        if isinstance(cfg, str):
            name = cfg
        else:
            name = getattr(cfg, "name", "default_train")

        if name in ("default_train", "train", "random"):
            return RandomSampler(dataset)

        if name in ("default_eval", "eval", "sequential"):
            return SequentialSampler(dataset)

        return RandomSampler(dataset)

    return _builder


from torch.utils.data import DataLoader


def get_loader():
    def _builder(dataset, cfg=None):
        batch_size = 1
        shuffle = True
        num_workers = 0
        pin_memory = False
        drop_last = False

        if cfg is not None:
            if isinstance(cfg, dict):
                batch_size = cfg.get("batch_size", cfg.get("batch_size_per_gpu", batch_size))
                shuffle = cfg.get("shuffle", shuffle)
                num_workers = cfg.get("num_workers", cfg.get("dataset_num_workers_per_gpu", num_workers))
                pin_memory = cfg.get("pin_memory", pin_memory)
                drop_last = cfg.get("drop_last", drop_last)
            else:
                batch_size = getattr(cfg, "batch_size", getattr(cfg, "batch_size_per_gpu", batch_size))
                shuffle = getattr(cfg, "shuffle", shuffle)
                num_workers = getattr(cfg, "num_workers", getattr(cfg, "dataset_num_workers_per_gpu", num_workers))
                pin_memory = getattr(cfg, "pin_memory", pin_memory)
                drop_last = getattr(cfg, "drop_last", drop_last)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            collate_fn=collate(),
        )

    return _builder


def get_transform():

    def _builder(cfg=None):
        def _identity(x):
            return x
        return _identity

    return _builder


def get_estimator():
    def _builder(cfg=None):
        return None

    return _builder


def get_formatter():
    def _builder(cfg=None):
        return None

    return _builder
