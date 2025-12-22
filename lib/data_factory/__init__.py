from lib.data_factory.my_vd_dataset import MyVDDataset
from lib.data_factory.my_dataset_tar import MyVDDatasetTar
from torch.utils.data.dataloader import default_collate
from torch.utils.data import RandomSampler, SequentialSampler



def get_dataset():
    """
    返回一个构造函数，给 utils.train.prepare_dataloader 用：
        trainset = get_dataset()(cfg.train.dataset)
    """
    def _builder(ds_cfg):
        # 如果以后有别的数据集，可以在这里根据 ds_cfg.name 分支
        name = getattr(ds_cfg, "name", "my_vd_dataset")
        if name == "my_vd_dataset":
            return MyVDDataset(ds_cfg)
        if name in ("my_vd_dataset_tar", "my_vd_tar"):
            return MyVDDatasetTar(ds_cfg)
        else:
            raise ValueError(f"Unknown dataset name: {name}")

        # if name != "my_vd_dataset":
        #     raise ValueError(f"Unknown dataset name: {name}")
        # return MyVDDataset(ds_cfg)

    return _builder


def collate():
    """
    返回给 DataLoader 用的 collate_fn：
        collate_fn = collate()
    """
    return default_collate


def get_sampler():
    """
    返回一个构造 sampler 的函数，给 utils.train.prepare_dataloader 用：
        sampler = get_sampler()(dataset=trainset,
                                cfg=cfg.train.dataset.get('sampler', 'default_train'))
    """
    def _builder(dataset, cfg="default_train"):
        # cfg 这里通常是一个字符串，比如 "default_train" / "default_eval"
        if isinstance(cfg, str):
            name = cfg
        else:
            # 如果你以后传的是 dict / cfg 对象，可以在这里再加逻辑
            name = getattr(cfg, "name", "default_train")

        # 训练用随机采样
        if name in ("default_train", "train", "random"):
            return RandomSampler(dataset)

        # 验证/测试用顺序采样
        if name in ("default_eval", "eval", "sequential"):
            return SequentialSampler(dataset)

        # 默认退回随机采样
        return RandomSampler(dataset)

    return _builder


from torch.utils.data import DataLoader


def get_loader():
    """
    Loader 工厂。

    目前你的训练主流程在 utils.train.prepare_dataloader 里已经直接用
    DataLoader 了，一般不会真的用到 get_loader。这里给一个通用实现，
    主要是为了避免 ImportError，有别的脚本用的时候也能正常工作。

    用法示例：
        loader = get_loader()(dataset=trainset, cfg=cfg.train)
    """
    def _builder(dataset, cfg=None):
        # 一些合理的默认值
        batch_size = 1
        shuffle = True
        num_workers = 0
        pin_memory = False
        drop_last = False

        if cfg is not None:
            # 兼容 dict / yacs Node / EasyDict 等
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
    """
    Transform 工厂。

    原始项目里可能会根据配置构造 torchvision.transforms，
    你目前的微调流程其实没用到它，所以先给一个“恒等变换”的占位实现：

        transform = get_transform()(cfg.train.get("transform", None))
        img = transform(img)  # 现在就是原样返回
    """
    def _builder(cfg=None):
        def _identity(x):
            return x
        return _identity

    return _builder


def get_estimator():
    """
    Estimator 工厂（例如均值/方差估计、统计信息等）。

    目前你的训练 stage 没有用到 estimator，这里返回 None 作为占位，
    只为满足 import，不会影响现在的训练流程。
    """
    def _builder(cfg=None):
        return None

    return _builder


def get_formatter():
    """
    Formatter 工厂（例如把模型输出格式化为可视化结果）。

    同样，目前没用到，先返回 None 占位。如果之后需要，可以在这里根据 cfg
    实现真正的 formatter。
    """
    def _builder(cfg=None):
        return None

    return _builder
