import itertools
import numpy as np

import jittor as jt
import jittor.optim as optim


def singleton(class_):
    instances = {}
    def getinstance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]
    return getinstance


@singleton
class get_optimizer(object):
    def __init__(self):
        self.optimizer = {}
        self.register(optim.SGD,   'sgd')
        self.register(optim.Adam,  'adam')
        if hasattr(optim, "AdamW"):
            self.register(optim.AdamW, 'adamw')

    def register(self, optim_cls, name):
        self.optimizer[name] = optim_cls

    def __call__(self, net, cfg):
        if cfg is None:
            return None

        t = cfg.type
        if t not in self.optimizer:
            raise KeyError(f"Optimizer '{t}' not registered. Available: {list(self.optimizer.keys())}")

        netm = getattr(net, "module", net)

        pg = getattr(netm, "parameter_group", None)

        if pg is not None:
            params = []
            for group_name, module_or_para in pg.items():
                if not isinstance(module_or_para, list):
                    module_or_para = [module_or_para]

                grouped_params = []
                for mi in module_or_para:
                    if hasattr(mi, "parameters") and callable(mi.parameters):
                        grouped_params.extend(list(mi.parameters()))
                    else:
                        grouped_params.append(mi)

                grouped_params = [p for p in grouped_params if p is not None]
                if len(grouped_params) == 0:
                    continue

                params.append({"params": grouped_params, "name": group_name})
        else:
            if hasattr(netm, "parameters") and callable(netm.parameters):
                params = list(netm.parameters())
            else:
                params = []

        opt_cls = self.optimizer[t]

        try:
            return opt_cls(params, **cfg.args)
        except TypeError:
            if isinstance(params, list) and len(params) > 0 and isinstance(params[0], dict):
                flat = []
                for g in params:
                    flat.extend(g.get("params", []))
                params = flat
            return opt_cls(params, **cfg.args)
