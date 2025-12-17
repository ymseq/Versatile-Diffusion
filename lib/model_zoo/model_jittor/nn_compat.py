
import re
import jittor.nn as nn

def _safe_attr_name(k: str) -> str:
    s = re.sub(r'[^0-9a-zA-Z_]', '_', str(k))
    return "m_" + s

class ModuleDict(nn.Module):
    """
    Jittor compatibility for torch.nn.ModuleDict:
    - support __getitem__/items()
    - register submodules via setattr so parameters are tracked
    """
    def __init__(self, modules=None):
        super().__init__()
        self._data = {}
        self._keys = []
        self._alias = {}
        if modules is not None:
            for k, v in modules.items():
                self[k] = v

    def __setitem__(self, key, module):
        if key not in self._data:
            self._keys.append(key)
        self._data[key] = module

        alias = _safe_attr_name(key)
        self._alias[key] = alias
        setattr(self, alias, module)

    def __getitem__(self, key):
        return self._data[key]

    def get(self, key, default=None):
        return self._data.get(key, default)

    def __contains__(self, key):
        return key in self._data

    def __len__(self):
        return len(self._keys)

    def keys(self):
        return list(self._keys)

    def values(self):
        return [self._data[k] for k in self._keys]

    def items(self):
        return [(k, self._data[k]) for k in self._keys]
