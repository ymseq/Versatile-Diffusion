# sync.py (Jittor)
import os
import time
import random
import pickle
from typing import Any, Optional, Tuple

import jittor as jt


def singleton(class_):
    instances = {}
    def getinstance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]
    return getinstance


def _env_int(keys, default: Optional[int] = None) -> Optional[int]:
    for k in keys:
        v = os.environ.get(k, None)
        if v is not None:
            try:
                return int(v)
            except Exception:
                pass
    return default


def is_ddp() -> bool:
    # keep old API name; in Jittor this means "in mpi"
    return (jt is not None) and bool(getattr(jt, "in_mpi", False)) and int(getattr(jt, "world_size", 1)) > 1


def _global_rank() -> int:
    if jt is not None:
        return int(getattr(jt, "rank", 0))
    return _env_int(["OMPI_COMM_WORLD_RANK", "PMI_RANK", "RANK"], 0) or 0


def _global_world_size() -> int:
    if jt is not None:
        return int(getattr(jt, "world_size", 1))
    return _env_int(["OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "WORLD_SIZE"], 1) or 1


def _local_rank() -> int:
    return _env_int(
        ["OMPI_COMM_WORLD_LOCAL_RANK", "MPI_LOCALRANKID", "LOCAL_RANK", "SLURM_LOCALID"],
        None
    ) or (_global_rank() % max(_local_world_size(), 1))


def _local_world_size() -> int:
    # prefer MPI-provided local size; fallback to visible CUDA count if you want
    return _env_int(
        ["OMPI_COMM_WORLD_LOCAL_SIZE", "MPI_LOCALNRANKS", "LOCAL_WORLD_SIZE", "SLURM_NTASKS_PER_NODE"],
        None
    ) or int(os.environ.get("CUDA_VISIBLE_DEVICES", "").count(",") + 1 if os.environ.get("CUDA_VISIBLE_DEVICES") else 1)


def _node_rank() -> int:
    # node rank (which node am I on)
    nr = _env_int(["OMPI_COMM_WORLD_NODE_RANK", "SLURM_NODEID"], None)
    if nr is not None:
        return nr
    lws = max(_local_world_size(), 1)
    return _global_rank() // lws


def get_rank(type: str = "local"):
    gr = _global_rank()
    lr = _local_rank()
    nr = _node_rank()
    if type == "global":
        return gr
    elif type == "local":
        return lr
    elif type == "node":
        return nr
    elif type == "all":
        return gr, lr, nr
    else:
        raise ValueError("Unknown type")


def get_world_size(type: str = "local"):
    gws = _global_world_size()
    lws = _local_world_size()
    nodes = max(gws // max(lws, 1), 1)
    if type == "global":
        return gws
    elif type == "local":
        return lws
    elif type == "node":
        return nodes
    elif type == "all":
        return gws, lws, nodes
    else:
        raise ValueError("Unknown type")


def _mpi_barrier():
    if not is_ddp():
        return
    # emulate a barrier via all_reduce; everyone must call it
    x = jt.array([1], dtype=jt.int32)
    _ = x.mpi_all_reduce("add")
    _.sync()


def _mpi_broadcast_bytes(buf: bytes, root: int) -> bytes:
    """
    Broadcast arbitrary bytes from root to all ranks, using jt.mpi_broadcast.
    Everyone must call this function.
    """
    if not is_ddp():
        return buf

    # 1) broadcast length
    if _global_rank() == root:
        n = len(buf)
    else:
        n = 0
    n_var = jt.array([n], dtype=jt.int32)
    n_var = jt.mpi.mpi_broadcast(n_var, root=root)
    n = int(n_var.data[0])

    # 2) broadcast payload
    if _global_rank() == root:
        arr = jt.array(list(buf), dtype=jt.uint8)
    else:
        arr = jt.zeros([n], dtype=jt.uint8)
    arr = jt.mpi.mpi_broadcast(arr, root=root)
    out = bytes(arr.data.tolist())
    return out


class nodewise_sync_global(object):
    """
    In torch-ddp-spawn version this carried shared-memory barrier resources.
    In MPI mode we don't need a separate "global" object; keep it for compatibility.
    """
    def __init__(self):
        pass

    def destroy(self):
        return


@singleton
class nodewise_sync(object):
    """
    Keep original interface: barrier() and broadcast_r0().
    In MPI we implement them with MPI collectives (global scope).
    """
    def __init__(self):
        self.local_rank = None
        self.global_rank = None
        self.node_rank = None
        self.global_world_size = None
        self.local_world_size = None
        self.nodes = None

    def copy_global(self, reference):
        # compatibility no-op
        return self

    def local_init(self):
        self.global_rank, self.local_rank, self.node_rank = get_rank("all")
        self.global_world_size, self.local_world_size, self.nodes = get_world_size("all")
        return self

    def random_sync_id(self) -> int:
        # global broadcast a random int (root=0)
        rid = int(random.random() * 10000) + int(time.time()) * 10000
        if is_ddp():
            v = jt.array([rid if self.global_rank == 0 else 0], dtype=jt.int32)
            v = jt.mpi.mpi_broadcast(v, root=0)
            rid = int(v.data[0])
        return rid

    def barrier(self):
        _mpi_barrier()

    def broadcast_r0(self, data: Any = None):
        """
        Broadcast python object from *global rank 0* to everyone.
        Everyone must call it.
        - rank0 passes data
        - others pass None
        """
        if not is_ddp():
            return data

        if self.global_rank == 0:
            payload = pickle.dumps(data)
        else:
            payload = b""

        payload = _mpi_broadcast_bytes(payload, root=0)
        if self.global_rank == 0:
            return None
        return pickle.loads(payload)

    def destroy(self):
        return
