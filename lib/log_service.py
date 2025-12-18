# log_service.py (Jittor)
import timeit
import os.path as osp

from .cfg_holder import cfg_unique_holder as cfguh
from . import sync

import jittor as jt

print_console_local_rank0_only = True


def print_log(*console_info):
    local_rank = sync.get_rank('local')
    global_rank = sync.get_rank('global')

    if print_console_local_rank0_only and (local_rank != 0):
        return

    msg = " ".join([str(i) for i in console_info])
    print(msg)

    # only global rank0 writes file to avoid race
    if global_rank != 0:
        return

    log_file = None
    try:
        log_file = cfguh().cfg.train.log_file
    except Exception:
        try:
            log_file = cfguh().cfg.eval.log_file
        except Exception:
            return

    if log_file is not None:
        with open(log_file, 'a') as f:
            f.write(msg + "\n")


def _to_float(x):
    # accept python number / torch scalar / jt.Var / numpy scalar
    try:
        if jt is not None and isinstance(x, jt.Var):
            x.sync()
            return float(x.data[0]) if x.data.size == 1 else float(x.mean().data[0])
    except Exception:
        pass
    try:
        # torch tensor
        return float(x.item())
    except Exception:
        pass
    try:
        return float(x)
    except Exception:
        return float(x)


class distributed_log_manager(object):
    def __init__(self):
        self.sum = {}
        self.cnt = {}
        self.time_check = timeit.default_timer()

        cfgt = cfguh().cfg.train
        use_tensorboard = getattr(cfgt, 'log_tensorboard', False)

        self.ddp = sync.is_ddp()
        self.rank = sync.get_rank('global')
        self.world_size = sync.get_world_size('global')

        self.tb = None
        if use_tensorboard and (self.rank == 0):
            import tensorboardX
            monitoring_dir = osp.join(cfguh().cfg.train.log_dir, 'tensorboard')
            self.tb = tensorboardX.SummaryWriter(osp.join(monitoring_dir))

    def accumulate(self, n, **data):
        if n < 0:
            raise ValueError
        for itemn, di in data.items():
            di = _to_float(di)
            if itemn in self.sum:
                self.sum[itemn] += di * n
                self.cnt[itemn] += n
            else:
                self.sum[itemn] = di * n
                self.cnt[itemn] = n

    def get_mean_value_dict(self):
        keys = sorted(self.sum.keys())
        if len(keys) == 0:
            return {}

        sums = [self.sum[k] for k in keys]
        cnts = [self.cnt[k] for k in keys]

        # global reduce (sum and cnt separately)
        if self.ddp and jt is not None:
            s = jt.array(sums, dtype=jt.float32).mpi_all_reduce("add")
            c = jt.array(cnts, dtype=jt.float32).mpi_all_reduce("add")
            s.sync(); c.sync()
            sums = s.data.tolist()
            cnts = c.data.tolist()

        mean = {}
        for k, s, c in zip(keys, sums, cnts):
            mean[k] = float(s) / max(float(c), 1e-12)
        return mean

    def tensorboard_log(self, step, data, mode='train', **extra):
        if self.tb is None:
            return

        # IMPORTANT: inside rank==0 logging, avoid calling any jittor API.
        # (we only pass python floats here)

        if mode == 'train':
            self.tb.add_scalar('other/epochn', extra.get('epochn', 0), step)
            if 'lr' in extra and extra['lr'] is not None:
                self.tb.add_scalar('other/lr', float(extra['lr']), step)
            for itemn, di in data.items():
                di = float(di)
                if itemn.find('loss') == 0:
                    self.tb.add_scalar('loss/' + itemn, di, step)
                elif itemn == 'Loss':
                    self.tb.add_scalar('Loss', di, step)
                else:
                    self.tb.add_scalar('other/' + itemn, di, step)
        elif mode == 'eval':
            if isinstance(data, dict):
                for itemn, di in data.items():
                    self.tb.add_scalar('eval/' + itemn, float(di), step)
            else:
                self.tb.add_scalar('eval', float(data), step)

    def train_summary(self, itern, epochn, samplen, lr, tbstep=None):
        console_info = [
            f'Iter:{itern}',
            f'Epoch:{epochn}',
            f'Sample:{samplen}',
        ]
        if lr is not None:
            console_info += [f'LR:{lr:.4E}']

        mean = self.get_mean_value_dict()

        tbstep = itern if tbstep is None else tbstep
        if self.rank == 0:
            self.tensorboard_log(
                tbstep, mean, mode='train',
                itern=itern, epochn=epochn, lr=lr)

        loss = mean.pop('Loss', None)
        if loss is not None:
            mean_info = [f'Loss:{loss:.4f}'] + [
                f'{k}:{mean[k]:.4f}'
                for k in sorted(mean.keys())
                if k.find('loss') == 0
            ]
            console_info += mean_info

        console_info.append('Time:{:.2f}s'.format(
            timeit.default_timer() - self.time_check))
        return ' , '.join(console_info)

    def clear(self):
        self.sum = {}
        self.cnt = {}
        self.time_check = timeit.default_timer()

    def tensorboard_close(self):
        if self.tb is not None:
            self.tb.close()
