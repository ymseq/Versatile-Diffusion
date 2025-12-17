
import os
import os.path as osp
import time
import timeit
import numpy as np
import importlib

from .cfg_holder import cfg_unique_holder as cfguh
from .data_factory import (
    get_dataset, collate,
    get_loader,
    get_transform,
    get_estimator,
    get_formatter,
    get_sampler,
)
from .model_zoo import get_model, get_optimizer, get_scheduler
from .log_service import print_log, distributed_log_manager

# ------------------------
# backend detection utils
# ------------------------

def _is_torch_tensor(x):
    try:
        import torch
        return isinstance(x, torch.Tensor)
    except Exception:
        return False

def _is_torch_module(x):
    try:
        import torch.nn as nn
        return isinstance(x, nn.Module)
    except Exception:
        return False

def _is_jittor_var(x):
    try:
        import jittor as jt
        return isinstance(x, jt.Var)
    except Exception:
        return False

def _is_jittor_module(x):
    # Jittor Module class
    try:
        import jittor.nn as jnn
        return isinstance(x, jnn.Module)
    except Exception:
        return False

def to_numpy(data):
    """
    Torch Tensor / Jittor Var / nested(list/tuple/dict) -> numpy
    """
    if _is_torch_tensor(data):
        import torch
        return data.detach().cpu().numpy()
    if _is_jittor_var(data):
        return data.numpy()

    if isinstance(data, (list, tuple)):
        return [to_numpy(x) for x in data]
    if isinstance(data, dict):
        return {k: to_numpy(v) for k, v in data.items()}
    return data

def set_global_seed(seed: int):
    if seed is None:
        return
    np.random.seed(seed)
    try:
        import random
        random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    try:
        import jittor as jt
        jt.set_global_seed(seed)
    except Exception:
        pass

def move_to_device(net, device_id: int = 0):
    # Torch
    if _is_torch_module(net):
        import torch
        if torch.cuda.is_available():
            return net.cuda(device_id)
        return net

    # Jittor
    if _is_jittor_module(net):
        try:
            import jittor as jt
            jt.flags.use_cuda = 1
        except Exception:
            pass
        return net

    return net

def save_state_dict(net, path: str):
    try:
        import torch
        if isinstance(net, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
            net = net.module
    except Exception:
        pass

    # torch
    if _is_torch_module(net):
        import torch
        torch.save(net.state_dict(), path)
        return

    # jittor
    if _is_jittor_module(net):
        import jittor as jt
        sd = net.state_dict()
        jt.save(sd, path)
        return

    raise TypeError(f"Unknown model type for saving: {type(net)}")


# ------------------------
# stages
# ------------------------

class train_stage(object):

    def __init__(self):
        self.nested_eval_stage = None
        self.rv_keep = None

    def is_better(self, x):
        return (self.rv_keep is None) or (x > self.rv_keep)

    def set_model(self, net, mode):
        # Torch/Jittor 都有 train()/eval() 习惯接口（jittor.nn.Module 也有）
        if mode == 'train':
            return net.train()
        elif mode == 'eval':
            return net.eval()
        raise ValueError

    def __call__(self, **paras):
        cfg = cfguh().cfg
        cfgt = cfg.train
        logm = distributed_log_manager()  # 单卡下内部不会 all_reduce
        epochn, itern, samplen = 0, 0, 0

        step_type = cfgt.get('step_type', 'iter')
        assert step_type in ['epoch', 'iter', 'sample']

        step_num      = cfgt.get('step_num', None)
        gradacc_every = cfgt.get('gradacc_every', 1)
        log_every     = cfgt.get('log_every', None)
        ckpt_every    = cfgt.get('ckpt_every', None)
        eval_start    = cfgt.get('eval_start', 0)
        eval_every    = cfgt.get('eval_every', None)

        # resume_step（你原逻辑保留）
        if paras.get('resume_step', None) is not None:
            resume_step = paras['resume_step']
            assert step_type == resume_step['type']
            epochn = resume_step['epochn']
            itern = resume_step['itern']
            samplen = resume_step['samplen']
            del paras['resume_step']

        trainloader = paras['trainloader']
        optimizer   = paras.get('optimizer', None)
        scheduler   = paras.get('scheduler', None)
        net         = paras['net']

        weight_path = osp.join(cfgt.log_dir, 'weight')
        if (not osp.isdir(weight_path)):
            os.makedirs(weight_path, exist_ok=True)
        if cfgt.get('save_init_model', False):
            self.save(net, is_init=True, step=0, optimizer=optimizer)

        epoch_time = timeit.default_timer()
        end_flag = False
        net.train()

        while True:
            if step_type == 'epoch':
                lr = scheduler[epochn] if scheduler is not None else None

            for batch in trainloader:
                # batch size 推断（保持你原逻辑）
                if not isinstance(batch[0], list):
                    bs = batch[0].shape[0]
                else:
                    bs = len(batch[0])

                if cfgt.get('skip_partial_batch', False) and (bs != cfgt.batch_size_per_gpu):
                    continue

                itern_next = itern + 1
                samplen_next = samplen + bs  # 单卡不乘 world size

                if step_type == 'iter':
                    lr = scheduler[itern // gradacc_every] if scheduler is not None else None
                    grad_update = (itern % gradacc_every) == (gradacc_every - 1)
                elif step_type == 'sample':
                    lr = scheduler[samplen] if scheduler is not None else None
                    grad_update = True
                else:
                    grad_update = True

                paras_new = self.main(
                    batch=batch,
                    lr=lr,
                    itern=itern,
                    epochn=epochn,
                    samplen=samplen,
                    isinit=False,
                    grad_update=grad_update,
                    **paras
                )

                if paras_new is None:
                    paras_new = {}
                paras.update(paras_new)

                # 记录 log_info
                if 'log_info' in paras:
                    logm.accumulate(bs, **paras['log_info'])

                # log
                display_flag = False
                if log_every is not None:
                    display_i = (itern // log_every) != (itern_next // log_every)
                    display_s = (samplen // log_every) != (samplen_next // log_every)
                    display_flag = (display_i and (step_type == 'iter')) or (display_s and (step_type == 'sample'))

                if display_flag:
                    tbstep = itern_next if step_type == 'iter' else samplen_next
                    console_info = logm.train_summary(itern_next, epochn, samplen_next, lr, tbstep=tbstep)
                    logm.clear()
                    print_log(console_info)

                # eval（单卡：直接执行）
                eval_flag = False
                if (self.nested_eval_stage is not None) and (eval_every is not None):
                    if step_type == 'iter':
                        eval_flag = (itern // eval_every) != (itern_next // eval_every)
                        eval_flag = eval_flag and (itern_next >= eval_start)
                        eval_flag = eval_flag or (itern == 0)
                    elif step_type == 'sample':
                        eval_flag = (samplen // eval_every) != (samplen_next // eval_every)
                        eval_flag = eval_flag and (samplen_next >= eval_start)
                        eval_flag = eval_flag or (samplen == 0)

                if eval_flag:
                    eval_cnt = itern_next if step_type == 'iter' else samplen_next
                    net = self.set_model(net, 'eval')
                    rv = self.nested_eval_stage(eval_cnt=eval_cnt, **paras).get('eval_rv', None)
                    if rv is not None:
                        logm.tensorboard_log(eval_cnt, rv, mode='eval')
                    if self.is_better(rv):
                        self.rv_keep = rv
                        step = {'epochn': epochn, 'itern': itern_next, 'samplen': samplen_next, 'type': step_type}
                        self.save(net, is_best=True, step=step, optimizer=optimizer)
                    net = self.set_model(net, 'train')

                # ckpt
                ckpt_flag = False
                if ckpt_every is not None:
                    ckpt_i = (itern // ckpt_every) != (itern_next // ckpt_every)
                    ckpt_s = (samplen // ckpt_every) != (samplen_next // ckpt_every)
                    ckpt_flag = (ckpt_i and (step_type == 'iter')) or (ckpt_s and (step_type == 'sample'))

                if ckpt_flag:
                    step = {'epochn': epochn, 'itern': itern_next, 'samplen': samplen_next, 'type': step_type}
                    if step_type == 'iter':
                        print_log(f'Checkpoint... {itern_next}')
                        self.save(net, itern=itern_next, step=step, optimizer=optimizer)
                    else:
                        print_log(f'Checkpoint... {samplen_next}')
                        self.save(net, samplen=samplen_next, step=step, optimizer=optimizer)

                # end
                itern = itern_next
                samplen = samplen_next

                if step_num is not None:
                    end_flag = (itern >= step_num and step_type == 'iter') or (samplen >= step_num and step_type == 'sample')
                if end_flag:
                    break

            epochn += 1
            print_log(f'Epoch {epochn} time:{timeit.default_timer()-epoch_time:.2f}s.')
            epoch_time = timeit.default_timer()

            if end_flag:
                break
            elif step_type != 'epoch':
                trainloader = self.trick_update_trainloader(trainloader)
                continue

            # epoch-step logging/eval/ckpt（保持你原结构）
            display_flag = False
            if (log_every is not None) and (step_type == 'epoch'):
                display_flag = (epochn == 1) or (epochn % log_every == 0)
            if display_flag:
                console_info = logm.train_summary(itern, epochn, samplen, lr, tbstep=epochn)
                logm.clear()
                print_log(console_info)

            eval_flag = False
            if (self.nested_eval_stage is not None) and (eval_every is not None) and (step_type == 'epoch'):
                eval_flag = (epochn % eval_every == 0) and (itern >= eval_start)
                eval_flag = (epochn == 1) or eval_flag

            if eval_flag:
                net = self.set_model(net, 'eval')
                rv = self.nested_eval_stage(eval_cnt=epochn, **paras).get('eval_rv', None)
                if rv is not None:
                    logm.tensorboard_log(epochn, rv, mode='eval')
                if self.is_better(rv):
                    self.rv_keep = rv
                    step = {'epochn': epochn, 'itern': itern, 'samplen': samplen, 'type': step_type}
                    self.save(net, is_best=True, step=step, optimizer=optimizer)
                net = self.set_model(net, 'train')

            ckpt_flag = False
            if (ckpt_every is not None) and (step_type == 'epoch'):
                ckpt_flag = (epochn % ckpt_every) == 0
            if ckpt_flag:
                step = {'epochn': epochn, 'itern': itern, 'samplen': samplen, 'type': step_type}
                print_log(f'Checkpoint... {epochn}')
                self.save(net, epochn=epochn, step=step, optimizer=optimizer)

            if (step_type == 'epoch') and (step_num is not None) and (epochn >= step_num):
                break

            trainloader = self.trick_update_trainloader(trainloader)

        logm.tensorboard_close()
        return {}

    def main(self, **paras):
        raise NotImplementedError

    def trick_update_trainloader(self, trainloader):
        return trainloader

    def save_model(self, net, path_noext, **paras):
        path = path_noext + '.pth'
        save_state_dict(net, path)
        print_log(f'Saving model file {path}')

    def save(self, net, itern=None, epochn=None, samplen=None,
             is_init=False, is_best=False, is_last=False, **paras):
        exid = cfguh().cfg.env.experiment_id
        cfgt = cfguh().cfg.train
        cfgm = cfguh().cfg.model
        net_symbol = cfgm.symbol

        check = sum([itern is not None, samplen is not None, epochn is not None, is_init, is_best, is_last])
        assert check < 2

        if itern is not None:
            path_noexp = f'{exid}_{net_symbol}_iter_{itern}'
        elif samplen is not None:
            path_noexp = f'{exid}_{net_symbol}_samplen_{samplen}'
        elif epochn is not None:
            path_noexp = f'{exid}_{net_symbol}_epoch_{epochn}'
        elif is_init:
            path_noexp = f'{exid}_{net_symbol}_init'
        elif is_best:
            path_noexp = f'{exid}_{net_symbol}_best'
        elif is_last:
            path_noexp = f'{exid}_{net_symbol}_last'
        else:
            path_noexp = f'{exid}_{net_symbol}_default'

        path_noexp = osp.join(cfgt.log_dir, 'weight', path_noexp)
        self.save_model(net, path_noexp, **paras)


class eval_stage(object):
    def __init__(self):
        self.evaluator = None

    def create_dir(self, path):
        if not osp.isdir(path):
            os.makedirs(path, exist_ok=True)

    def __call__(self, evalloader, net, **paras):
        cfgv = cfguh().cfg.eval
        if self.evaluator is None:
            from .evaluator import get_evaluator
            self.evaluator = get_evaluator()(cfgv.evaluator)

        evaluator = self.evaluator
        time_check = timeit.default_timer()

        for idx, batch in enumerate(evalloader):
            rv = self.main(batch, net)
            evaluator.add_batch(**rv)

            if cfgv.get('output_result', False):
                try:
                    self.output_f(**rv, cnt=paras['eval_cnt'])
                except Exception:
                    self.output_f(**rv)

            if idx % cfgv.log_display == cfgv.log_display - 1:
                print_log(f'processed.. {idx+1}, Time:{timeit.default_timer() - time_check:.2f}s')
                time_check = timeit.default_timer()

        evaluator.set_sample_n(len(evalloader.dataset))
        eval_rv = evaluator.compute()
        evaluator.one_line_summary()
        evaluator.save(cfgv.log_dir)
        evaluator.clear_data()
        return {'eval_rv': eval_rv}

    def main(self, batch, net):
        raise NotImplementedError


# ------------------------
# execution containers
# ------------------------

class exec_container(object):
    """
    单卡版执行容器：
    - 不 init_process_group
    - 不 spawn 多进程
    - local_rank 固定 0
    """
    def __init__(self, cfg, **kwargs):
        self.cfg = cfg
        self.registered_stages = []

    def register_stage(self, stage):
        self.registered_stages.append(stage)

    def __call__(self, **kwargs):
        cfg = self.cfg
        cfguh().save_cfg(cfg)

        if isinstance(cfg.env.get('rnd_seed', None), int):
            set_global_seed(cfg.env.rnd_seed)

        time_start = timeit.default_timer()

        para = {'itern_total': 0}
        para.update(self.prepare_dataloader() or {})
        para.update(self.prepare_model() or {})

        for stage in self.registered_stages:
            stage_para = stage(**para)
            if stage_para:
                para.update(stage_para)

        # 可选：保存 last
        self.save_last_model(**para)

        print_log(f'Total {timeit.default_timer() - time_start:.2f} seconds')
        return para

    def prepare_dataloader(self):
        return {'trainloader': None, 'evalloader': None}

    def prepare_model(self):
        return {'net': None}

    def save_last_model(self, **para):
        return


class train(exec_container):
    def prepare_dataloader(self):
        cfg = cfguh().cfg
        
        import torch

        trainset = get_dataset()(cfg.train.dataset)
        sampler = get_sampler()(dataset=trainset, cfg=cfg.train.dataset.get('sampler', 'default_train'))
        trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=cfg.train.batch_size_per_gpu,
            sampler=sampler,
            num_workers=cfg.train.dataset_num_workers_per_gpu,
            drop_last=False,
            pin_memory=cfg.train.dataset.get('pin_memory', False),
            collate_fn=collate(),
        )

        evalloader = None
        if 'eval' in cfg:
            evalset = get_dataset()(cfg.eval.dataset)
            if evalset is not None:
                sampler = get_sampler()(dataset=evalset, cfg=cfg.eval.dataset.get('sampler', 'default_eval'))
                evalloader = torch.utils.data.DataLoader(
                    evalset,
                    batch_size=cfg.eval.batch_size_per_gpu,
                    sampler=sampler,
                    num_workers=cfg.eval.dataset_num_workers_per_gpu,
                    drop_last=False,
                    pin_memory=cfg.eval.dataset.get('pin_memory', False),
                    collate_fn=collate(),
                )

        return {'trainloader': trainloader, 'evalloader': evalloader}

    def prepare_model(self):
        cfg = cfguh().cfg
        net = get_model()(cfg.model)
        net = move_to_device(net, device_id=0)
        net.train()

        scheduler = get_scheduler()(cfg.train.scheduler) if 'scheduler' in cfg.train else None
        optimizer = get_optimizer()(net, cfg.train.optimizer) if 'optimizer' in cfg.train else None

        return {'net': net, 'optimizer': optimizer, 'scheduler': scheduler}

    def save_last_model(self, **para):
        cfgt = cfguh().cfg.train
        net = para['net']
        net_symbol = cfguh().cfg.model.symbol
        exid = cfguh().cfg.env.experiment_id
        path = osp.join(cfgt.log_dir, f'{exid}_{net_symbol}_last.pth')
        save_state_dict(net, path)
        print_log(f'Saving model file {path}')


class eval(exec_container):
    def prepare_dataloader(self):
        cfg = cfguh().cfg
        import torch

        evalloader = None
        if cfg.eval.get('dataset', None) is not None:
            evalset = get_dataset()(cfg.eval.dataset)
            if evalset is None:
                return {'trainloader': None, 'evalloader': None}

            sampler = get_sampler()(dataset=evalset, cfg=getattr(cfg.eval.dataset, 'sampler', 'default_eval'))
            evalloader = torch.utils.data.DataLoader(
                evalset,
                batch_size=cfg.eval.batch_size_per_gpu,
                sampler=sampler,
                num_workers=cfg.eval.dataset_num_workers_per_gpu,
                drop_last=False,
                pin_memory=False,
                collate_fn=collate(),
            )
        return {'trainloader': None, 'evalloader': evalloader}

    def prepare_model(self):
        cfg = cfguh().cfg
        net = get_model()(cfg.model)
        net = move_to_device(net, device_id=0)
        net.eval()
        return {'net': net}

    def save_last_model(self, **para):
        return


# ------------------------
# dynamic import helper
# ------------------------

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)
