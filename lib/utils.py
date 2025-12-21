# exec_and_stage.py (Jittor MPI version; keep original class names)
import os
import os.path as osp
import timeit
import numpy as np

import torch  # keep torch for frozen VAE/CLIP etc.

import jittor as jt
import importlib

from .cfg_holder import cfg_unique_holder as cfguh
from .data_factory import get_dataset, collate, get_sampler
from .model_zoo import get_model, get_optimizer, get_scheduler
from .log_service import print_log, distributed_log_manager
from .evaluator.evaluator import get_evaluator
from . import sync


def _is_jittor_module(net) -> bool:
    return (jt is not None) and hasattr(jt, "Module") and isinstance(net, jt.Module)


def _is_torch_module(net) -> bool:
    import torch.nn as nn
    return isinstance(net, nn.Module)


def _jt_safe_save(obj, path: str):
    """
    Everyone calls this function; it writes only on rank0 but does not cause mismatch,
    because it's pure python branching + jt.safepickle inside both ranks? -> we avoid that:
    - use jt.single_process_scope if available
    """
    if jt is None:
        raise RuntimeError("Jittor not available")

    # best effort: single_process_scope exists in jittor mpi docs
    if hasattr(jt, "single_process_scope"):
        @jt.single_process_scope()
        def _do():
            # NOTE: inside this scope, MPI is disabled, so safe to call jt APIs
            jt.safepickle(obj, path)
        _do()
    else:
        # fallback: all ranks save to different files (only for emergency)
        gr = sync.get_rank("global")
        jt.safepickle(obj, path + f".rank{gr}")


class train_stage(object):
    def __init__(self):
        self.nested_eval_stage = None
        self.rv_keep = None

    def is_better(self, x):
        return (self.rv_keep is None) or (x > self.rv_keep)

    def set_model(self, net, mode):
        if hasattr(net, mode):
            return getattr(net, mode)()
        # torch style
        if mode == 'train':
            return net.train()
        elif mode == 'eval':
            return net.eval()
        else:
            raise ValueError

    def __call__(self, **paras):
        cfg = cfguh().cfg
        cfgt = cfg.train
        logm = distributed_log_manager()

        epochn, itern, samplen = 0, 0, 0

        step_type = cfgt.get('step_type', 'iter')
        assert step_type in ['epoch', 'iter', 'sample']
        step_num      = cfgt.get('step_num', None)
        gradacc_every = cfgt.get('gradacc_every', 1)
        log_every     = cfgt.get('log_every', None)
        ckpt_every    = cfgt.get('ckpt_every', None)
        eval_start    = cfgt.get('eval_start', 0)
        eval_every    = cfgt.get('eval_every', None)

        if paras.get('resume_step', None) is not None:
            resume_step = paras['resume_step']
            assert step_type == resume_step['type']
            epochn = resume_step['epochn']
            itern = resume_step['itern']
            samplen = resume_step['samplen']
            del paras['resume_step']

        trainloader = paras['trainloader']
        optimizer   = paras['optimizer']
        scheduler   = paras['scheduler']
        net         = paras['net']

        GRANK, LRANK, NRANK = sync.get_rank('all')
        GWSIZE, LWSIZE, NODES = sync.get_world_size('all')

        weight_path = osp.join(cfgt.log_dir, 'weight')
        if (GRANK == 0) and (not osp.isdir(weight_path)):
            os.makedirs(weight_path)
        # sync before any save so all ranks see the directory
        sync.nodewise_sync().barrier()

        # init save (all ranks enter; actual save handled safely inside)
        if cfgt.save_init_model:
            self.save(net, is_init=True, step=0, optimizer=optimizer)

        epoch_time = timeit.default_timer()
        end_flag = False
        net = self.set_model(net, 'train')
        net.to('cuda')

        while True:
            if step_type == 'epoch':
                lr = scheduler[epochn] if scheduler is not None else None
            for batch in trainloader:
                b0 = batch[0]
                if isinstance(b0, list):
                    bs = len(b0)
                else:
                    bs = int(getattr(b0, "shape", [len(b0)])[0])
                if cfgt.skip_partial_batch and (bs != cfgt.batch_size_per_gpu):
                    continue
                itern_next = itern + 1
                samplen_next = samplen + bs * GWSIZE

                grad_update = True
                if step_type == 'iter':
                    lr = scheduler[itern // gradacc_every] if scheduler is not None else None
                    grad_update = (itern % gradacc_every) == (gradacc_every - 1)
                elif step_type == 'sample':
                    lr = scheduler[samplen] if scheduler is not None else None
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

                if paras_new is not None:
                    paras.update(paras_new)

                logm.accumulate(bs, **paras.get('log_info', {}))

                # -------- log (all ranks compute reduce; only local rank0 prints) --------
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

                # -------- eval (IMPORTANT: all ranks must enter together) --------
                eval_flag = False
                if (self.nested_eval_stage is not None) and (eval_every is not None):
                    if step_type == 'iter':
                        eval_flag = ((itern // eval_every) != (itern_next // eval_every))
                        eval_flag = eval_flag and (itern_next >= eval_start)
                        eval_flag = eval_flag or (itern == 0)
                    elif step_type == 'sample':
                        eval_flag = ((samplen // eval_every) != (samplen_next // eval_every))
                        eval_flag = eval_flag and (samplen_next >= eval_start)
                        eval_flag = eval_flag or (samplen == 0)

                if eval_flag:
                    eval_cnt = itern_next if step_type == 'iter' else samplen_next
                    net = self.set_model(net, 'eval')
                    rv = self.nested_eval_stage(eval_cnt=eval_cnt, **paras)
                    rv = rv.get('eval_rv', None) if isinstance(rv, dict) else rv

                    # rank0 tensorboard/best
                    if (GRANK == 0) and (rv is not None):
                        logm.tensorboard_log(eval_cnt, rv, mode='eval')

                    if (GRANK == 0) and (rv is not None) and self.is_better(rv):
                        self.rv_keep = rv
                        step = {'epochn': epochn, 'itern': itern_next,
                                'samplen': samplen_next, 'type': step_type}
                        self.save(net, is_best=True, step=step, optimizer=optimizer)

                    net = self.set_model(net, 'train')

                # -------- ckpt (all ranks enter; actual write guarded safely) --------
                ckpt_flag = False
                if ckpt_every is not None:
                    ckpt_i = (itern // ckpt_every) != (itern_next // ckpt_every)
                    ckpt_s = (samplen // ckpt_every) != (samplen_next // ckpt_every)
                    ckpt_flag = (ckpt_i and (step_type == 'iter')) or (ckpt_s and (step_type == 'sample'))

                if ckpt_flag:
                    step = {'epochn': epochn, 'itern': itern_next,
                            'samplen': samplen_next, 'type': step_type}
                    if GRANK == 0:
                        print_log(f'Checkpoint... {itern_next if step_type=="iter" else samplen_next}')
                    if step_type == 'iter':
                        self.save(net, itern=itern_next, step=step, optimizer=optimizer)
                    else:
                        self.save(net, samplen=samplen_next, step=step, optimizer=optimizer)

                # -------- end --------
                itern = itern_next
                samplen = samplen_next

                if step_type is not None:
                    end_flag = (itern >= step_num and (step_type == 'iter')) or \
                               (samplen >= step_num and (step_type == 'sample'))
                if end_flag:
                    break

            epochn += 1
            print_log('Epoch {} time:{:.2f}s.'.format(epochn, timeit.default_timer() - epoch_time))
            epoch_time = timeit.default_timer()

            if end_flag:
                break

        logm.tensorboard_close()
        return {}

    def main(self, **paras):
        raise NotImplementedError

    def save_model(self, net, path_noext, **paras):
        path = path_noext + '.pth'

        # torch module: rank0 only is fine (no jittor api)
        if _is_torch_module(net):
            if sync.get_rank("global") == 0:
                netm = net.module if hasattr(net, "module") else net
                torch.save(netm.state_dict(), path)
                print_log(f"Saving model file {path}")
            sync.nodewise_sync().barrier()
            return

        # jittor module: everyone calls; actual writing handled safely
        if _is_jittor_module(net):
            # prefer state_dict
            obj = net.state_dict() if hasattr(net, "state_dict") else net
            _jt_safe_save(obj, path)
            if sync.get_rank("global") == 0:
                print_log(f"Saving model file {path}")
            sync.nodewise_sync().barrier()
            return

        # unknown type: rank0 pickle
        if sync.get_rank("global") == 0:
            import pickle
            with open(path + ".pkl", "wb") as f:
                pickle.dump(net, f)

    def save(self, net, itern=None, epochn=None, samplen=None,
             is_init=False, is_best=False, is_last=False, **paras):
        exid = cfguh().cfg.env.experiment_id
        cfgt = cfguh().cfg.train
        cfgm = cfguh().cfg.model
        net_symbol = cfgm.symbol

        check = sum([itern is not None, samplen is not None, epochn is not None,
                     is_init, is_best, is_last])
        assert check < 2

        if itern is not None:
            name = f'{exid}_{net_symbol}_iter_{itern}'
        elif samplen is not None:
            name = f'{exid}_{net_symbol}_samplen_{samplen}'
        elif epochn is not None:
            name = f'{exid}_{net_symbol}_epoch_{epochn}'
        elif is_init:
            name = f'{exid}_{net_symbol}_init'
        elif is_best:
            name = f'{exid}_{net_symbol}_best'
        elif is_last:
            name = f'{exid}_{net_symbol}_last'
        else:
            name = f'{exid}_{net_symbol}_default'

        path_noext = osp.join(cfgt.log_dir, 'weight', name)
        self.save_model(net, path_noext, **paras)

class eval_stage(object):
    def __init__(self):
        self.evaluator = None

    def create_dir(self, path):
        if (sync.get_rank('global') == 0) and (not osp.isdir(path)):
            os.makedirs(path)
        sync.nodewise_sync().barrier()

    def __call__(self, evalloader, net, **paras):
        cfgt = cfguh().cfg.eval

        if self.evaluator is None:
            evaluator = get_evaluator()(cfgt.evaluator)
            self.evaluator = evaluator
        else:
            evaluator = self.evaluator

        time_check = timeit.default_timer()

        for idx, batch in enumerate(evalloader):
            rv = self.main(batch, net)
            evaluator.add_batch(**rv)

            if cfgt.output_result:
                try:
                    self.output_f(**rv, cnt=paras['eval_cnt'])
                except Exception:
                    self.output_f(**rv)

            if idx % cfgt.log_display == cfgt.log_display - 1:
                print_log('processed.. {}, Time:{:.2f}s'.format(
                    idx + 1, timeit.default_timer() - time_check))
                time_check = timeit.default_timer()

        # NOTE: if you use jittor.dataset with mpi sharding, sample_n logic depends on your dataset.
        try:
            sample_n = len(getattr(evalloader, "dataset", evalloader))
        except Exception:
            sample_n = 0
        evaluator.set_sample_n(sample_n)

        eval_rv = evaluator.compute()

        if sync.get_rank('global') == 0:
            evaluator.one_line_summary()
            evaluator.save(cfgt.log_dir)

        evaluator.clear_data()
        sync.nodewise_sync().barrier()
        return {'eval_rv': eval_rv}

class exec_container(object):
    def __init__(self, cfg, **kwargs):
        self.cfg = cfg
        self.registered_stages = []
        self.nodewise_sync_global_obj = sync.nodewise_sync_global()

    def register_stage(self, stage):
        self.registered_stages.append(stage)

    def __call__(self, local_rank=None, **kwargs):
        cfg = self.cfg
        cfguh().save_cfg(cfg)

        # init sync info
        sync.nodewise_sync().copy_global(self.nodewise_sync_global_obj).local_init()

        GRANK, LRANK, NRANK = sync.get_rank('all')
        GWSIZE, LWSIZE, NODES = sync.get_world_size('all')

        # set seeds (numpy + torch; jittor optional)
        if isinstance(cfg.env.rnd_seed, int):
            np.random.seed(cfg.env.rnd_seed + GRANK)
            torch.manual_seed(cfg.env.rnd_seed + GRANK)
            if jt is not None and hasattr(jt, "misc") and hasattr(jt.misc, "set_global_seed"):
                jt.misc.set_global_seed(cfg.env.rnd_seed + GRANK, different_seed_for_mpi=False)

        # torch device for frozen torch modules (vae/clip)
        if torch.cuda.is_available():
            torch.cuda.set_device(LRANK)

        time_start = timeit.default_timer()

        para = {'itern_total': 0}

        dl_para = self.prepare_dataloader()
        assert isinstance(dl_para, dict)
        para.update(dl_para)

        md_para = self.prepare_model()
        assert isinstance(md_para, dict)
        para.update(md_para)

        for stage in self.registered_stages:
            # print_log(f'Executing stage: {stage.__class__.__name__}')
            stage_para = stage(**para)
            if stage_para is not None:
                para.update(stage_para)

        # last save (all ranks enter; safe save inside)
        self.save_last_model(**para)

        if GRANK == 0:
            print_log('Total {:.2f} seconds'.format(timeit.default_timer() - time_start))

        sync.nodewise_sync().barrier()

    def prepare_dataloader(self):
        return {'trainloader': None, 'evalloader': None}

    def prepare_model(self):
        return {'net': None}

    def save_last_model(self, **para):
        return

    def destroy(self):
        self.nodewise_sync_global_obj.destroy()

class train(exec_container):
    def prepare_dataloader(self):
        cfg = cfguh().cfg
        gws = sync.get_world_size("global")
        trainset = get_dataset()(cfg.train.dataset)

        # IMPORTANT: Jittor Dataset batch_size is global total batch size (sum over all ranks). :contentReference[oaicite:3]{index=3}
        total_bs = int(cfg.train.batch_size_per_gpu) * int(gws)
        if hasattr(trainset, "set_attrs"):
            trainset.set_attrs(
                batch_size=total_bs,
                shuffle=bool(cfg.train.dataset.get("shuffle", True)),
                num_workers=int(cfg.train.dataset_num_workers_per_gpu),
                drop_last=False,
            )
            trainloader = trainset
        else:
            # fallback (not recommended in MPI mode)
            raise RuntimeError("trainset is not a Jittor Dataset; please migrate dataset pipeline first.")

        evalloader = None
        if 'eval' in cfg:
            evalset = get_dataset()(cfg.eval.dataset)
            if evalset is not None:
                total_bs_eval = int(cfg.eval.batch_size_per_gpu) * int(gws)
                if hasattr(evalset, "set_attrs"):
                    evalset.set_attrs(
                        batch_size=total_bs_eval,
                        shuffle=False,
                        num_workers=int(cfg.eval.dataset_num_workers_per_gpu),
                        drop_last=False,
                    )
                    evalloader = evalset
        # print_log("Dataloaders prepared.")
        return {'trainloader': trainloader, 'evalloader': evalloader}

    def prepare_model(self):
        cfg = cfguh().cfg
        net = get_model()(cfg.model)

        if _is_jittor_module(net) and sync.is_ddp() and hasattr(net, "mpi_param_broadcast"):
            net.mpi_param_broadcast(0)

        sd = torch.load(cfg.model_pretrain_path, map_location='cpu')
        net.load_state_dict(sd, strict=False)

        scheduler = get_scheduler()(cfg.train.scheduler)
        optimizer = get_optimizer()(net, cfg.train.optimizer)

        return {'net': net, 'optimizer': optimizer, 'scheduler': scheduler}

    def save_last_model(self, **para):
        cfgt = cfguh().cfg.train
        net = para['net']
        net_symbol = cfguh().cfg.model.symbol
        path = osp.join(cfgt.log_dir, f'{cfgt.experiment_id}_{net_symbol}_last')

        # reuse train_stage saver style
        ts = train_stage()
        ts.save_model(net, path)

class eval(exec_container):
    def prepare_dataloader(self):
        cfg = cfguh().cfg
        gws = sync.get_world_size("global")
        evalloader = None
        if cfg.eval.get('dataset', None) is not None:
            evalset = get_dataset()(cfg.eval.dataset)
            if evalset is None:
                return
            total_bs_eval = int(cfg.eval.batch_size_per_gpu) * int(gws)
            if hasattr(evalset, "set_attrs"):
                evalset.set_attrs(
                    batch_size=total_bs_eval,
                    shuffle=False,
                    num_workers=int(cfg.eval.dataset_num_workers_per_gpu),
                    drop_last=False,
                )
                evalloader = evalset
        return {
            'trainloader' : None,
            'evalloader'  : evalloader,}

    def prepare_model(self):
        cfg = cfguh().cfg
        net = get_model()(cfg.model)
        if _is_jittor_module(net) and sync.is_ddp() and hasattr(net, "mpi_param_broadcast"):
            net.mpi_param_broadcast(0)

        sd = torch.load(cfg.model_pretrain_path, map_location='cpu')
        net.load_state_dict(sd, strict=False)
        net.eval()
        return {'net' : net,}

    def save_last_model(self, **para):
        return

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)