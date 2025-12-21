# lib/vd_finetune_stage.py
import jittor as jt
from lib.log_service import print_log
from lib.utils import train_stage
from lib.cfg_holder import cfg_unique_holder as cfguh
from lib import sync
import torch


class TextToImageTrainStage(train_stage):
    def __init__(self):
        super().__init__()

    def _get_core_net(self, net):
        if hasattr(net, "module"):
            return net.module
        return net

    def main(self,
             batch,
             lr,
             itern,
             epochn,
             samplen,
             isinit,
             grad_update,
             net,
             optimizer,
             scheduler,
             **paras):

        cfg = cfguh().cfg
        cfgt = cfg.train

        if lr is not None:
            if hasattr(optimizer, "param_groups"):
                for pg in optimizer.param_groups:
                    pg["lr"] = lr
            elif hasattr(optimizer, "lr"):
                optimizer.lr = lr


        images, captions = batch
        core = self._get_core_net(net)

        images = torch.from_numpy(images.cpu().numpy())
        images = images.cuda()
        with torch.no_grad():
            x_image = core.vae["image"].encode(images)
        x_image = jt.array(x_image.cpu().numpy())
        x_image = x_image.cuda()

        with torch.no_grad():              
            c_text = core.ctx_encode(captions, which="text")
        c_text = jt.array(c_text.cpu().numpy())
        c_text = c_text.cuda()        

        x_info = {
            "type": "image",
            "x": x_image,
        }
        c_info = {
            "type": "text",
            "c": c_text,
        }

        loss, loss_dict = core(x_info, c_info)
        gradacc_every = cfgt.get("gradacc_every", 1)
        accum_start = (itern % gradacc_every) == 0
        if accum_start:
            optimizer.zero_grad()        
        loss_scaled = loss / gradacc_every
        optimizer.backward(loss_scaled)

        if grad_update:
            optimizer.step()
            optimizer.zero_grad()
 
        log_info = {}
        for k, v in loss_dict.items():
            if isinstance(v, jt.Var):
                log_info[k] = float(v.detach().mean().item())
            else:
                log_info[k] = float(v)

        log_info["Loss"] = float(loss.detach().mean().item())
        if lr is not None:
            log_info["lr"] = lr

        return {
            "log_info": log_info,
        }

