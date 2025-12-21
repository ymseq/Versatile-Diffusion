# lib/vd_finetune_stage.py
import jittor as jt
from lib.log_service import print_log
from lib.utils import eval_stage
from lib.cfg_holder import cfg_unique_holder as cfguh
from lib import sync
import torch


class TextToImageEvalStage(eval_stage):
    def __init__(self):
        super().__init__()

    def _get_core_net(self, net):
        if hasattr(net, "module"):
            return net.module
        return net

    def main(self,
             batch,
             net):

        cfg = cfguh().cfg
        cfgt = cfg.train

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

        with jt.no_grad():
            loss, loss_dict = core(x_info, c_info)

        bs = 1
        try:
            images = batch[0]
            bs = int(getattr(images, "shape", [len(images)])[0])
        except Exception:
            bs = 1

        return {
            "loss": loss,
            "bs": bs,
        }