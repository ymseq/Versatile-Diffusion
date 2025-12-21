# lib/vd_finetune_stage.py
import jittor as jt
from lib.log_service import print_log
from lib.utils import eval_stage
from lib.cfg_holder import cfg_unique_holder as cfguh
from lib import sync
import torch


class ImageToTextEvalStage(eval_stage):
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
            c_img = core.ctx_encode(images, which="image")
        c_img = jt.array(c_img.cpu().numpy())
        c_img = c_img.cuda()

        max_length = getattr(cfgt, "max_txt_len", 77)
        with torch.no_grad():
            x_text = core.vae["text"].encode(captions, max_length=max_length)
        x_text = jt.array(x_text.cpu().numpy())
        x_text = x_text.cuda()

        x_info = {
            "type": "text",
            "x": x_text,
        }
        c_info = {
            "type": "image",
            "c": c_img,
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

