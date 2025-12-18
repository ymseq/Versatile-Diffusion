# lib/vd_finetune_stage.py
import torch

from lib.utils import train_stage
from lib.cfg_holder import cfg_unique_holder as cfguh
from lib import sync


class VDFinetuneTrainStage(train_stage):
    """
    用于微调 Versatile Diffusion 的 Image→Text 路径。

    假设：
      - Dataset 返回 (image_tensor, caption_str)
      - net 是 VD_v2_0（DDP 包了一层也没关系）
      - 文本 VAE 用的是 optimus_vae_next，挂在 net.vae["text"]
      - 图像 context encoder 用的是 clip_image_context_encoder，挂在 ctx["image"]
    """

    def __init__(self):
        super().__init__()

    def _get_core_net(self, net):
        """从 DDP / DataParallel 中拿到真正的 VD_v2_0 实例"""
        if isinstance(net, (torch.nn.DataParallel,
                            torch.nn.parallel.DistributedDataParallel)):
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

        # 1. 设置当前学习率（scheduler 已经算好 lr 传进来了）
        if lr is not None:
            for pg in optimizer.param_groups:
                pg["lr"] = lr
        

        # 2. 解包 batch
        #    来自你写的 MyVDDataset: return image, caption
        images, captions = batch   # images: [B,3,H,W], captions: list[str]

        # 3. 搬到当前 GPU
        local_rank = sync.get_rank('local')
        device = torch.device(f"cuda:{local_rank}" if cfg.env.cuda else "cpu")
        images = images.to(device)

        # 4. 取出真正的 VD_v2_0 对象（去掉 DDP 包装）
        core = self._get_core_net(net)

        # 5. 图像 context：用 ctx["image"] 做 encode
        #    VD 已经帮你封装成 ctx_encode 接口了
        with torch.no_grad():
            # which="image" 对应 ctx_cfg_list 里的那个 key
            c_img = core.ctx_encode(images, which="image")
            c_img = c_img.to(device)

        # 6. 文本 latent：用 optimus_vae_next.encode(captions)
        #    core.vae["text"] 就是你注册的 optimus_vae_next
        text_vae = core.vae["text"]
        # max_length 可以自己调，这里用 77（CLIP 习惯）或你想要的值
        max_length = getattr(cfgt, "max_txt_len", 77)
        z_txt = text_vae.encode(captions, max_length=max_length)  # [B, nz]
        z_txt = z_txt.to(device)

        # 7. 组装 x_info / c_info，喂给 VD
        #    type 字符串要和 diffuser_cfg_list 里的 key 对上
        x_info = {
            "type": "text",   # 使用 diffuser["text"] 这条 data flow
            "x": z_txt,
        }
        c_info = {
            "type": "image",  # 使用 diffuser["image"] 这条 context flow
            "c": c_img,
        }

        # 可选：batch 开始的回调（VD 内部目前是空实现）
        if hasattr(core, "on_train_batch_start"):
            core.on_train_batch_start(x_info["x"])

        # 8. 调用 VD_v2_0.forward：
        #    - 内部会随机采 t
        #    - 用 q_sample(x, t, noise) 对 x 加噪
        #    - apply_model 做一次 U-Net / FC-ResNet 去噪预测
        #    - 算好 loss, loss_dict
        loss, loss_dict = core(x_info, c_info)

        # 9. 梯度累积：除以 gradacc_every 再 backward
        gradacc_every = cfgt.get("gradacc_every", 1)
        loss = loss / gradacc_every
        loss.backward()

        # 10. 到真正该 update 的 step 再 step / zero_grad / EMA
        if grad_update:
            max_grad_norm = cfgt.get("max_grad_norm", None)
            if max_grad_norm is not None and max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(core.parameters(), max_grad_norm)

            optimizer.step()
            optimizer.zero_grad()

            if hasattr(core, "on_train_batch_end"):
                core.on_train_batch_end()

        # 11. 整理 log_info 给 train_stage.__call__ 的 log_manager 用
        log_info = {}
        for k, v in loss_dict.items():
            if torch.is_tensor(v):
                log_info[k] = v.detach().mean().item()
            else:
                log_info[k] = float(v)

        # 这里的 loss 已经 /gradacc_every 之后了，
        # 一般可以直接拿来当主 Loss 看
        log_info["Loss"] = loss.detach().mean().item()
        if lr is not None:
            log_info["lr"] = lr

        return {
            "log_info": log_info,
        }
