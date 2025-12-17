from abc import abstractmethod
from functools import partial
import math
import copy
import numpy as np

import jittor as jt
from jittor import nn

from .diffusion_utils import (
    checkpoint, conv_nd, linear, avg_pool_nd,
    zero_module, normalization, timestep_embedding
)
from .attention import (
    SpatialTransformer,
    SpatialTransformerNoContext,
    DualSpatialTransformer,
)

from lib.model_zoo.common.get_model import get_model, register

symbol = "openai"

# dummy replace
def convert_module_to_f16(x): pass
def convert_module_to_f32(x): pass


def _expand_to_nd(x, target_nd):
    while len(x.shape) < target_nd:
        x = x.unsqueeze(-1)
    return x


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads_channels: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            jt.randn(embed_dim, spacial_dim * spacial_dim + 1) / (embed_dim ** 0.5)
        )
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        self.num_heads = embed_dim // num_heads_channels
        self.attention = QKVAttention(self.num_heads)

    def execute(self, x):
        b, c = x.shape[0], x.shape[1]
        x = x.reshape(b, c, -1)
        x = jt.concat([x.mean(dim=-1, keepdims=True), x], dim=-1)
        x = x + self.positional_embedding.unsqueeze(0).cast(x.dtype)
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:, :, 0]


class TimestepBlock(nn.Module):
    @abstractmethod
    def execute(self, x, emb):
        pass


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    def execute(self, x, emb, context=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=padding)

    def execute(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = nn.interpolate(
                x,
                size=(x.shape[2], x.shape[3] * 2, x.shape[4] * 2),
                mode="nearest",
            )
        else:
            x = nn.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class TransposedUpsample(nn.Module):
    def __init__(self, channels, out_channels=None, ks=5):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.up = nn.ConvTranspose2d(self.channels, self.out_channels, kernel_size=ks, stride=2)

    def execute(self, x):
        return self.up(x)


class Downsample(nn.Module):
    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(dims, self.channels, self.out_channels, 3, stride=stride, padding=padding)
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def execute(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down
        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(emb_channels, 2 * self.out_channels if use_scale_shift_norm else self.out_channels),
        )

        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 3, padding=1)
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def execute(self, x, emb):
        return checkpoint(self._execute, (x, emb), self.parameters(), self.use_checkpoint)

    def _execute(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        emb_out = self.emb_layers(emb).cast(h.dtype)

        if self.use_scale_shift_norm:
            scale = emb_out[:, : self.out_channels]
            shift = emb_out[:, self.out_channels : 2 * self.out_channels]
            scale = _expand_to_nd(scale, len(h.shape))
            shift = _expand_to_nd(shift, len(h.shape))
            out_norm = self.out_layers[0]
            out_rest = self.out_layers[1:]
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            emb_out = _expand_to_nd(emb_out, len(h.shape))
            h = h + emb_out
            h = self.out_layers(h)

        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=1, num_head_channels=-1, use_checkpoint=False, use_new_attention_order=False):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert channels % num_head_channels == 0
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        self.attention = QKVAttention(self.num_heads) if use_new_attention_order else QKVAttentionLegacy(self.num_heads)
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def execute(self, x):
        return checkpoint(self._execute, (x,), self.parameters(), True)

    def _execute(self, x):
        b, c = x.shape[0], x.shape[1]
        spatial = x.shape[2:]
        x_in = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x_in))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x_in + h).reshape(b, c, *spatial)


class QKVAttentionLegacy(nn.Module):
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def execute(self, qkv):
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)

        qkv_ = qkv.reshape(bs * self.n_heads, ch * 3, length)
        q = qkv_[:, 0:ch, :]
        k = qkv_[:, ch:2 * ch, :]
        v = qkv_[:, 2 * ch:3 * ch, :]

        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = jt.einsum("bct,bcs->bts", q * scale, k * scale)
        weight = nn.softmax(weight.cast("float32"), dim=-1).cast(weight.dtype)
        a = jt.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)


class QKVAttention(nn.Module):
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def execute(self, qkv):
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)

        hch = self.n_heads * ch
        q = qkv[:, 0:hch, :]
        k = qkv[:, hch:2 * hch, :]
        v = qkv[:, 2 * hch:3 * hch, :]

        scale = 1 / math.sqrt(math.sqrt(ch))
        q2 = (q * scale).reshape(bs * self.n_heads, ch, length)
        k2 = (k * scale).reshape(bs * self.n_heads, ch, length)
        weight = jt.einsum("bct,bcs->bts", q2, k2)
        weight = nn.softmax(weight.cast("float32"), dim=-1).cast(weight.dtype)

        v2 = v.reshape(bs * self.n_heads, ch, length)
        a = jt.einsum("bts,bcs->bct", weight, v2)
        return a.reshape(bs, -1, length)


@register("openai_unet")
class UNetModel(nn.Module):
    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=False,
        transformer_depth=1,
        context_dim=None,
        n_embed=None,
        legacy=True,
        disable_self_attentions=None,
        num_attention_blocks=None,
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, "context_dim required when use_spatial_transformer=True"
        if context_dim is not None:
            assert use_spatial_transformer, "use_spatial_transformer required when context_dim is provided"
            try:
                from omegaconf.listconfig import ListConfig
                if isinstance(context_dim, ListConfig):
                    context_dim = list(context_dim)
            except Exception:
                pass

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads
        if num_heads == -1:
            assert num_head_channels != -1
        if num_head_channels == -1:
            assert num_heads != -1

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels

        if isinstance(num_res_blocks, int):
            self.num_res_blocks = [num_res_blocks] * len(channel_mult)
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("num_res_blocks must be int or list with same length as channel_mult")
            self.num_res_blocks = num_res_blocks

        if disable_self_attentions is not None:
            assert len(disable_self_attentions) == len(channel_mult)

        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(self.num_res_blocks[i] >= num_attention_blocks[i] for i in range(len(num_attention_blocks)))

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = "float16" if use_fp16 else "float32"
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )
        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        self.input_blocks = nn.ModuleList([TimestepEmbedSequential(conv_nd(dims, in_channels, model_channels, 3, padding=1))])
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1

        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                        nh = num_heads
                    else:
                        nh = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        dim_head = ch // nh if use_spatial_transformer else num_head_channels

                    disabled_sa = disable_self_attentions[level] if disable_self_attentions is not None else False
                    if (num_attention_blocks is None) or (nr < num_attention_blocks[level]):
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=nh,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            )
                            if not use_spatial_transformer
                            else SpatialTransformer(
                                ch,
                                nh,
                                dim_head,
                                depth=transformer_depth,
                                context_dim=context_dim,
                                disable_self_attn=disabled_sa,
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)

            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
            nh = num_heads
        else:
            nh = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            dim_head = ch // nh if use_spatial_transformer else num_head_channels

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=nh,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            )
            if not use_spatial_transformer
            else SpatialTransformer(ch, nh, dim_head, depth=transformer_depth, context_dim=context_dim),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(self.num_res_blocks[level] + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                        nh2 = num_heads
                    else:
                        nh2 = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        dim_head = ch // nh2 if use_spatial_transformer else num_head_channels

                    disabled_sa = disable_self_attentions[level] if disable_self_attentions is not None else False
                    if (num_attention_blocks is None) or (i < num_attention_blocks[level]):
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads_upsample,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            )
                            if not use_spatial_transformer
                            else SpatialTransformer(
                                ch, nh2, dim_head, depth=transformer_depth, context_dim=context_dim, disable_self_attn=disabled_sa
                            )
                        )
                if level and i == self.num_res_blocks[level]:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )
        if self.predict_codebook_ids:
            self.id_predictor = nn.Sequential(
                normalization(ch),
                conv_nd(dims, model_channels, n_embed, 1),
            )

    def convert_to_fp16(self):
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def execute(self, x, timesteps=None, context=None, y=None, **kwargs):
        assert (y is not None) == (self.num_classes is not None)
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        if self.num_classes is not None:
            emb = emb + self.label_emb(y)

        h = x.cast(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb, context)
            hs.append(h)
        h = self.middle_block(h, emb, context)
        for module in self.output_blocks:
            h = jt.concat([h, hs.pop()], dim=1)
            h = module(h, emb, context)
        h = h.cast(x.dtype)

        if self.predict_codebook_ids:
            return self.id_predictor(h)
        return self.out(h)


class EncoderUNetModel(nn.Module):
    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        pool="adaptive",
        *args,
        **kwargs,
    ):
        super().__init__()
        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = "float16" if use_fp16 else "float32"
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList([TimestepEmbedSequential(conv_nd(dims, in_channels, model_channels, 3, padding=1))])
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1

        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)

            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch
        self.pool = pool

        if pool == "adaptive":
            self.out = nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                zero_module(conv_nd(dims, ch, out_channels, 1)),
                nn.Flatten(),
            )
        elif pool == "attention":
            assert num_head_channels != -1
            self.out = nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                AttentionPool2d((image_size // ds), ch, num_head_channels, out_channels),
            )
        elif pool == "spatial":
            self.out = nn.Sequential(
                nn.Linear(self._feature_size, 2048),
                nn.ReLU(),
                nn.Linear(2048, self.out_channels),
            )
        elif pool == "spatial_v2":
            self.out = nn.Sequential(
                nn.Linear(self._feature_size, 2048),
                normalization(2048),
                nn.SiLU(),
                nn.Linear(2048, self.out_channels),
            )
        else:
            raise NotImplementedError(pool)

    def convert_to_fp16(self):
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)

    def execute(self, x, timesteps):
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        results = []
        h = x.cast(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            if isinstance(self.pool, str) and self.pool.startswith("spatial"):
                results.append(h.cast(x.dtype).mean(dim=[2, 3]))
        h = self.middle_block(h, emb)
        if isinstance(self.pool, str) and self.pool.startswith("spatial"):
            results.append(h.cast(x.dtype).mean(dim=[2, 3]))
            h = jt.concat(results, dim=-1)
            return self.out(h)
        h = h.cast(x.dtype)
        return self.out(h)


@register("openai_unet_nocontext")
class UNetModelNoContext(nn.Module):
    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=False,
        transformer_depth=1,
        n_embed=None,
        legacy=True,
        num_attention_blocks=None,
    ):
        super().__init__()
        if num_heads_upsample == -1:
            num_heads_upsample = num_heads
        if num_heads == -1:
            assert num_head_channels != -1
        if num_head_channels == -1:
            assert num_heads != -1

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels

        if isinstance(num_res_blocks, int):
            self.num_res_blocks = [num_res_blocks] * len(channel_mult)
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError
            self.num_res_blocks = num_res_blocks

        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(self.num_res_blocks[i] >= num_attention_blocks[i] for i in range(len(num_attention_blocks)))

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = "float16" if use_fp16 else "float32"
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )
        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        self.input_blocks = nn.ModuleList([TimestepEmbedSequential(conv_nd(dims, in_channels, model_channels, 3, padding=1))])
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1

        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                        nh = num_heads
                    else:
                        nh = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        dim_head = ch // nh if use_spatial_transformer else num_head_channels

                    if (num_attention_blocks is None) or (nr < num_attention_blocks[level]):
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=nh,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            )
                            if not use_spatial_transformer
                            else SpatialTransformerNoContext(ch, nh, dim_head, depth=transformer_depth)
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)

            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
            nh = num_heads
        else:
            nh = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            dim_head = ch // nh if use_spatial_transformer else num_head_channels

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=nh,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            )
            if not use_spatial_transformer
            else SpatialTransformerNoContext(ch, nh, dim_head, depth=transformer_depth),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(self.num_res_blocks[level] + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                        nh2 = num_heads
                    else:
                        nh2 = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        dim_head = ch // nh2 if use_spatial_transformer else num_head_channels

                    if (num_attention_blocks is None) or (i < num_attention_blocks[level]):
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads_upsample,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            )
                            if not use_spatial_transformer
                            else SpatialTransformerNoContext(ch, nh2, dim_head, depth=transformer_depth)
                        )
                if level and i == self.num_res_blocks[level]:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )
        if self.predict_codebook_ids:
            self.id_predictor = nn.Sequential(
                normalization(ch),
                conv_nd(dims, model_channels, n_embed, 1),
            )

    def execute(self, x, timesteps):
        assert self.num_classes is None
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        h = x.cast(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = jt.concat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.cast(x.dtype)
        if self.predict_codebook_ids:
            return self.id_predictor(h)
        return self.out(h)


@register("openai_unet_nocontext_noatt")
class UNetModelNoContextNoAtt(nn.Module):
    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        use_scale_shift_norm=False,
        resblock_updown=False,
        n_embed=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels

        if isinstance(num_res_blocks, int):
            self.num_res_blocks = [num_res_blocks] * len(channel_mult)
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError
            self.num_res_blocks = num_res_blocks

        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = "float16" if use_fp16 else "float32"
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )
        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        self.input_blocks = nn.ModuleList([TimestepEmbedSequential(conv_nd(dims, in_channels, model_channels, 3, padding=1))])
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1

        for level, mult in enumerate(channel_mult):
            for _ in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(self.num_res_blocks[level] + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if level and i == self.num_res_blocks[level]:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )
        if self.predict_codebook_ids:
            self.id_predictor = nn.Sequential(
                normalization(ch),
                conv_nd(dims, model_channels, n_embed, 1),
            )

    def execute(self, x, timesteps):
        assert self.num_classes is None
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        h = x.cast(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = jt.concat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.cast(x.dtype)
        if self.predict_codebook_ids:
            return self.id_predictor(h)
        return self.out(h)


@register("openai_unet_nocontext_noatt_decoderonly")
class UNetModelNoContextNoAttDecoderOnly(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        model_channels,
        num_res_blocks,
        dropout=0,
        channel_mult=(4, 2, 1),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        use_scale_shift_norm=False,
        resblock_updown=False,
        n_embed=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels

        if isinstance(num_res_blocks, int):
            self.num_res_blocks = [num_res_blocks] * len(channel_mult)
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError
            self.num_res_blocks = num_res_blocks

        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = "float16" if use_fp16 else "float32"
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )
        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        ch = model_channels * self.channel_mult[0]
        self.output_blocks = nn.ModuleList([TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))])

        for level, mult in enumerate(channel_mult):
            for i in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if level != len(channel_mult) - 1 and (i == self.num_res_blocks[level] - 1):
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )
        if self.predict_codebook_ids:
            self.id_predictor = nn.Sequential(
                normalization(ch),
                conv_nd(dims, model_channels, n_embed, 1),
            )

    def execute(self, x, timesteps):
        assert self.num_classes is None
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        h = x.cast(self.dtype)
        for module in self.output_blocks:
            h = module(h, emb)
        h = h.cast(x.dtype)
        if self.predict_codebook_ids:
            return self.id_predictor(h)
        return self.out(h)


class TimestepEmbedSequentialExtended(nn.Sequential, TimestepBlock):
    def execute(self, x, emb, context=None, which_attn=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context)
            elif isinstance(layer, DualSpatialTransformer):
                x = layer(x, context, which=which_attn)
            else:
                x = layer(x)
        return x


@register("openai_unet_dual_context")
class UNetModelDualContext(nn.Module):
    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=False,
        transformer_depth=1,
        context_dim=None,
        n_embed=None,
        legacy=True,
        disable_self_attentions=None,
        num_attention_blocks=None,
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None
        if context_dim is not None:
            assert use_spatial_transformer
            try:
                from omegaconf.listconfig import ListConfig
                if isinstance(context_dim, ListConfig):
                    context_dim = list(context_dim)
            except Exception:
                pass

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads
        if num_heads == -1:
            assert num_head_channels != -1
        if num_head_channels == -1:
            assert num_heads != -1

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels

        if isinstance(num_res_blocks, int):
            self.num_res_blocks = [num_res_blocks] * len(channel_mult)
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError
            self.num_res_blocks = num_res_blocks

        if disable_self_attentions is not None:
            assert len(disable_self_attentions) == len(channel_mult)

        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(self.num_res_blocks[i] >= num_attention_blocks[i] for i in range(len(num_attention_blocks)))

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = "float16" if use_fp16 else "float32"
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )
        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        self.input_blocks = nn.ModuleList([TimestepEmbedSequentialExtended(conv_nd(dims, in_channels, model_channels, 3, padding=1))])
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1

        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                        nh = num_heads
                    else:
                        nh = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        dim_head = ch // nh if use_spatial_transformer else num_head_channels

                    disabled_sa = disable_self_attentions[level] if disable_self_attentions is not None else False
                    if (num_attention_blocks is None) or (nr < num_attention_blocks[level]):
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=nh,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            )
                            if not use_spatial_transformer
                            else DualSpatialTransformer(
                                ch, nh, dim_head, depth=transformer_depth, context_dim=context_dim, disable_self_attn=disabled_sa
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequentialExtended(*layers))
                input_block_chans.append(ch)

            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequentialExtended(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2

        if num_head_channels == -1:
            dim_head = ch // num_heads
            nh = num_heads
        else:
            nh = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            dim_head = ch // nh if use_spatial_transformer else num_head_channels

        self.middle_block = TimestepEmbedSequentialExtended(
            ResBlock(ch, time_embed_dim, dropout, dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm),
            AttentionBlock(
                ch, use_checkpoint=use_checkpoint, num_heads=nh, num_head_channels=dim_head, use_new_attention_order=use_new_attention_order
            )
            if not use_spatial_transformer
            else DualSpatialTransformer(ch, nh, dim_head, depth=transformer_depth, context_dim=context_dim),
            ResBlock(ch, time_embed_dim, dropout, dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm),
        )

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(self.num_res_blocks[level] + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                        nh2 = num_heads
                    else:
                        nh2 = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        dim_head = ch // nh2 if use_spatial_transformer else num_head_channels

                    disabled_sa = disable_self_attentions[level] if disable_self_attentions is not None else False
                    if (num_attention_blocks is None) or (i < num_attention_blocks[level]):
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads_upsample,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            )
                            if not use_spatial_transformer
                            else DualSpatialTransformer(
                                ch, nh2, dim_head, depth=transformer_depth, context_dim=context_dim, disable_self_attn=disabled_sa
                            )
                        )
                if level and i == self.num_res_blocks[level]:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequentialExtended(*layers))

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )
        if self.predict_codebook_ids:
            self.id_predictor = nn.Sequential(
                normalization(ch),
                conv_nd(dims, model_channels, n_embed, 1),
            )

    def execute(self, x, timesteps=None, context=None, y=None, which_attn=None, **kwargs):
        assert (y is not None) == (self.num_classes is not None)
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False).cast(context.dtype)
        emb = self.time_embed(t_emb)

        if self.num_classes is not None:
            emb = emb + self.label_emb(y)

        h = x.cast(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb, context, which_attn=which_attn)
            hs.append(h)
        h = self.middle_block(h, emb, context, which_attn=which_attn)
        for module in self.output_blocks:
            h = jt.concat([h, hs.pop()], dim=1)
            h = module(h, emb, context, which_attn=which_attn)
        h = h.cast(x.dtype)

        if self.predict_codebook_ids:
            return self.id_predictor(h)
        return self.out(h)


@register("openai_unet_2d")
class UNetModel2D(nn.Module):
    def __init__(
        self,
        input_channels,
        model_channels,
        output_channels,
        context_dim=768,
        num_noattn_blocks=(2, 2, 2, 2),
        channel_mult=(1, 2, 4, 8),
        with_attn=(True, True, True, False),
        num_heads=8,
        use_checkpoint=True,
    ):
        super().__init__()
        ResBlockPreset = partial(ResBlock, dropout=0, dims=2, use_checkpoint=use_checkpoint, use_scale_shift_norm=False)

        self.input_channels = input_channels
        self.model_channels = model_channels
        self.num_noattn_blocks = num_noattn_blocks
        self.channel_mult = channel_mult
        self.num_heads = num_heads

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        current_channel = model_channels
        input_blocks = [TimestepEmbedSequential(conv_nd(2, input_channels, model_channels, 3, padding=1))]
        input_block_channels = [current_channel]

        for level_idx, mult in enumerate(channel_mult):
            for _ in range(self.num_noattn_blocks[level_idx]):
                layers = [
                    ResBlockPreset(
                        current_channel,
                        time_embed_dim,
                        out_channels=mult * model_channels,
                    )
                ]
                current_channel = mult * model_channels
                dim_head = current_channel // num_heads
                if with_attn[level_idx]:
                    layers += [SpatialTransformer(current_channel, num_heads, dim_head, depth=1, context_dim=context_dim)]
                input_blocks += [TimestepEmbedSequential(*layers)]
                input_block_channels.append(current_channel)

            if level_idx != len(channel_mult) - 1:
                input_blocks += [TimestepEmbedSequential(Downsample(current_channel, use_conv=True, dims=2, out_channels=current_channel))]
                input_block_channels.append(current_channel)

        self.input_blocks = nn.ModuleList(input_blocks)

        middle_block = [
            ResBlockPreset(current_channel, time_embed_dim),
            SpatialTransformer(current_channel, num_heads, dim_head, depth=1, context_dim=context_dim),
            ResBlockPreset(current_channel, time_embed_dim),
        ]
        self.middle_block = TimestepEmbedSequential(*middle_block)

        output_blocks = []
        for level_idx, mult in list(enumerate(channel_mult))[::-1]:
            for block_idx in range(self.num_noattn_blocks[level_idx] + 1):
                extra_channel = input_block_channels.pop()
                layers = [
                    ResBlockPreset(
                        current_channel + extra_channel,
                        time_embed_dim,
                        out_channels=model_channels * mult,
                    )
                ]
                current_channel = model_channels * mult
                dim_head = current_channel // num_heads
                if with_attn[level_idx]:
                    layers += [SpatialTransformer(current_channel, num_heads, dim_head, depth=1, context_dim=context_dim)]
                if level_idx != 0 and block_idx == self.num_noattn_blocks[level_idx]:
                    layers += [Upsample(current_channel, use_conv=True, dims=2, out_channels=current_channel)]
                output_blocks += [TimestepEmbedSequential(*layers)]
        self.output_blocks = nn.ModuleList(output_blocks)

        self.out = nn.Sequential(
            normalization(current_channel),
            nn.SiLU(),
            zero_module(conv_nd(2, model_channels, output_channels, 3, padding=1)),
        )

    def execute(self, x, timesteps=None, context=None):
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)
        h = x
        for module in self.input_blocks:
            h = module(h, emb, context)
            hs.append(h)
        h = self.middle_block(h, emb, context)
        for module in self.output_blocks:
            h = jt.concat([h, hs.pop()], dim=1)
            h = module(h, emb, context)
        return self.out(h)


class FCBlock(TimestepBlock):
    def __init__(self, channels, emb_channels, dropout, out_channels=None, use_checkpoint=False):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_checkpoint = use_checkpoint

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(2, channels, self.out_channels, 1, padding=0),
        )
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(emb_channels, self.out_channels),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(conv_nd(2, self.out_channels, self.out_channels, 1, padding=0)),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = conv_nd(2, channels, self.out_channels, 1, padding=0)

    def execute(self, x, emb):
        x2d = x
        is_vec = False
        if len(x.shape) == 2:
            is_vec = True
            x2d = x[:, :, None, None]
        elif len(x.shape) == 4:
            pass
        else:
            raise ValueError

        y = checkpoint(self._execute, (x2d, emb), self.parameters(), self.use_checkpoint)
        if is_vec:
            return y[:, :, 0, 0]
        return y

    def _execute(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).cast(h.dtype)
        emb_out = _expand_to_nd(emb_out, len(h.shape))
        h = h + emb_out
        h = self.out_layers(h)
        return self.skip_connection(x) + h


@register("openai_unet_0d")
class UNetModel0D(nn.Module):
    def __init__(
        self,
        input_channels,
        model_channels,
        output_channels,
        context_dim=768,
        num_noattn_blocks=(2, 2, 2, 2),
        channel_mult=(1, 2, 4, 8),
        with_attn=(True, True, True, False),
        num_heads=8,
        use_checkpoint=True,
    ):
        super().__init__()
        FCBlockPreset = partial(FCBlock, dropout=0, use_checkpoint=use_checkpoint)

        self.input_channels = input_channels
        self.model_channels = model_channels
        self.num_noattn_blocks = num_noattn_blocks
        self.channel_mult = channel_mult
        self.num_heads = num_heads

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        current_channel = model_channels
        input_blocks = [TimestepEmbedSequential(conv_nd(2, input_channels, model_channels, 1, padding=0))]
        input_block_channels = [current_channel]

        for level_idx, mult in enumerate(channel_mult):
            for _ in range(self.num_noattn_blocks[level_idx]):
                layers = [FCBlockPreset(current_channel, time_embed_dim, out_channels=mult * model_channels)]
                current_channel = mult * model_channels
                dim_head = current_channel // num_heads
                if with_attn[level_idx]:
                    layers += [SpatialTransformer(current_channel, num_heads, dim_head, depth=1, context_dim=context_dim)]
                input_blocks += [TimestepEmbedSequential(*layers)]
                input_block_channels.append(current_channel)

            if level_idx != len(channel_mult) - 1:
                input_blocks += [TimestepEmbedSequential(Downsample(current_channel, use_conv=True, dims=2, out_channels=current_channel))]
                input_block_channels.append(current_channel)

        self.input_blocks = nn.ModuleList(input_blocks)

        middle_block = [
            FCBlockPreset(current_channel, time_embed_dim),
            SpatialTransformer(current_channel, num_heads, dim_head, depth=1, context_dim=context_dim),
            FCBlockPreset(current_channel, time_embed_dim),
        ]
        self.middle_block = TimestepEmbedSequential(*middle_block)

        output_blocks = []
        for level_idx, mult in list(enumerate(channel_mult))[::-1]:
            for block_idx in range(self.num_noattn_blocks[level_idx] + 1):
                extra_channel = input_block_channels.pop()
                layers = [
                    FCBlockPreset(current_channel + extra_channel, time_embed_dim, out_channels=model_channels * mult)
                ]
                current_channel = model_channels * mult
                dim_head = current_channel // num_heads
                if with_attn[level_idx]:
                    layers += [SpatialTransformer(current_channel, num_heads, dim_head, depth=1, context_dim=context_dim)]
                if level_idx != 0 and block_idx == self.num_noattn_blocks[level_idx]:
                    layers += [conv_nd(2, current_channel, current_channel, 1, padding=0)]
                output_blocks += [TimestepEmbedSequential(*layers)]
        self.output_blocks = nn.ModuleList(output_blocks)

        self.out = nn.Sequential(
            normalization(current_channel),
            nn.SiLU(),
            zero_module(conv_nd(2, model_channels, output_channels, 1, padding=0)),
        )

    def execute(self, x, timesteps=None, context=None):
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)
        h = x
        for module in self.input_blocks:
            h = module(h, emb, context)
            hs.append(h)
        h = self.middle_block(h, emb, context)
        for module in self.output_blocks:
            h = jt.concat([h, hs.pop()], dim=1)
            h = module(h, emb, context)
        return self.out(h)


class Linear_MultiDim(nn.Linear):
    def __init__(self, in_features, out_features, *args, **kwargs):
        in_features = [in_features] if isinstance(in_features, int) else list(in_features)
        out_features = [out_features] if isinstance(out_features, int) else list(out_features)
        self.in_features_multidim = in_features
        self.out_features_multidim = out_features
        super().__init__(int(np.prod(in_features)), int(np.prod(out_features)), *args, **kwargs)

    def execute(self, x):
        shape = x.shape
        n = len(shape) - len(self.in_features_multidim)
        x = x.reshape(*shape[:n], int(np.prod(self.in_features_multidim)))
        y = super().execute(x)
        y = y.reshape(*shape[:n], *self.out_features_multidim)
        return y


class FCBlock_MultiDim(FCBlock):
    def __init__(self, channels, emb_channels, dropout, out_channels=None, use_checkpoint=False):
        channels = [channels] if isinstance(channels, int) else list(channels)
        channels_all = int(np.prod(channels))
        self.channels_multidim = channels

        if out_channels is not None:
            out_channels = [out_channels] if isinstance(out_channels, int) else list(out_channels)
            out_channels_all = int(np.prod(out_channels))
            self.out_channels_multidim = out_channels
        else:
            out_channels_all = channels_all
            self.out_channels_multidim = self.channels_multidim

        super().__init__(channels=channels_all, emb_channels=emb_channels, dropout=dropout, out_channels=out_channels_all, use_checkpoint=use_checkpoint)

    def execute(self, x, emb):
        shape = x.shape
        n = len(self.channels_multidim)
        x = x.reshape(*shape[:-n], int(np.prod(self.channels_multidim)), 1, 1)
        x = x.reshape(-1, int(np.prod(self.channels_multidim)), 1, 1)
        y = checkpoint(self._execute, (x, emb), self.parameters(), self.use_checkpoint)
        y = y.reshape(*shape[:-n], int(np.prod(self.out_channels_multidim)))
        y = y.reshape(*shape[:-n], *self.out_channels_multidim)
        return y


@register("openai_unet_0dmd")
class UNetModel0D_MultiDim(nn.Module):
    def __init__(
        self,
        input_channels,
        model_channels,
        output_channels,
        context_dim=768,
        num_noattn_blocks=(2, 2, 2, 2),
        channel_mult=(1, 2, 4, 8),
        second_dim=(4, 4, 4, 4),
        with_attn=(True, True, True, False),
        num_heads=8,
        use_checkpoint=True,
    ):
        super().__init__()
        FCBlockPreset = partial(FCBlock_MultiDim, dropout=0, use_checkpoint=use_checkpoint)

        self.input_channels = input_channels
        self.model_channels = model_channels
        self.num_noattn_blocks = num_noattn_blocks
        self.channel_mult = channel_mult
        self.second_dim = second_dim
        self.num_heads = num_heads

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        sdim = second_dim[0]
        current_channel = [model_channels, sdim, 1]
        input_blocks = [TimestepEmbedSequential(Linear_MultiDim([input_channels, 1, 1], current_channel, bias=True))]
        input_block_channels = [current_channel]

        for level_idx, (mult, sdim) in enumerate(zip(channel_mult, second_dim)):
            for _ in range(self.num_noattn_blocks[level_idx]):
                layers = [
                    FCBlockPreset(
                        current_channel,
                        time_embed_dim,
                        out_channels=[mult * model_channels, sdim, 1],
                    )
                ]
                current_channel = [mult * model_channels, sdim, 1]
                dim_head = current_channel[0] // num_heads
                if with_attn[level_idx]:
                    layers += [SpatialTransformer(current_channel[0], num_heads, dim_head, depth=1, context_dim=context_dim)]
                input_blocks += [TimestepEmbedSequential(*layers)]
                input_block_channels.append(current_channel)

            if level_idx != len(channel_mult) - 1:
                input_blocks += [TimestepEmbedSequential(Linear_MultiDim(current_channel, current_channel, bias=True))]
                input_block_channels.append(current_channel)

        self.input_blocks = nn.ModuleList(input_blocks)

        middle_block = [
            FCBlockPreset(current_channel, time_embed_dim),
            SpatialTransformer(current_channel[0], num_heads, dim_head, depth=1, context_dim=context_dim),
            FCBlockPreset(current_channel, time_embed_dim),
        ]
        self.middle_block = TimestepEmbedSequential(*middle_block)

        output_blocks = []
        for level_idx, (mult, sdim) in list(enumerate(zip(channel_mult, second_dim)))[::-1]:
            for block_idx in range(self.num_noattn_blocks[level_idx] + 1):
                extra_channel = input_block_channels.pop()
                layers = [
                    FCBlockPreset(
                        [current_channel[0] + extra_channel[0]] + current_channel[1:],
                        time_embed_dim,
                        out_channels=[mult * model_channels, sdim, 1],
                    )
                ]
                current_channel = [mult * model_channels, sdim, 1]
                dim_head = current_channel[0] // num_heads
                if with_attn[level_idx]:
                    layers += [SpatialTransformer(current_channel[0], num_heads, dim_head, depth=1, context_dim=context_dim)]
                if level_idx != 0 and block_idx == self.num_noattn_blocks[level_idx]:
                    layers += [Linear_MultiDim(current_channel, current_channel, bias=True)]
                output_blocks += [TimestepEmbedSequential(*layers)]
        self.output_blocks = nn.ModuleList(output_blocks)

        self.out = nn.Sequential(
            normalization(current_channel[0]),
            nn.SiLU(),
            zero_module(Linear_MultiDim(current_channel, [output_channels, 1, 1], bias=True)),
        )

    def execute(self, x, timesteps=None, context=None):
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        h = x
        for module in self.input_blocks:
            h = module(h, emb, context)
            hs.append(h)
        h = self.middle_block(h, emb, context)
        for module in self.output_blocks:
            h = jt.concat([h, hs.pop()], dim=1)
            h = module(h, emb, context)
        return self.out(h)


@register("openai_unet_vd")
class UNetModelVD(nn.Module):
    def __init__(self, unet_image_cfg, unet_text_cfg):
        super().__init__()
        self.unet_image = get_model()(unet_image_cfg)
        self.unet_text = get_model()(unet_text_cfg)
        self.time_embed = self.unet_image.time_embed
        del self.unet_image.time_embed
        del self.unet_text.time_embed
        self.model_channels = self.unet_image.model_channels

    def execute(self, x, timesteps, context, xtype="image", ctype="prompt"):
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb.cast(x.dtype))

        if xtype == "text":
            x = x[:, :, None, None]

        h = x
        for i_module, t_module in zip(self.unet_image.input_blocks, self.unet_text.input_blocks):
            h = self.mixed_run(i_module, t_module, h, emb, context, xtype, ctype)
            hs.append(h)

        h = self.mixed_run(self.unet_image.middle_block, self.unet_text.middle_block, h, emb, context, xtype, ctype)

        for i_module, t_module in zip(self.unet_image.output_blocks, self.unet_text.output_blocks):
            h = jt.concat([h, hs.pop()], dim=1)
            h = self.mixed_run(i_module, t_module, h, emb, context, xtype, ctype)

        if xtype == "image":
            return self.unet_image.out(h)
        return self.unet_text.out(h).squeeze(-1).squeeze(-1)

    def mixed_run(self, inet, tnet, x, emb, context, xtype, ctype):
        h = x
        for ilayer, tlayer in zip(inet, tnet):
            if isinstance(ilayer, TimestepBlock) and xtype == "image":
                h = ilayer(h, emb)
            elif isinstance(tlayer, TimestepBlock) and xtype == "text":
                h = tlayer(h, emb)
            elif isinstance(ilayer, SpatialTransformer) and ctype == "vision":
                h = ilayer(h, context)
            elif isinstance(ilayer, SpatialTransformer) and ctype == "prompt":
                h = tlayer(h, context)
            elif xtype == "image":
                h = ilayer(h)
            elif xtype == "text":
                h = tlayer(h)
            else:
                raise ValueError
        return h

    def forward_dc(self, x, timesteps, c0, c1, xtype, c0_type, c1_type, mixed_ratio):
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb.cast(x.dtype))

        if xtype == "text":
            x = x[:, :, None, None]
        h = x

        for i_module, t_module in zip(self.unet_image.input_blocks, self.unet_text.input_blocks):
            h = self.mixed_run_dc(i_module, t_module, h, emb, c0, c1, xtype, c0_type, c1_type, mixed_ratio)
            hs.append(h)

        h = self.mixed_run_dc(self.unet_image.middle_block, self.unet_text.middle_block, h, emb, c0, c1, xtype, c0_type, c1_type, mixed_ratio)

        for i_module, t_module in zip(self.unet_image.output_blocks, self.unet_text.output_blocks):
            h = jt.concat([h, hs.pop()], dim=1)
            h = self.mixed_run_dc(i_module, t_module, h, emb, c0, c1, xtype, c0_type, c1_type, mixed_ratio)

        if xtype == "image":
            return self.unet_image.out(h)
        return self.unet_text.out(h).squeeze(-1).squeeze(-1)

    def mixed_run_dc(self, inet, tnet, x, emb, c0, c1, xtype, c0_type, c1_type, mixed_ratio):
        h = x
        for ilayer, tlayer in zip(inet, tnet):
            if isinstance(ilayer, TimestepBlock) and xtype == "image":
                h = ilayer(h, emb)
            elif isinstance(tlayer, TimestepBlock) and xtype == "text":
                h = tlayer(h, emb)
            elif isinstance(ilayer, SpatialTransformer):
                h0 = (ilayer(h, c0) - h) if c0_type == "vision" else (tlayer(h, c0) - h)
                h1 = (ilayer(h, c1) - h) if c1_type == "vision" else (tlayer(h, c1) - h)
                h = h0 * mixed_ratio + h1 * (1 - mixed_ratio) + h
            elif xtype == "image":
                h = ilayer(h)
            elif xtype == "text":
                h = tlayer(h)
            else:
                raise ValueError
        return h


@register("openai_unet_2d_next")
class UNetModel2D_Next(nn.Module):
    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        context_dim,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        use_checkpoint=False,
        num_heads=8,
        num_head_channels=None,
        parts=("global", "data", "context"),
    ):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels

        if isinstance(num_res_blocks, int):
            self.num_res_blocks = [num_res_blocks] * len(channel_mult)
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError
            self.num_res_blocks = num_res_blocks

        self.attention_resolutions = attention_resolutions
        self.context_dim = context_dim
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        assert (num_heads is None) + (num_head_channels is None) == 1

        self.parts = list(parts) if isinstance(parts, (list, tuple)) else [parts]
        self.glayer_included = "global" in self.parts
        self.dlayer_included = "data" in self.parts
        self.clayer_included = "context" in self.parts
        self.layer_sequence_ordering = []

        time_embed_dim = model_channels * 4
        if self.glayer_included:
            self.time_embed = nn.Sequential(
                linear(model_channels, time_embed_dim),
                nn.SiLU(),
                linear(time_embed_dim, time_embed_dim),
            )

        if self.dlayer_included:
            self.data_blocks = nn.ModuleList([])
            ResBlockDefault = partial(
                ResBlock,
                emb_channels=time_embed_dim,
                dropout=dropout,
                dims=2,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=False,
            )
        else:
            ResBlockDefault = None

        if self.clayer_included:
            self.context_blocks = nn.ModuleList([])
            CrossAttnDefault = partial(SpatialTransformer, context_dim=context_dim, disable_self_attn=False)
        else:
            CrossAttnDefault = None

        def add_data_layer(layer):
            if self.dlayer_included:
                self.data_blocks.append(TimestepEmbedSequential(layer) if not isinstance(layer, (list, tuple)) else TimestepEmbedSequential(*layer))
            self.layer_sequence_ordering.append("d")

        def add_context_layer(layer):
            if self.clayer_included:
                self.context_blocks.append(TimestepEmbedSequential(layer) if not isinstance(layer, (list, tuple)) else TimestepEmbedSequential(*layer))
            self.layer_sequence_ordering.append("c")

        self.add_data_layer = add_data_layer
        self.add_context_layer = add_context_layer

        self.add_data_layer(conv_nd(2, in_channels, model_channels, 3, padding=1))
        self.layer_sequence_ordering.append("save_hidden_feature")
        input_block_chans = [model_channels]

        ch = model_channels
        ds = 1

        for level, mult in enumerate(channel_mult):
            for _ in range(self.num_res_blocks[level]):
                self.add_data_layer(ResBlockDefault(channels=ch, out_channels=mult * model_channels))
                ch = mult * model_channels
                if ds in attention_resolutions and self.clayer_included:
                    d_head, n_heads = self.get_d_head_n_heads(ch)
                    self.add_context_layer(CrossAttnDefault(in_channels=ch, d_head=d_head, n_heads=n_heads))
                input_block_chans.append(ch)
                self.layer_sequence_ordering.append("save_hidden_feature")

            if level != len(channel_mult) - 1:
                self.add_data_layer(Downsample(ch, use_conv=True, dims=2, out_channels=ch))
                input_block_chans.append(ch)
                self.layer_sequence_ordering.append("save_hidden_feature")
                ds *= 2

        self.i_order = copy.deepcopy(self.layer_sequence_ordering)
        self.layer_sequence_ordering = []

        self.add_data_layer(ResBlockDefault(channels=ch))
        if self.clayer_included:
            d_head, n_heads = self.get_d_head_n_heads(ch)
            self.add_context_layer(CrossAttnDefault(in_channels=ch, d_head=d_head, n_heads=n_heads))
        self.add_data_layer(ResBlockDefault(channels=ch))

        self.m_order = copy.deepcopy(self.layer_sequence_ordering)
        self.layer_sequence_ordering = []

        for level, mult in list(enumerate(channel_mult))[::-1]:
            for _ in range(self.num_res_blocks[level] + 1):
                self.layer_sequence_ordering.append("load_hidden_feature")
                ich = input_block_chans.pop()
                self.add_data_layer(ResBlockDefault(channels=ch + ich, out_channels=model_channels * mult))
                ch = model_channels * mult
                if ds in attention_resolutions and self.clayer_included:
                    d_head, n_heads = self.get_d_head_n_heads(ch)
                    self.add_context_layer(CrossAttnDefault(in_channels=ch, d_head=d_head, n_heads=n_heads))
            if level != 0:
                self.add_data_layer(Upsample(ch, conv_resample, dims=2, out_channels=ch))
                ds //= 2

        self.add_data_layer(
            nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                zero_module(conv_nd(2, model_channels, out_channels, 3, padding=1)),
            )
        )

        self.o_order = copy.deepcopy(self.layer_sequence_ordering)
        self.layer_order = copy.deepcopy(self.i_order + self.m_order + self.o_order)
        del self.layer_sequence_ordering

        self.parameter_group = {}
        if self.glayer_included:
            self.parameter_group["global"] = self.time_embed
        if self.dlayer_included:
            self.parameter_group["data"] = self.data_blocks
        if self.clayer_included:
            self.parameter_group["context"] = self.context_blocks

    def get_d_head_n_heads(self, ch):
        if self.num_head_channels is None:
            return ch // self.num_heads, self.num_heads
        return self.num_head_channels, ch // self.num_head_channels

    def execute(self, x, timesteps, context):
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        d_iter = iter(self.data_blocks) if self.dlayer_included else iter([])
        c_iter = iter(self.context_blocks) if self.clayer_included else iter([])

        h = x
        for ltype in self.i_order:
            if ltype == "d":
                h = next(d_iter)(h, emb, context)
            elif ltype == "c":
                h = next(c_iter)(h, emb, context)
            elif ltype == "save_hidden_feature":
                hs.append(h)

        for ltype in self.m_order:
            if ltype == "d":
                h = next(d_iter)(h, emb, context)
            elif ltype == "c":
                h = next(c_iter)(h, emb, context)

        for ltype in self.o_order:
            if ltype == "load_hidden_feature":
                h = jt.concat([h, hs.pop()], dim=1)
            elif ltype == "d":
                h = next(d_iter)(h, emb, context)
            elif ltype == "c":
                h = next(c_iter)(h, emb, context)

        return h


@register("openai_unet_0d_next")
class UNetModel0D_Next(UNetModel2D_Next):
    def __init__(
        self,
        input_channels,
        model_channels,
        output_channels,
        context_dim=788,
        num_noattn_blocks=(2, 2, 2, 2),
        channel_mult=(1, 2, 4, 8),
        second_dim=(4, 4, 4, 4),
        with_attn=(True, True, True, False),
        num_heads=8,
        num_head_channels=None,
        use_checkpoint=False,
        parts=("global", "data", "context"),
    ):
        nn.Module.__init__(self)

        self.input_channels = input_channels
        self.model_channels = model_channels
        self.output_channels = output_channels
        self.num_noattn_blocks = num_noattn_blocks
        self.channel_mult = channel_mult
        self.second_dim = second_dim
        self.with_attn = with_attn
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels

        self.parts = list(parts) if isinstance(parts, (list, tuple)) else [parts]
        self.glayer_included = "global" in self.parts
        self.dlayer_included = "data" in self.parts
        self.clayer_included = "context" in self.parts
        self.layer_sequence_ordering = []

        time_embed_dim = model_channels * 4
        if self.glayer_included:
            self.time_embed = nn.Sequential(
                linear(model_channels, time_embed_dim),
                nn.SiLU(),
                linear(time_embed_dim, time_embed_dim),
            )

        if self.dlayer_included:
            self.data_blocks = nn.ModuleList([])
            FCBlockDefault = partial(FCBlock_MultiDim, dropout=0, use_checkpoint=use_checkpoint)
        else:
            FCBlockDefault = None

        if self.clayer_included:
            self.context_blocks = nn.ModuleList([])
            CrossAttnDefault = partial(SpatialTransformer, context_dim=context_dim, disable_self_attn=False)
        else:
            CrossAttnDefault = None

        def add_data_layer(layer):
            if self.dlayer_included:
                self.data_blocks.append(TimestepEmbedSequential(layer) if not isinstance(layer, (list, tuple)) else TimestepEmbedSequential(*layer))
            self.layer_sequence_ordering.append("d")

        def add_context_layer(layer):
            if self.clayer_included:
                self.context_blocks.append(TimestepEmbedSequential(layer) if not isinstance(layer, (list, tuple)) else TimestepEmbedSequential(*layer))
            self.layer_sequence_ordering.append("c")

        self.add_data_layer = add_data_layer
        self.add_context_layer = add_context_layer

        sdim0 = second_dim[0]
        current_channel = [model_channels, sdim0, 1]
        self.add_data_layer(Linear_MultiDim([input_channels], current_channel, bias=True))
        self.layer_sequence_ordering.append("save_hidden_feature")
        input_block_channels = [current_channel]

        for level_idx, (mult, sdim) in enumerate(zip(channel_mult, second_dim)):
            for _ in range(self.num_noattn_blocks[level_idx]):
                self.add_data_layer(
                    FCBlockDefault(current_channel, time_embed_dim, out_channels=[mult * model_channels, sdim, 1])
                )
                current_channel = [mult * model_channels, sdim, 1]

                if with_attn[level_idx] and self.clayer_included:
                    d_head, n_heads = self.get_d_head_n_heads(current_channel[0])
                    self.add_context_layer(CrossAttnDefault(in_channels=current_channel[0], d_head=d_head, n_heads=n_heads))

                input_block_channels.append(current_channel)
                self.layer_sequence_ordering.append("save_hidden_feature")

            if level_idx != len(channel_mult) - 1:
                self.add_data_layer(Linear_MultiDim(current_channel, current_channel, bias=True))
                input_block_channels.append(current_channel)
                self.layer_sequence_ordering.append("save_hidden_feature")

        self.i_order = copy.deepcopy(self.layer_sequence_ordering)
        self.layer_sequence_ordering = []

        self.add_data_layer(FCBlockDefault(current_channel, time_embed_dim))
        if self.clayer_included:
            d_head, n_heads = self.get_d_head_n_heads(current_channel[0])
            self.add_context_layer(CrossAttnDefault(in_channels=current_channel[0], d_head=d_head, n_heads=n_heads))
        self.add_data_layer(FCBlockDefault(current_channel, time_embed_dim))

        self.m_order = copy.deepcopy(self.layer_sequence_ordering)
        self.layer_sequence_ordering = []

        for level_idx, (mult, sdim) in list(enumerate(zip(channel_mult, second_dim)))[::-1]:
            for _ in range(self.num_noattn_blocks[level_idx] + 1):
                self.layer_sequence_ordering.append("load_hidden_feature")
                extra_channel = input_block_channels.pop()
                self.add_data_layer(
                    FCBlockDefault(
                        [current_channel[0] + extra_channel[0]] + current_channel[1:],
                        time_embed_dim,
                        out_channels=[mult * model_channels, sdim, 1],
                    )
                )
                current_channel = [mult * model_channels, sdim, 1]

                if with_attn[level_idx] and self.clayer_included:
                    d_head, n_heads = self.get_d_head_n_heads(current_channel[0])
                    self.add_context_layer(CrossAttnDefault(in_channels=current_channel[0], d_head=d_head, n_heads=n_heads))

            if level_idx != 0:
                self.add_data_layer(Linear_MultiDim(current_channel, current_channel, bias=True))

        self.add_data_layer(
            nn.Sequential(
                normalization(current_channel[0]),
                nn.SiLU(),
                zero_module(Linear_MultiDim(current_channel, [output_channels], bias=True)),
            )
        )

        self.o_order = copy.deepcopy(self.layer_sequence_ordering)
        self.layer_order = copy.deepcopy(self.i_order + self.m_order + self.o_order)
        del self.layer_sequence_ordering

        self.parameter_group = {}
        if self.glayer_included:
            self.parameter_group["global"] = self.time_embed
        if self.dlayer_included:
            self.parameter_group["data"] = self.data_blocks
        if self.clayer_included:
            self.parameter_group["context"] = self.context_blocks
