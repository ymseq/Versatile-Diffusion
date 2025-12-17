from inspect import isfunction
import math
import jittor as jt
from jittor import nn
from jittor.einops import rearrange, repeat

from .diffusion_utils import checkpoint


def exists(val):
    return val is not None


def uniq(arr):
    return {el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    if t.dtype == jt.float16:
        return -1e4
    return -1e9


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.assign(jt.rand(tensor.shape, dtype=tensor.dtype) * (2 * std) - std)
    return tensor


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def execute(self, x):
        xg = self.proj(x)
        x, gate = jt.chunk(xg, 2, dim=-1)
        return x * nn.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)

        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def execute(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.assign(jt.zeros_like(p))
    return module


def Normalize(in_channels):
    return nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv(hidden_dim, dim, 1)

    def execute(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)',
                            heads=self.heads, qkv=3)
        k = nn.softmax(k, dim=-1)
        context = jt.einsum('bhdn,bhen->bhde', k, v)
        out = jt.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w',
                        heads=self.heads, h=h, w=w)
        return self.to_out(out)


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.norm = Normalize(in_channels)
        self.q = nn.Conv(in_channels, in_channels, 1, stride=1, padding=0)
        self.k = nn.Conv(in_channels, in_channels, 1, stride=1, padding=0)
        self.v = nn.Conv(in_channels, in_channels, 1, stride=1, padding=0)
        self.proj_out = nn.Conv(in_channels, in_channels, 1, stride=1, padding=0)

    def execute(self, x):
        h_ = self.norm(x)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        w_ = jt.einsum('bij,bjk->bik', q, k)

        w_ = w_ * (float(c) ** (-0.5))
        w_ = nn.softmax(w_, dim=2)

        v = rearrange(v, 'b c h w -> b c (h w)')
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = jt.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out(h_)

        return x + h_


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def execute(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = jt.einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            neg = max_neg_value(sim)
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            # mask==1 keep, mask==0 drop
            sim = sim + (1 - mask) * neg

        attn = nn.softmax(sim, dim=-1)
        out = jt.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None,
                 gated_ff=True, checkpoint=True, disable_self_attn=False):
        super().__init__()
        self.disable_self_attn = disable_self_attn
        self.attn1 = CrossAttention(
            query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout,
            context_dim=context_dim if self.disable_self_attn else None
        )
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(
            query_dim=dim, context_dim=context_dim,
            heads=n_heads, dim_head=d_head, dropout=dropout
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def execute(self, x, context=None):
        return checkpoint(self._execute, (x, context), self.parameters(), self.checkpoint)

    def _execute(self, x, context=None):
        x = self.attn1(self.norm1(x), context=context if self.disable_self_attn else None) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer(nn.Module):
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None, disable_self_attn=False):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)

        self.proj_in = nn.Conv(in_channels, inner_dim, 1, stride=1, padding=0)

        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout,
                                  context_dim=context_dim, disable_self_attn=disable_self_attn)
            for _ in range(depth)
        ])

        self.proj_out = zero_module(nn.Conv(inner_dim, in_channels, 1, stride=1, padding=0))

    def execute(self, x, context=None):
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        for block in self.transformer_blocks:
            x = block(x, context=context)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.proj_out(x)
        return x + x_in


# transformer no context

class BasicTransformerBlockNoContext(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., gated_ff=True, checkpoint=True):
        super().__init__()
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout, context_dim=None)
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout, context_dim=None)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def execute(self, x):
        return checkpoint(self._execute, (x,), self.parameters(), self.checkpoint)

    def _execute(self, x):
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x)) + x
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformerNoContext(nn.Module):
    def __init__(self, in_channels, n_heads, d_head, depth=1, dropout=0.):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)

        self.proj_in = nn.Conv(in_channels, inner_dim, 1, stride=1, padding=0)

        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlockNoContext(inner_dim, n_heads, d_head, dropout=dropout)
            for _ in range(depth)
        ])

        self.proj_out = zero_module(nn.Conv(inner_dim, in_channels, 1, stride=1, padding=0))

    def execute(self, x):
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        for block in self.transformer_blocks:
            x = block(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.proj_out(x)
        return x + x_in


# Spatial Transformer with Two Branch

class DualSpatialTransformer(nn.Module):
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None, disable_self_attn=False):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head

        self.norm_0 = Normalize(in_channels)
        self.proj_in_0 = nn.Conv(in_channels, inner_dim, 1, stride=1, padding=0)
        self.transformer_blocks_0 = nn.ModuleList([
            BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout,
                                  context_dim=context_dim, disable_self_attn=disable_self_attn)
            for _ in range(depth)
        ])
        self.proj_out_0 = zero_module(nn.Conv(inner_dim, in_channels, 1, stride=1, padding=0))

        self.norm_1 = Normalize(in_channels)
        self.proj_in_1 = nn.Conv(in_channels, inner_dim, 1, stride=1, padding=0)
        self.transformer_blocks_1 = nn.ModuleList([
            BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout,
                                  context_dim=context_dim, disable_self_attn=disable_self_attn)
            for _ in range(depth)
        ])
        self.proj_out_1 = zero_module(nn.Conv(inner_dim, in_channels, 1, stride=1, padding=0))

    def execute(self, x, context=None, which=None):
        b, c, h, w = x.shape
        x_in = x

        if which == 0:
            x0 = self.norm_0(x)
            x0 = self.proj_in_0(x0)
            x0 = rearrange(x0, 'b c h w -> b (h w) c')
            for block in self.transformer_blocks_0:
                x0 = block(x0, context=context)
            x0 = rearrange(x0, 'b (h w) c -> b c h w', h=h, w=w)
            x0 = self.proj_out_0(x0)
            return x0 + x_in

        if which == 1:
            x1 = self.norm_1(x)
            x1 = self.proj_in_1(x1)
            x1 = rearrange(x1, 'b c h w -> b (h w) c')
            for block in self.transformer_blocks_1:
                x1 = block(x1, context=context)
            x1 = rearrange(x1, 'b (h w) c -> b c h w', h=h, w=w)
            x1 = self.proj_out_1(x1)
            return x1 + x_in

        # mixture: context is [context0, context1]
        x0 = self.norm_0(x)
        x0 = self.proj_in_0(x0)
        x0 = rearrange(x0, 'b c h w -> b (h w) c')
        for block in self.transformer_blocks_0:
            x0 = block(x0, context=context[0])
        x0 = rearrange(x0, 'b (h w) c -> b c h w', h=h, w=w)
        x0 = self.proj_out_0(x0)

        x1 = self.norm_1(x)
        x1 = self.proj_in_1(x1)
        x1 = rearrange(x1, 'b c h w -> b (h w) c')
        for block in self.transformer_blocks_1:
            x1 = block(x1, context=context[1])
        x1 = rearrange(x1, 'b (h w) c -> b c h w', h=h, w=w)
        x1 = self.proj_out_1(x1)

        return x0 * which + x1 * (1 - which) + x_in
