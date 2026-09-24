# Modified for LayerVerse from UniEdit-Flow (https://github.com/DSL-Lab/UniEdit-Flow, Apache-2.0):
# attention that also returns the image-to-text cross-attention map.
import torch
from einops import rearrange
from torch import Tensor
import math

def attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor) -> Tensor:
    q, k = apply_rope(q, k, pe)

    x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    x = rearrange(x, "B H L D -> B L (H D)")

    return x

def adaptive_attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor, txt_shape: int, img_shape: int) -> Tensor:
    q, k = apply_rope(q, k, pe)

    x, txt_img_cross = scaled_dot_product_attention(q, k, v, txt_shape, img_shape)
    x = rearrange(x, "B H L D -> B L (H D)")

    return x, txt_img_cross

def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)

# Written out rather than using scaled_dot_product_attention because the
# image-to-text block of the attention matrix is needed to build the editing mask.
def scaled_dot_product_attention(query, key, value, txt_shape, img_shape) -> Tensor:
    scale_factor = 1 / math.sqrt(query.size(-1))

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight = torch.softmax(attn_weight, dim=-1)

    # (1,24,4608,4608) -> (1,24,4096,512)
    txt_img_cross = attn_weight[:, :, -img_shape:, :txt_shape]
    txt_img_cross = txt_img_cross.mean(dim=1).detach()

    return attn_weight @ value, txt_img_cross