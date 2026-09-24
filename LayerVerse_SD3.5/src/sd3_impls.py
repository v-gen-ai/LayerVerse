# Modified for LayerVerse from https://github.com/Stability-AI/sd3.5 (MIT License):
# adds the automatic editing mask and the Euler / RF-Solver / FireFlow inversion and
# KV-injection samplers, passes `info` through BaseModel.apply_model, and runs the VAE
# in its own dtype instead of under autocast.
### Impls of the SD3 core diffusion model and VAE

import math
import re
from copy import deepcopy

import cv2
import einops
from safetensors import safe_open
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage.morphology import closing, remove_small_holes
from tqdm import tqdm
import numpy as np
import torch.nn as nn

from dit_embedder import ControlNetEmbedder
from mmditx import MMDiTX
from typing import Tuple

#################################################################################################
### MMDiT Model Wrapping
#################################################################################################


class ModelSamplingDiscreteFlow(torch.nn.Module):
    """Helper for sampler scheduling (ie timestep/sigma calculations) for Discrete Flow models"""

    def __init__(self, shift=1.0):
        super().__init__()
        self.shift = shift
        timesteps = 1000
        ts = self.sigma(torch.arange(1, timesteps + 1, 1))
        self.register_buffer("sigmas", ts)

    @property
    def sigma_min(self):
        return self.sigmas[0]

    @property
    def sigma_max(self):
        return self.sigmas[-1]

    def timestep(self, sigma):
        return sigma * 1000

    def sigma(self, timestep: torch.Tensor):
        timestep = timestep / 1000.0
        if self.shift == 1.0:
            return timestep
        return self.shift * timestep / (1 + (self.shift - 1) * timestep)

    def calculate_denoised(self, sigma, model_output, model_input):
        sigma = sigma.view(sigma.shape[:1] + (1,) * (model_output.ndim - 1))
        return model_input - model_output * sigma

    def noise_scaling(self, sigma, noise, latent_image, max_denoise=False):
        return sigma * noise + (1.0 - sigma) * latent_image


class BaseModel(torch.nn.Module):
    """Wrapper around the core MM-DiT model"""

    def __init__(
        self,
        shift=1.0,
        device=None,
        dtype=torch.float32,
        file=None,
        prefix="",
        control_model_ckpt=None,
        verbose=False,
    ):
        super().__init__()
        # Important configuration values can be quickly determined by checking shapes in the source file
        # Some of these will vary between models (eg 2B vs 8B primarily differ in their depth, but also other details change)
        patch_size = file.get_tensor(f"{prefix}x_embedder.proj.weight").shape[2]
        depth = file.get_tensor(f"{prefix}x_embedder.proj.weight").shape[0] // 64
        num_patches = file.get_tensor(f"{prefix}pos_embed").shape[1]
        pos_embed_max_size = round(math.sqrt(num_patches))
        adm_in_channels = file.get_tensor(f"{prefix}y_embedder.mlp.0.weight").shape[1]
        context_shape = file.get_tensor(f"{prefix}context_embedder.weight").shape

        qk_norm = (
            "rms"
            if f"{prefix}joint_blocks.0.context_block.attn.ln_k.weight" in file.keys()
            else None
        )
        x_block_self_attn_layers = sorted(
            [
                int(key.split(".x_block.attn2.ln_k.weight")[0].split(".")[-1])
                for key in list(
                    filter(
                        re.compile(".*.x_block.attn2.ln_k.weight").match, file.keys()
                    )
                )
            ]
        )

        context_embedder_config = {
            "target": "torch.nn.Linear",
            "params": {
                "in_features": context_shape[1],
                "out_features": context_shape[0],
            },
        }
        self.diffusion_model = MMDiTX(
            input_size=None,
            pos_embed_scaling_factor=None,
            pos_embed_offset=None,
            pos_embed_max_size=pos_embed_max_size,
            patch_size=patch_size,
            in_channels=16,
            depth=depth,
            num_patches=num_patches,
            adm_in_channels=adm_in_channels,
            context_embedder_config=context_embedder_config,
            qk_norm=qk_norm,
            x_block_self_attn_layers=x_block_self_attn_layers,
            device=device,
            dtype=dtype,
            verbose=verbose,
        )
        self.model_sampling = ModelSamplingDiscreteFlow(shift=shift)
        self.control_model = None
        if control_model_ckpt is not None:
            n_controlnet_layers = len(
                list(
                    filter(
                        re.compile(".*.attn.proj.weight").match,
                        control_model_ckpt.keys(),
                    )
                )
            )

            hidden_size = 64 * depth
            num_heads = depth
            head_dim = hidden_size // num_heads
            pooled_projection_size = control_model_ckpt.get_tensor('time_text_embed.text_embedder.linear_1.weight').shape[1]
            if verbose:
                print(
                    f"Initializing ControlNetEmbedder with {n_controlnet_layers} layers, y_in of {pooled_projection_size}"
                )
            self.control_model = ControlNetEmbedder(
                img_size=None,
                patch_size=patch_size,
                in_chans=16,
                num_layers=n_controlnet_layers,
                attention_head_dim=head_dim,
                num_attention_heads=num_heads,
                pooled_projection_size=pooled_projection_size,
                device=device,
                dtype=dtype,
            )

    def apply_model(self, x, sigma, c_crossattn=None, y=None, skip_layers=[], controlnet_cond=None, info=None):
        dtype = self.get_dtype()
        timestep = self.model_sampling.timestep(sigma).float()
        controlnet_hidden_states = None
        if controlnet_cond is not None:
            y_cond = y.to(dtype)
            controlnet_cond = controlnet_cond.to(dtype=x.dtype, device=x.device)
            controlnet_cond = controlnet_cond.repeat(x.shape[0], 1, 1, 1)

            if not self.control_model.using_8b_controlnet:
                y_cond = self.diffusion_model.y_embedder(y)
            
            x_controlnet = x
            if self.control_model.using_8b_controlnet:
                hw = x.shape[-2:]
                x_controlnet = self.diffusion_model.x_embedder(x) + self.diffusion_model.cropped_pos_embed(hw)
            controlnet_hidden_states = self.control_model(
                x_controlnet, controlnet_cond, y_cond, 1, sigma.to(torch.float32)
            )
        model_output = self.diffusion_model(
            x.to(dtype),
            timestep,
            context=c_crossattn.to(dtype),
            y=y.to(dtype),
            controlnet_hidden_states=controlnet_hidden_states,
            skip_layers=skip_layers,
            info=info,
        ).float()
        return self.model_sampling.calculate_denoised(sigma, model_output, x)

    def forward(self, *args, **kwargs):
        return self.apply_model(*args, **kwargs)

    def get_dtype(self):
        return self.diffusion_model.dtype


class CFGDenoiser(torch.nn.Module):
    """Helper for applying CFG Scaling to diffusion outputs"""

    def __init__(self, model, *args):
        super().__init__()
        self.model = model

    def forward(
        self,
        x,
        timestep,
        cond,
        uncond,
        cond_scale,
        **kwargs,
    ):
        # Run cond and uncond in a batch together
        batched = self.model.apply_model(
            torch.cat([x, x]),
            torch.cat([timestep, timestep]),
            c_crossattn=torch.cat([cond["c_crossattn"], uncond["c_crossattn"]]),
            y=torch.cat([cond["y"], uncond["y"]]),
            **kwargs,
        )
        # Then split and apply CFG Scaling
        pos_out, neg_out = batched.chunk(2)
        scaled = neg_out + (pos_out - neg_out) * cond_scale
        return scaled


class SkipLayerCFGDenoiser(torch.nn.Module):
    """Helper for applying CFG Scaling to diffusion outputs"""

    def __init__(self, model, steps, skip_layer_config):
        super().__init__()
        self.model = model
        self.steps = steps
        self.slg = skip_layer_config["scale"]
        self.skip_start = skip_layer_config["start"]
        self.skip_end = skip_layer_config["end"]
        self.skip_layers = skip_layer_config["layers"]
        self.step = 0

    def forward(
        self,
        x,
        timestep,
        cond,
        uncond,
        cond_scale,
        **kwargs,
    ):
        # Run cond and uncond in a batch together
        batched = self.model.apply_model(
            torch.cat([x, x]),
            torch.cat([timestep, timestep]),
            c_crossattn=torch.cat([cond["c_crossattn"], uncond["c_crossattn"]]),
            y=torch.cat([cond["y"], uncond["y"]]),
            **kwargs,
        )
        # Then split and apply CFG Scaling
        pos_out, neg_out = batched.chunk(2)
        scaled = neg_out + (pos_out - neg_out) * cond_scale
        # Then run with skip layer
        if (
            self.slg > 0
            and self.step > (self.skip_start * self.steps)
            and self.step < (self.skip_end * self.steps)
        ):
            skip_layer_out = self.model.apply_model(
                x,
                timestep,
                c_crossattn=cond["c_crossattn"],
                y=cond["y"],
                skip_layers=self.skip_layers,
            )
            # Then scale acc to skip layer guidance
            scaled = scaled + (pos_out - skip_layer_out) * self.slg
        self.step += 1
        return scaled


class SD3LatentFormat:
    """Latents are slightly shifted from center - this class must be called after VAE Decode to correct for the shift"""

    def __init__(self):
        self.scale_factor = 1.5305
        self.shift_factor = 0.0609

    def process_in(self, latent):
        return (latent - self.shift_factor) * self.scale_factor

    def process_out(self, latent):
        return (latent / self.scale_factor) + self.shift_factor

    def decode_latent_to_preview(self, x0):
        """Quick RGB approximate preview of sd3 latents"""
        factors = torch.tensor(
            [
                [-0.0645, 0.0177, 0.1052],
                [0.0028, 0.0312, 0.0650],
                [0.1848, 0.0762, 0.0360],
                [0.0944, 0.0360, 0.0889],
                [0.0897, 0.0506, -0.0364],
                [-0.0020, 0.1203, 0.0284],
                [0.0855, 0.0118, 0.0283],
                [-0.0539, 0.0658, 0.1047],
                [-0.0057, 0.0116, 0.0700],
                [-0.0412, 0.0281, -0.0039],
                [0.1106, 0.1171, 0.1220],
                [-0.0248, 0.0682, -0.0481],
                [0.0815, 0.0846, 0.1207],
                [-0.0120, -0.0055, -0.0867],
                [-0.0749, -0.0634, -0.0456],
                [-0.1418, -0.1457, -0.1259],
            ],
            device="cpu",
        )
        latent_image = x0[0].permute(1, 2, 0).cpu() @ factors

        latents_ubyte = (
            ((latent_image + 1) / 2)
            .clamp(0, 1)  # change scale from -1..1 to 0..1
            .mul(0xFF)  # to 0..255
            .byte()
        ).cpu()

        return Image.fromarray(latents_ubyte.numpy())


#################################################################################################
### Samplers
#################################################################################################


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    return x[(...,) + (None,) * dims_to_append]


def to_d(x, sigma, denoised):
    """Converts a denoiser output to a Karras ODE derivative."""
    return (x - denoised) / append_dims(sigma, x.ndim)


def _set_info_for_step(info, sigma_value: float, inverse: bool):
    """Mutates `info` in-place for KV-caching logic."""
    if info is None:
        return
    info["inverse"] = bool(inverse)
    info["timestep"] = float(sigma_value)


def set_edit_map_indices(info, shared_mask_unpacked):
    """Convert mask to edit_map_indices and store in info for KV injection."""
    if shared_mask_unpacked is None:
        return
    edit_map_flat = shared_mask_unpacked.flatten()
    info["edit_map_indices"] = (edit_map_flat > 0).nonzero(as_tuple=False).squeeze(1)


# ---------------------------------------------------------------------------
# Single-step automatic editing mask (Algorithm 1 of the LayerVerse paper)
# ---------------------------------------------------------------------------

def convert_attn_map_to_image(attn_map, target_size, sigmoid_gain=3.0, smoothing_sigma=0.72,
                              closing_size=3, hole_area=136):
    """Binarise an averaged cross-attention map into an editing mask.

    ``target_size`` is the latent size; the map lives on the patch grid
    ``(latent_h // 2, latent_w // 2)``. The map is min-max normalised, sharpened
    with a sigmoid, Gaussian-smoothed, Otsu-thresholded, closed and hole-filled.
    """
    H, W = target_size[0] // 2, target_size[1] // 2
    image = attn_map.reshape(H, W)
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)
    image = torch.sigmoid((image - 0.5) * sigmoid_gain)

    smoothed = gaussian_filter(image.detach().float().cpu().numpy(), sigma=smoothing_sigma)
    _, binary = cv2.threshold((smoothed * 255).astype(np.uint8), 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    mask = closing(binary > 0, np.ones((closing_size, closing_size), dtype=np.uint8))
    mask = remove_small_holes(mask, area_threshold=hole_area, connectivity=1)
    return torch.from_numpy(mask.astype(np.float32)).to(attn_map.device, dtype=attn_map.dtype).reshape(1, H * W, 1)


@torch.no_grad()
def compute_attention_mask(sd3_model, noise_latent, source_latent, sigmas, mask_sigma_index, selected_cond,
                           clip_token_indices, t5_token_indices, info, mask_blocks_indices, latent_size):
    """Editing mask from one forward pass at ``sigmas[mask_sigma_index]`` (no CFG).

    The source latent is interpolated towards its inverted noise, and the
    image-to-text cross-attention of ``mask_blocks_indices`` is averaged over the
    changed tokens. CLIP and T5 tokenise differently, so each encoder's changed
    tokens are read from its own slice (CLIP indices already include the BOS
    offset). Returns an ``(H, W)`` mask on the patch grid.
    """
    H, W = latent_size[0] // 2, latent_size[1] // 2
    empty = torch.zeros(H, W, device=noise_latent.device, dtype=noise_latent.dtype)
    if len(clip_token_indices) == 0 and len(t5_token_indices) == 0:
        return empty

    sigma = sigmas[mask_sigma_index]
    noisy = sigma * noise_latent + (1.0 - sigma) * source_latent

    mask_info = {k: deepcopy(v) for k, v in info.items() if k != "features"}
    mask_info.update(use_kv_injection=False, extract_attn_maps=True, extract_blocks=mask_blocks_indices,
                     clip_attn_maps={}, t5_attn_maps={}, inverse=False, timestep=float(sigma.item()))

    sd3_model.apply_model(noisy, sigma.unsqueeze(0), c_crossattn=selected_cond["c_crossattn"][:1],
                          y=selected_cond["y"][:1], info=mask_info)

    attn_maps = []
    for bid in mask_blocks_indices:
        parts = []
        if clip_token_indices:
            parts.append(mask_info["clip_attn_maps"][bid][..., clip_token_indices])
        if t5_token_indices:
            parts.append(mask_info["t5_attn_maps"][bid][..., t5_token_indices])
        attn_maps.append(torch.cat(parts, dim=-1))

    attn_avg = torch.cat(attn_maps, dim=0).mean(dim=[0, -1], keepdim=True)
    return convert_attn_map_to_image(attn_avg, latent_size).to(noise_latent.device).reshape(H, W)


@torch.no_grad()
@torch.autocast("cuda", dtype=torch.float16)
def sample_euler(model, x, sigmas, extra_args=None):
    """Implements Algorithm 2 (Euler steps) from Karras et al. (2022)."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in tqdm(range(len(sigmas) - 1)):
        sigma_hat = sigmas[i]
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, denoised)
        dt = sigmas[i + 1] - sigma_hat
        # Euler method
        x = x + d * dt
    return x


@torch.no_grad()
def sample_euler_inverse(model, x, sigmas, extra_args=None, info=None):
    """Euler inversion: image -> noise. Skips sigma=0 to avoid div by zero."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    sigmas_inv = sigmas.flip(0)[1:]
    if info is not None:
        extra_args["info"] = info
    for i in range(len(sigmas_inv) - 1):
        sigma_hat = sigmas_inv[i]
        _set_info_for_step(info, sigma_hat.item(), inverse=True)
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, denoised)
        dt = sigmas_inv[i + 1] - sigma_hat
        x = x + d * dt
    return x


@torch.no_grad()
def sample_euler_kv(model, x, sigmas, extra_args=None, info=None):
    """Euler sampling with KV-injection."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    if info is not None:
        extra_args["info"] = info
    for i in range(len(sigmas) - 1):
        sigma_hat = sigmas[i]
        _set_info_for_step(info, sigma_hat.item(), inverse=False)
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, denoised)
        dt = sigmas[i + 1] - sigma_hat
        x = x + d * dt
    return x


@torch.no_grad()
def sample_rf_solver_inverse(model, x, sigmas, extra_args=None, info=None):
    """RF-Solver inversion (image -> noise) with second-order Taylor correction."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    sigmas_inv = sigmas.flip(0)[1:]
    if info is not None:
        extra_args["info"] = info
    for i in range(len(sigmas_inv) - 1):
        sigma_hat = sigmas_inv[i]
        dt = sigmas_inv[i + 1] - sigma_hat

        _set_info_for_step(info, sigma_hat.item(), inverse=True)
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, denoised)

        sigma_mid = sigma_hat + dt / 2
        x_mid = x + dt / 2 * d
        _set_info_for_step(info, sigma_mid.item(), inverse=True)
        denoised_mid = model(x_mid, sigma_mid * s_in, **extra_args)
        d_mid = to_d(x_mid, sigma_mid, denoised_mid)

        first_order = (d_mid - d) / (dt / 2)
        x = x + dt * d + 0.5 * dt ** 2 * first_order
    return x


@torch.no_grad()
def sample_rf_solver_kv(model, x, sigmas, extra_args=None, info=None):
    """RF-Solver sampling (noise -> image) with KV-injection and second-order Taylor correction."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    if info is not None:
        extra_args["info"] = info
    for i in range(len(sigmas) - 1):
        sigma_hat = sigmas[i]
        dt = sigmas[i + 1] - sigma_hat

        _set_info_for_step(info, sigma_hat.item(), inverse=False)
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x, sigma_hat, denoised)

        sigma_mid = sigma_hat + dt / 2
        x_mid = x + dt / 2 * d
        _set_info_for_step(info, sigma_mid.item(), inverse=False)
        denoised_mid = model(x_mid, sigma_mid * s_in, **extra_args)
        d_mid = to_d(x_mid, sigma_mid, denoised_mid)

        first_order = (d_mid - d) / (dt / 2)
        x = x + dt * d + 0.5 * dt ** 2 * first_order
    return x


@torch.no_grad()
def sample_fireflow_inverse(model, x, sigmas, extra_args=None, info=None):
    """FireFlow inversion (image -> noise): midpoint method with velocity reuse."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    sigmas_inv = sigmas.flip(0)[1:]
    if info is not None:
        extra_args["info"] = info
    next_d = None
    for i in range(len(sigmas_inv) - 1):
        sigma_hat = sigmas_inv[i]
        dt = sigmas_inv[i + 1] - sigma_hat

        if next_d is None:
            _set_info_for_step(info, sigma_hat.item(), inverse=True)
            denoised = model(x, sigma_hat * s_in, **extra_args)
            d = to_d(x, sigma_hat, denoised)
        else:
            d = next_d

        sigma_mid = sigma_hat + dt / 2
        x_mid = x + dt / 2 * d
        _set_info_for_step(info, sigma_mid.item(), inverse=True)
        denoised_mid = model(x_mid, sigma_mid * s_in, **extra_args)
        d_mid = to_d(x_mid, sigma_mid, denoised_mid)

        next_d = d_mid
        x = x + dt * d_mid
    return x


@torch.no_grad()
def sample_fireflow_kv(model, x, sigmas, extra_args=None, info=None):
    """FireFlow sampling (noise -> image) with KV-injection: midpoint method with velocity reuse."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    if info is not None:
        extra_args["info"] = info
    next_d = None
    for i in range(len(sigmas) - 1):
        sigma_hat = sigmas[i]
        dt = sigmas[i + 1] - sigma_hat

        if next_d is None:
            _set_info_for_step(info, sigma_hat.item(), inverse=False)
            denoised = model(x, sigma_hat * s_in, **extra_args)
            d = to_d(x, sigma_hat, denoised)
        else:
            d = next_d

        sigma_mid = sigma_hat + dt / 2
        x_mid = x + dt / 2 * d
        _set_info_for_step(info, sigma_mid.item(), inverse=False)
        denoised_mid = model(x_mid, sigma_mid * s_in, **extra_args)
        d_mid = to_d(x_mid, sigma_mid, denoised_mid)

        next_d = d_mid
        x = x + dt * d_mid
    return x


@torch.no_grad()
@torch.autocast("cuda", dtype=torch.float16)
def sample_dpmpp_2m(model, x, sigmas, extra_args=None):
    """DPM-Solver++(2M)."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()
    old_denoised = None
    for i in tqdm(range(len(sigmas) - 1)):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
        h = t_next - t
        if old_denoised is None or sigmas[i + 1] == 0:
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised
        else:
            h_last = t - t_fn(sigmas[i - 1])
            r = h_last / h
            denoised_d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old_denoised
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised_d
        old_denoised = denoised
    return x


#################################################################################################
### VAE
#################################################################################################


def Normalize(in_channels, num_groups=32, dtype=torch.float32, device=None):
    return torch.nn.GroupNorm(
        num_groups=num_groups,
        num_channels=in_channels,
        eps=1e-6,
        affine=True,
        dtype=dtype,
        device=device,
    )


class ResnetBlock(torch.nn.Module):
    def __init__(
        self, *, in_channels, out_channels=None, dtype=torch.float32, device=None
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = Normalize(in_channels, dtype=dtype, device=device)
        self.conv1 = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            dtype=dtype,
            device=device,
        )
        self.norm2 = Normalize(out_channels, dtype=dtype, device=device)
        self.conv2 = torch.nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            dtype=dtype,
            device=device,
        )
        if self.in_channels != self.out_channels:
            self.nin_shortcut = torch.nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                dtype=dtype,
                device=device,
            )
        else:
            self.nin_shortcut = None
        self.swish = torch.nn.SiLU(inplace=True)

    def forward(self, x):
        hidden = x
        hidden = self.norm1(hidden)
        hidden = self.swish(hidden)
        hidden = self.conv1(hidden)
        hidden = self.norm2(hidden)
        hidden = self.swish(hidden)
        hidden = self.conv2(hidden)
        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        return x + hidden


class AttnBlock(torch.nn.Module):
    def __init__(self, in_channels, dtype=torch.float32, device=None):
        super().__init__()
        self.norm = Normalize(in_channels, dtype=dtype, device=device)
        self.q = torch.nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dtype=dtype,
            device=device,
        )
        self.k = torch.nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dtype=dtype,
            device=device,
        )
        self.v = torch.nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dtype=dtype,
            device=device,
        )
        self.proj_out = torch.nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dtype=dtype,
            device=device,
        )

    def forward(self, x):
        hidden = self.norm(x)
        q = self.q(hidden)
        k = self.k(hidden)
        v = self.v(hidden)
        b, c, h, w = q.shape
        q, k, v = map(
            lambda x: einops.rearrange(x, "b c h w -> b 1 (h w) c").contiguous(),
            (q, k, v),
        )
        hidden = torch.nn.functional.scaled_dot_product_attention(
            q, k, v
        )  # scale is dim ** -0.5 per default
        hidden = einops.rearrange(hidden, "b 1 (h w) c -> b c h w", h=h, w=w, c=c, b=b)
        hidden = self.proj_out(hidden)
        return x + hidden


class Downsample(torch.nn.Module):
    def __init__(self, in_channels, dtype=torch.float32, device=None):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=2,
            padding=0,
            dtype=dtype,
            device=device,
        )

    def forward(self, x):
        pad = (0, 1, 0, 1)
        x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)
        return x


class Upsample(torch.nn.Module):
    def __init__(self, in_channels, dtype=torch.float32, device=None):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            dtype=dtype,
            device=device,
        )

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)
        return x


class VAEEncoder(torch.nn.Module):
    def __init__(
        self,
        ch=128,
        ch_mult=(1, 2, 4, 4),
        num_res_blocks=2,
        in_channels=3,
        z_channels=16,
        dtype=torch.float32,
        device=None,
    ):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        # downsampling
        self.conv_in = torch.nn.Conv2d(
            in_channels,
            ch,
            kernel_size=3,
            stride=1,
            padding=1,
            dtype=dtype,
            device=device,
        )
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = torch.nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = torch.nn.ModuleList()
            attn = torch.nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(num_res_blocks):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        dtype=dtype,
                        device=device,
                    )
                )
                block_in = block_out
            down = torch.nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, dtype=dtype, device=device)
            self.down.append(down)
        # middle
        self.mid = torch.nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in, out_channels=block_in, dtype=dtype, device=device
        )
        self.mid.attn_1 = AttnBlock(block_in, dtype=dtype, device=device)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in, out_channels=block_in, dtype=dtype, device=device
        )
        # end
        self.norm_out = Normalize(block_in, dtype=dtype, device=device)
        self.conv_out = torch.nn.Conv2d(
            block_in,
            2 * z_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            dtype=dtype,
            device=device,
        )
        self.swish = torch.nn.SiLU(inplace=True)

    def forward(self, x):
        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))
        # middle
        h = hs[-1]
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        # end
        h = self.norm_out(h)
        h = self.swish(h)
        h = self.conv_out(h)
        return h


class VAEDecoder(torch.nn.Module):
    def __init__(
        self,
        ch=128,
        out_ch=3,
        ch_mult=(1, 2, 4, 4),
        num_res_blocks=2,
        resolution=256,
        z_channels=16,
        dtype=torch.float32,
        device=None,
    ):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        # z to block_in
        self.conv_in = torch.nn.Conv2d(
            z_channels,
            block_in,
            kernel_size=3,
            stride=1,
            padding=1,
            dtype=dtype,
            device=device,
        )
        # middle
        self.mid = torch.nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in, out_channels=block_in, dtype=dtype, device=device
        )
        self.mid.attn_1 = AttnBlock(block_in, dtype=dtype, device=device)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in, out_channels=block_in, dtype=dtype, device=device
        )
        # upsampling
        self.up = torch.nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = torch.nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        dtype=dtype,
                        device=device,
                    )
                )
                block_in = block_out
            up = torch.nn.Module()
            up.block = block
            if i_level != 0:
                up.upsample = Upsample(block_in, dtype=dtype, device=device)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order
        # end
        self.norm_out = Normalize(block_in, dtype=dtype, device=device)
        self.conv_out = torch.nn.Conv2d(
            block_in,
            out_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            dtype=dtype,
            device=device,
        )
        self.swish = torch.nn.SiLU(inplace=True)

    def forward(self, z):
        # z to block_in
        hidden = self.conv_in(z)
        # middle
        hidden = self.mid.block_1(hidden)
        hidden = self.mid.attn_1(hidden)
        hidden = self.mid.block_2(hidden)
        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                hidden = self.up[i_level].block[i_block](hidden)
            if i_level != 0:
                hidden = self.up[i_level].upsample(hidden)
        # end
        hidden = self.norm_out(hidden)
        hidden = self.swish(hidden)
        hidden = self.conv_out(hidden)
        return hidden


class SDVAE(torch.nn.Module):
    def __init__(self, dtype=torch.float32, device=None):
        super().__init__()
        self.encoder = VAEEncoder(dtype=dtype, device=device)
        self.decoder = VAEDecoder(dtype=dtype, device=device)

    def decode(self, latent):
        return self.decoder(latent.to(self.decoder.conv_in.weight.dtype))

    def encode(self, image):
        hidden = self.encoder(image.to(self.encoder.conv_in.weight.dtype))
        mean, logvar = torch.chunk(hidden, 2, dim=1)
        logvar = torch.clamp(logvar, -30.0, 20.0)
        std = torch.exp(0.5 * logvar)
        return mean + std * torch.randn_like(mean)

