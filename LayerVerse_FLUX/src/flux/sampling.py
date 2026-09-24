# Modified for LayerVerse from UniEdit-Flow (https://github.com/DSL-Lab/UniEdit-Flow, Apache-2.0):
# single-step automatic editing mask and a masked Euler solver; unused samplers removed.
import math
from typing import Callable
from copy import deepcopy

import torch
from einops import rearrange, repeat
from torch import Tensor

from scipy.ndimage import gaussian_filter
import cv2
from skimage.morphology import closing, remove_small_holes
import numpy as np
from .model import Flux
from .modules.conditioner import HFEmbedder


def prepare(t5: HFEmbedder, clip: HFEmbedder, img: Tensor, prompt: str | list[str]) -> dict[str, Tensor]:
    bs, c, h, w = img.shape
    if bs == 1 and not isinstance(prompt, str):
        bs = len(prompt)

    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    if img.shape[0] == 1 and bs > 1:
        img = repeat(img, "1 ... -> bs ...", bs=bs)

    img_ids = torch.zeros(h // 2, w // 2, 3)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

    if isinstance(prompt, str):
        prompt = [prompt]
    txt = t5(prompt)
    if txt.shape[0] == 1 and bs > 1:
        txt = repeat(txt, "1 ... -> bs ...", bs=bs)
    txt_ids = torch.zeros(bs, txt.shape[1], 3)

    vec = clip(prompt)
    if vec.shape[0] == 1 and bs > 1:
        vec = repeat(vec, "1 ... -> bs ...", bs=bs)

    return {
        "img": img,
        "img_ids": img_ids.to(img.device),
        "txt": txt.to(img.device),
        "txt_ids": txt_ids.to(img.device),
        "vec": vec.to(img.device),
    }

def create_editing_mask(info, img, masking_timesteps, selected_txt, selected_txt_ids, selected_vec, curr_token_indices,
                       src_img, init_noise, img_ids, guidance_vec, model, inject_list):
    """

    Create editing mask from the averaged cross-attention maps

    """

    orig_h, orig_w = info['orig_latent_size']

    initial_noise = img.detach().clone()

    all_attn_maps = {f"timestep_{info['mask_setting'][ind]['timestep']}": None
                     for ind in range(len(info["mask_setting"]))}

    assert (initial_noise == init_noise).all()

    for mask_timestep_info in info["mask_setting"]:
        cur_mask_t = masking_timesteps[:-1][mask_timestep_info['timestep']]

        t_vec = torch.full((img.shape[0],), cur_mask_t, dtype=img.dtype, device=img.device)
        info['t'] = cur_mask_t
        info['inverse'] = False
        info['second_order'] = False
        info['inject'] = inject_list[mask_timestep_info['timestep']]
        info['timestep_id'] = mask_timestep_info['timestep']

        curr_img = cur_mask_t * initial_noise + (1 - cur_mask_t) * src_img

        _, _, attn_maps_double_blocks, attn_maps_single_blocks = model(
            img=curr_img,
            img_ids=img_ids,
            txt=selected_txt,
            txt_ids=selected_txt_ids,
            y=selected_vec,
            timesteps=t_vec,
            guidance=guidance_vec,
            info=info,
            cur_step=mask_timestep_info['timestep']
        )

        all_attn_maps[f"timestep_{mask_timestep_info['timestep']}"] = attn_maps_double_blocks + attn_maps_single_blocks

    assert len(all_attn_maps) == len(info["mask_setting"])

    attn_maps = [all_attn_maps[f"timestep_{mask_t_info['timestep']}"][layer_id]['tensor'][..., curr_token_indices]
                    for mask_t_info in info["mask_setting"]
                    for layer_id in mask_t_info["blocks_indices"]]

    attn_maps_avg = torch.cat(attn_maps, dim=0).mean(dim=[0, -1], keepdim=True)

    shared_mask, attn_check = convert_attn_map_to_image(attn_maps_avg, info['orig_latent_size'],
                                                        use_sigmoid=True, d=3.0, use_gaussian_filter=True, 
                                                        binarize=True, sigma=0.72, threshold=-0.1, closing_size=3, hole_area=136)
    shared_mask = shared_mask.to(img.device)
    assert (attn_check == attn_maps_avg).all()

    return shared_mask.reshape(orig_h // 2, orig_w // 2)


def convert_attn_map_to_image(attn_map: Tensor, target_size: tuple[int, int], use_sigmoid=False, d=3.0,
                              use_gaussian_filter=False, sigma=0.72, binarize=False, threshold=-0.1,
                              closing_size: int = 3,
                              hole_area: int = 136
                              ) -> Tensor:
    H = target_size[0] // 2
    W = target_size[1] // 2
    image = attn_map.reshape(H, W)

    # min max norm
    min_val = image.min()
    max_val = image.max()
    normalized_image = (image - min_val) / (max_val - min_val)

    if use_sigmoid:
        normalized_image = torch.sigmoid((normalized_image - 0.5) * d)

    if use_gaussian_filter:
        normalized_image_np = normalized_image.detach().to(dtype=torch.float32, device='cpu').numpy()
        normalized_image_np = gaussian_filter(normalized_image_np, sigma=sigma)
        normalized_image = torch.from_numpy(normalized_image_np).to(normalized_image.device, dtype=normalized_image.dtype)

    if binarize:
        if threshold < 0:
            img_uint8 = (normalized_image_np * 255).astype(np.uint8)
            _, binary_mask_np = cv2.threshold(img_uint8, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            binary_mask_np = (normalized_image.detach().cpu().numpy() > threshold).astype(np.uint8)

        m = binary_mask_np > 0
        if closing_size > 0:
                m = closing(m, np.ones((closing_size, closing_size), dtype=np.uint8))
        if hole_area > 0:
                m = remove_small_holes(m, area_threshold=hole_area, connectivity=1)
        final_mask = m
        
        normalized_image = torch.from_numpy(final_mask.astype(np.float32)).to(attn_map.device, dtype=attn_map.dtype)

    return normalized_image.reshape(1, H * W, 1), image.reshape(1, H * W, 1)

def time_shift(mu: float, sigma: float, t: Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(
    x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> list[float]:
    # extra step for zero
    timesteps = torch.linspace(1, 0, num_steps + 1)

    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        # estimate mu based on linear estimation between two points
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)

    return timesteps.tolist()


def denoise_rf_solver(
    model: Flux,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    # sampling parameters
    timesteps: list[float],
    inverse,
    info, 
    guidance: float = 4.0,
    src_img: Tensor | None = None,
    init_noise: Tensor | None = None,
    src_txt=None,
    src_txt_ids=None,
    src_vec=None,
):
    inject_list = [True] * info['inject_step'] + [False] * (len(timesteps[:-1]) - info['inject_step'])
    
    if info['use_mask']:
        masking_timesteps = info['masking_timesteps']
    else:
        masking_timesteps = deepcopy(timesteps)
    inject_list_masking = [True] * info['inject_step'] + [False] * (len(masking_timesteps[:-1]) - info['inject_step'])

    if inverse:
        timesteps = timesteps[::-1]
        inject_list = inject_list[::-1]
        masking_timesteps = masking_timesteps[::-1]
        inject_list_masking = inject_list_masking[::-1]
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)

    #==========================================================================
    if info['use_mask'] and not inverse:

        orig_h, orig_w = info['orig_latent_size']

        tgt_shared_mask_unpacked = torch.zeros((orig_h // 2, orig_w // 2), dtype=img.dtype, device=img.device)
        src_shared_mask_unpacked = torch.zeros((orig_h // 2, orig_w // 2), dtype=img.dtype, device=img.device)

        if len(info["new_token_indices"]) > 0:
            selected_txt = txt.clone().detach()
            selected_txt_ids = txt_ids.clone().detach()
            selected_vec = vec.clone().detach()

            curr_token_indices = info['new_token_indices']
            duplicate_info = {key: deepcopy(info[key]) for key in info.keys() if key != 'feature'}

            tgt_shared_mask_unpacked = create_editing_mask(duplicate_info, img, masking_timesteps,
                                                                            selected_txt, selected_txt_ids, selected_vec,
                                                                            curr_token_indices,
                                                                            src_img, init_noise, img_ids,
                                                                            guidance_vec, model, inject_list_masking)

        if len(info["deleted_token_indices"]) > 0:
            selected_txt = src_txt.clone().detach()
            selected_txt_ids = src_txt_ids.clone().detach()
            selected_vec = src_vec.clone().detach()

            curr_token_indices = info['deleted_token_indices']
            duplicate_info = {key: deepcopy(info[key]) for key in info.keys() if key != 'feature'}
    
            src_shared_mask_unpacked = create_editing_mask(duplicate_info, img, masking_timesteps,
                                                                            selected_txt, selected_txt_ids, selected_vec,
                                                                            curr_token_indices,
                                                                            src_img, init_noise, img_ids,
                                                                            guidance_vec, model, inject_list_masking)
        
        
        shared_mask_unpacked = torch.logical_or(tgt_shared_mask_unpacked, src_shared_mask_unpacked).to(img.device, dtype=img.dtype)

        edit_indices = shared_mask_unpacked
        edit_map_flat = edit_indices.flatten()  # [N_patch]
        edit_map_indices = (edit_map_flat > 0).nonzero(as_tuple=False).squeeze(1)  # [N_foreground]

        info['total_attn_map'] = shared_mask_unpacked
        info['edit_map_indices'] = edit_map_indices

        if 'timestep_id' in info:
            del info['timestep_id']
        assert 'timestep_id' not in info 

    #==========================================================================

    for i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        info['t'] = t_prev if inverse else t_curr
        info['inverse'] = inverse
        info['second_order'] = False
        info['inject'] = inject_list[i]

        pred, info, _, _ = model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
            info=info,
            cur_step=i
        )

        img_mid = img + (t_prev - t_curr) / 2 * pred

        t_vec_mid = torch.full((img.shape[0],), (t_curr + (t_prev - t_curr) / 2), dtype=img.dtype, device=img.device)
        info['second_order'] = True
        pred_mid, info, _, _ = model(
            img=img_mid,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec_mid,
            guidance=guidance_vec,
            info=info,
            cur_step=i
        )

        first_order = (pred_mid - pred) / ((t_prev - t_curr) / 2)
        img = img + (t_prev - t_curr) * pred + 0.5 * (t_prev - t_curr) ** 2 * first_order

    return img, info


def denoise_fireflow(
    model: Flux,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    # sampling parameters
    timesteps: list[float],
    inverse,
    info, 
    guidance: float = 4.0,
    src_img: Tensor | None = None,
    init_noise: Tensor | None = None,
    src_txt=None,
    src_txt_ids=None,
    src_vec=None,
):
    inject_list = [True] * info['inject_step'] + [False] * (len(timesteps[:-1]) - info['inject_step'])
    
    if info['use_mask']:
        masking_timesteps = info['masking_timesteps']
    else:
        masking_timesteps = deepcopy(timesteps)
    inject_list_masking = [True] * info['inject_step'] + [False] * (len(masking_timesteps[:-1]) - info['inject_step'])

    if inverse:
        timesteps = timesteps[::-1]
        inject_list = inject_list[::-1]
        masking_timesteps = masking_timesteps[::-1]
        inject_list_masking = inject_list_masking[::-1]
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)

    #==========================================================================
    if info['use_mask'] and not inverse:

        orig_h, orig_w = info['orig_latent_size']

        tgt_shared_mask_unpacked = torch.zeros((orig_h // 2, orig_w // 2), dtype=img.dtype, device=img.device)
        src_shared_mask_unpacked = torch.zeros((orig_h // 2, orig_w // 2), dtype=img.dtype, device=img.device)

        if len(info["new_token_indices"]) > 0:
            selected_txt = txt.clone().detach()
            selected_txt_ids = txt_ids.clone().detach()
            selected_vec = vec.clone().detach()

            curr_token_indices = info['new_token_indices']
            duplicate_info = {key: deepcopy(info[key]) for key in info.keys() if key != 'feature'}

            tgt_shared_mask_unpacked = create_editing_mask(duplicate_info, img, masking_timesteps,
                                                                            selected_txt, selected_txt_ids, selected_vec,
                                                                            curr_token_indices,
                                                                            src_img, init_noise, img_ids,
                                                                            guidance_vec, model, inject_list_masking)

        if len(info["deleted_token_indices"]) > 0:
            selected_txt = src_txt.clone().detach()
            selected_txt_ids = src_txt_ids.clone().detach()
            selected_vec = src_vec.clone().detach()

            curr_token_indices = info['deleted_token_indices']
            duplicate_info = {key: deepcopy(info[key]) for key in info.keys() if key != 'feature'}
    
            src_shared_mask_unpacked = create_editing_mask(duplicate_info, img, masking_timesteps,
                                                                            selected_txt, selected_txt_ids, selected_vec,
                                                                            curr_token_indices,
                                                                            src_img, init_noise, img_ids,
                                                                            guidance_vec, model, inject_list_masking)
        
        
        shared_mask_unpacked = torch.logical_or(tgt_shared_mask_unpacked, src_shared_mask_unpacked).to(img.device, dtype=img.dtype)

        edit_indices = shared_mask_unpacked
        edit_map_flat = edit_indices.flatten()  # [N_patch]
        edit_map_indices = (edit_map_flat > 0).nonzero(as_tuple=False).squeeze(1)  # [N_foreground]

        info['total_attn_map'] = shared_mask_unpacked
        info['edit_map_indices'] = edit_map_indices

        if 'timestep_id' in info:
            del info['timestep_id']
        assert 'timestep_id' not in info 

    #==========================================================================

    next_step_velocity = None
    for i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        info['t'] = t_prev if inverse else t_curr
        info['inverse'] = inverse
        info['second_order'] = False
        info['inject'] = inject_list[i]

        if next_step_velocity is None:
            pred, info, _, _ = model(
                img=img,
                img_ids=img_ids,
                txt=txt,
                txt_ids=txt_ids,
                y=vec,
                timesteps=t_vec,
                guidance=guidance_vec,
                info=info,
                cur_step=i
            )
        else:
            pred = next_step_velocity
        
        img_mid = img + (t_prev - t_curr) / 2 * pred

        t_vec_mid = torch.full((img.shape[0],), t_curr + (t_prev - t_curr) / 2, dtype=img.dtype, device=img.device)
        info['second_order'] = True
        pred_mid, info, _, _ = model(
            img=img_mid,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec_mid,
            guidance=guidance_vec,
            info=info,
            cur_step=i
        )
        next_step_velocity = pred_mid
        
        img = img + (t_prev - t_curr) * pred_mid

    return img, info


def unpack(x: Tensor, height: int, width: int) -> Tensor:
    return rearrange(
        x,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=math.ceil(height / 16),
        w=math.ceil(width / 16),
        ph=2,
        pw=2,
    )


def denoise_with_mask(
    model: Flux,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    # sampling parameters
    timesteps: list[float],
    inverse,
    info, 
    guidance: float = 4.0,
    src_img: Tensor | None = None,
    init_noise: Tensor | None = None,
    src_txt=None,
    src_txt_ids=None,
    src_vec=None,
):

    if info['use_mask']:
        masking_timesteps = info['masking_timesteps']
    else:
        masking_timesteps = deepcopy(timesteps)

    inject_list = [True] * info['inject_step'] + [False] * (len(timesteps[:-1]) - info['inject_step'])
    inject_list_masking = [True] * info['inject_step'] + [False] * (len(masking_timesteps[:-1]) - info['inject_step'])

    if inverse:
        timesteps = timesteps[::-1]
        inject_list = inject_list[::-1]
        masking_timesteps = masking_timesteps[::-1]
        inject_list_masking = inject_list_masking[::-1]
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)

    assert all(inject_list), "denoise_with_mask must use use_KV_injection in all layers!!!"
    #==========================================================================
    if info['use_mask'] and not inverse:

        orig_h, orig_w = info['orig_latent_size']

        tgt_shared_mask_unpacked = torch.zeros((orig_h // 2, orig_w // 2), dtype=img.dtype, device=img.device)
        src_shared_mask_unpacked = torch.zeros((orig_h // 2, orig_w // 2), dtype=img.dtype, device=img.device)

        if len(info["new_token_indices"]) > 0:
            selected_txt = txt.clone().detach()
            selected_txt_ids = txt_ids.clone().detach()
            selected_vec = vec.clone().detach()

            curr_token_indices = info['new_token_indices']
            duplicate_info = {key: deepcopy(info[key]) for key in info.keys() if key != 'feature'}

            tgt_shared_mask_unpacked = create_editing_mask(duplicate_info, img, masking_timesteps,
                                                                            selected_txt, selected_txt_ids, selected_vec,
                                                                            curr_token_indices,
                                                                            src_img, init_noise, img_ids,
                                                                            guidance_vec, model, inject_list_masking)

        if len(info["deleted_token_indices"]) > 0:
            selected_txt = src_txt.clone().detach()
            selected_txt_ids = src_txt_ids.clone().detach()
            selected_vec = src_vec.clone().detach()

            curr_token_indices = info['deleted_token_indices']
            duplicate_info = {key: deepcopy(info[key]) for key in info.keys() if key != 'feature'}
    
            src_shared_mask_unpacked = create_editing_mask(duplicate_info, img, masking_timesteps,
                                                                            selected_txt, selected_txt_ids, selected_vec,
                                                                            curr_token_indices,
                                                                            src_img, init_noise, img_ids,
                                                                            guidance_vec, model, inject_list_masking)
        
        
        shared_mask_unpacked = torch.logical_or(tgt_shared_mask_unpacked, src_shared_mask_unpacked).to(img.device, dtype=img.dtype)

        edit_indices = shared_mask_unpacked
        edit_map_flat = edit_indices.flatten()  # [N_patch]
        edit_map_indices = (edit_map_flat > 0).nonzero(as_tuple=False).squeeze(1)  # [N_foreground]

        info['total_attn_map'] = shared_mask_unpacked
        info['edit_map_indices'] = edit_map_indices

        if 'timestep_id' in info:
            del info['timestep_id']
        assert 'timestep_id' not in info 

    #==========================================================================

    for i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        info['t'] = t_prev if inverse else t_curr
        info['inverse'] = inverse
        info['second_order'] = False
        info['inject'] = inject_list[i]

        pred, info, _, _ = model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
            info=info,
            cur_step=i
        )

        img = img + (t_prev - t_curr) * pred

    return img, info