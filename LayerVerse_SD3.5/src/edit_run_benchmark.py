"""LayerVerse editing on PIE-Bench with SD3.5-medium.

For every image: invert it while caching the source K/V of the active blocks,
build the editing mask from one forward pass, then regenerate it under the target
prompt with masked and global KV-injection. Images are split across --devices.
"""

import argparse
import json
import math
import multiprocessing
import os

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from additional_utils import find_new_and_changed_tokens, get_benchmark_config, get_continuous_ranges
from sd3_impls import (
    CFGDenoiser,
    SD3LatentFormat,
    compute_attention_mask,
    sample_euler_inverse,
    sample_euler_kv,
    sample_fireflow_inverse,
    sample_fireflow_kv,
    sample_rf_solver_inverse,
    sample_rf_solver_kv,
    set_edit_map_indices,
)
from sd3_infer import CONFIGS, SHIFT, SD3Inferencer

SAMPLERS = {
    "euler": (sample_euler_inverse, sample_euler_kv),
    "rf_solver": (sample_rf_solver_inverse, sample_rf_solver_kv),
    "fireflow": (sample_fireflow_inverse, sample_fireflow_kv),
}

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SRC_DIR))
CLIP_LEN = 77


def tokenize(tokenizer, prompt, **kwargs):
    return tokenizer.convert_ids_to_tokens(tokenizer(prompt, **kwargs)["input_ids"])


def main(device_id, items, configs, args, threads_per_process):
    torch.set_num_threads(threads_per_process)
    torch.set_grad_enabled(False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    torch_device = torch.device(f"cuda:{device_id}")
    torch.cuda.set_device(torch_device)

    model_name = os.path.splitext(os.path.basename(args.model))[0]
    inferencer = SD3Inferencer()
    inferencer.load(model=args.model, shift=CONFIGS.get(model_name, {}).get("shift", SHIFT),
                    model_folder=args.model_folder, verbose=False)
    sd3_model = inferencer.sd3.model.to(torch_device)
    # The joint sequence is [registers | CLIP (77) | T5 | image], so T5 columns start here.
    t5_token_offset = sd3_model.diffusion_model.register_length + CLIP_LEN

    clip_tokenizer = inferencer.tokenizer.clip_l.tokenizer
    t5_tokenizer = inferencer.tokenizer.t5xxl.tokenizer
    sample_inverse, sample_edit = SAMPLERS[args.sampler]
    sigmas = inferencer.get_sigmas(sd3_model.model_sampling, args.num_steps).to(torch_device)

    for item in tqdm(items, desc=f"GPU {device_id}", position=int(device_id)):
        source_prompt = item["original_prompt"].replace("[", "").replace("]", "")
        target_prompt = item["editing_prompt"].replace("[", "").replace("]", "")
        image_path = os.path.join(args.data_path, "annotation_images", item["image_path"])

        image = np.array(Image.open(image_path).convert("RGB"))
        image = image[: image.shape[0] - image.shape[0] % 16, : image.shape[1] - image.shape[1] % 16]
        # SDVAE.encode samples the posterior; seeding per image keeps results independent of order.
        torch.manual_seed(args.seed)
        latent = SD3LatentFormat().process_in(inferencer.vae_encode(Image.fromarray(image))).half().to(torch_device)
        latent_size = latent.shape[2:]

        source_cond = inferencer.fix_cond(inferencer.get_cond(source_prompt))
        target_cond = inferencer.fix_cond(inferencer.get_cond(target_prompt))
        uncond = inferencer.fix_cond(inferencer.get_cond(""))

        # Tokens added to (new) or removed from (del) the prompt, located separately for each text encoder.
        # CLIP sequences are [BOS, tokens..., EOS, pad...], so indices are shifted by one for the BOS token.
        src, tgt = (tokenize(clip_tokenizer, p, truncation=True, max_length=CLIP_LEN)[1:-1]
                    for p in (source_prompt, target_prompt))
        new_clip = [i + 1 for i in find_new_and_changed_tokens(src, tgt)]
        del_clip = [i + 1 for i in find_new_and_changed_tokens(tgt, src)]
        # T5 sequences are [tokens..., EOS]; prompts are cut to 226 characters as in SD3Tokenizer.
        src, tgt = (tokenize(t5_tokenizer, p[:226], truncation=True)[:-1] for p in (source_prompt, target_prompt))
        new_t5 = find_new_and_changed_tokens(src, tgt)
        del_t5 = find_new_and_changed_tokens(tgt, src)

        for config in configs:
            blocks, global_blocks = config["selected_blocks"], config["global_blocks"]

            output_dir = os.path.join(args.output_dir, f"{args.sampler}_seed_{args.seed}")
            if config["use_mask"]:
                output_dir += "__use_attn_mask"
            if config["use_KV_injection"]:
                output_dir += "__use_KV_injection"
            output_dir += f"__blocks_{get_continuous_ranges(blocks)}"
            if global_blocks:
                output_dir += f"__gb_{get_continuous_ranges(global_blocks)}"
            save_path = os.path.join(output_dir, "annotation_images", item["image_path"])
            if os.path.exists(save_path) and not args.rerun_exist_images:
                continue
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.cuda.empty_cache()

            info = {
                "use_kv_injection": config["use_KV_injection"],
                "features": {},
                "selected_blocks": blocks,
                "global_blocks": global_blocks,
                "use_mask_injection": config["use_mask"],
                "t5_token_offset": t5_token_offset,
            }

            # Inversion under the source prompt (CFG 1) caches the source K/V.
            noise = sample_inverse(CFGDenoiser(sd3_model), latent.clone(), sigmas,
                                   {"cond": source_cond, "uncond": uncond, "cond_scale": 1.0}, info)

            if config["use_mask"]:
                mask_sigmas = sigmas
                if config["mask_setting_num_steps"] != args.num_steps:
                    mask_sigmas = inferencer.get_sigmas(sd3_model.model_sampling,
                                                        config["mask_setting_num_steps"]).to(torch_device)
                mask_args = (sd3_model, noise, latent, mask_sigmas, config["mask_sigma_index"])
                # Added and changed words are located under the target prompt, removed ones under the source.
                tgt_mask = compute_attention_mask(*mask_args, target_cond, new_clip, new_t5, info,
                                                  config["mask_blocks"], latent_size)
                src_mask = compute_attention_mask(*mask_args, source_cond, del_clip, del_t5, info,
                                                  config["mask_blocks"], latent_size)
                set_edit_map_indices(info, torch.logical_or(tgt_mask > 0, src_mask > 0).to(latent.dtype))

            edited = sample_edit(CFGDenoiser(sd3_model), noise.clone(), sigmas,
                                 {"cond": target_cond, "uncond": uncond, "cond_scale": args.guidance}, info)
            inferencer.vae_decode(SD3LatentFormat().process_out(edited)).save(save_path)

        torch.cuda.empty_cache()


if __name__ == "__main__":
    ckpt_dir = os.path.join(REPO_ROOT, "ckpts", "SD3.5-medium")
    parser = argparse.ArgumentParser(description="LayerVerse editing on PIE-Bench (SD3.5-medium)")
    parser.add_argument("--model", default=os.path.join(ckpt_dir, "sd3.5_medium.safetensors"), help="MMDiT checkpoint")
    parser.add_argument("--model_folder", default=ckpt_dir, help="folder with clip_l, clip_g and t5xxl safetensors")
    parser.add_argument("--data_path", default=os.path.join(REPO_ROOT, "data", "PIE-Bench_v1"), help="PIE-Bench root")
    parser.add_argument("--output_dir", default=os.path.join(REPO_ROOT, "results"))
    parser.add_argument("--i2i_config", default=os.path.join(SRC_DIR, "configs", "benchmark_config_SD3_editing.yaml"))
    parser.add_argument("--sampler", default="fireflow", choices=list(SAMPLERS))
    parser.add_argument("--num_steps", type=int, default=28)
    parser.add_argument("--guidance", type=float, default=7.5, help="CFG scale of the editing pass")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--devices", nargs="+", default=["0"], help="GPU ids; images are split across them")
    parser.add_argument("--edit_category_list", nargs="+", default=[str(i) for i in range(9)],
                        help="PIE-Bench editing types (9, style transfer, is excluded by default)")
    parser.add_argument("--rerun_exist_images", action="store_true")
    args = parser.parse_args()

    with open(os.path.join(args.data_path, "mapping_file.json")) as f:
        items = [item for item in json.load(f).values() if str(item["editing_type_id"]) in args.edit_category_list]
    configs = get_benchmark_config(args.i2i_config).all_configs
    shards = [items[i::len(args.devices)] for i in range(len(args.devices))]
    threads_per_process = max(1, math.floor(os.cpu_count() / len(args.devices)))
    print(f"{len(items)} images, {len(configs)} config(s), GPUs {args.devices}")

    multiprocessing.set_start_method("spawn", force=True)
    processes = [multiprocessing.Process(target=main, args=(device, shard, configs, args, threads_per_process))
                 for device, shard in zip(args.devices, shards) if shard]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
