# Modified for LayerVerse from UniEdit-Flow's src/edit.py (https://github.com/DSL-Lab/UniEdit-Flow, Apache-2.0).
"""LayerVerse editing on PIE-Bench with FLUX.1-dev.

For every image: invert it while caching the source K/V, build the editing mask
from one forward pass, then regenerate it under the target prompt with masked and
global KV-injection. Images are split across --devices.
"""
import argparse
import json
import math
import os

import numpy as np
import torch
import torch.multiprocessing as multiprocessing
import torch.nn.functional as F
from einops import rearrange
from PIL import Image
from torchvision.utils import save_image
from tqdm import tqdm

from flux.additional_utils import find_new_and_changed_tokens, get_config, get_continuous_ranges
from flux.sampling import denoise_fireflow, denoise_rf_solver, denoise_with_mask, get_schedule, prepare, unpack
from flux.util import load_ae, load_clip, load_flow_model, load_t5

SAMPLERS = {"euler": denoise_with_mask, "rf_solver": denoise_rf_solver, "fireflow": denoise_fireflow}
MODEL_NAME = "flux-dev"

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SRC_DIR))


@torch.inference_mode()
def encode(init_image, torch_device, ae):
    init_image = torch.from_numpy(init_image).permute(2, 0, 1).float() / 127.5 - 1
    init_image = init_image.unsqueeze(0)
    init_image = init_image.to(torch_device)
    init_image = ae.encode(init_image.to()).to(torch.bfloat16)
    return init_image


def t5_tokens(t5, prompt):
    ids = t5.tokenizer(prompt, truncation=True, return_length=False, return_overflowing_tokens=False)["input_ids"]
    return [t5.tokenizer.convert_ids_to_tokens(i) for i in ids][:-1]


def main(device_id, items, configs, args, threads_per_process):
    torch.set_num_threads(threads_per_process)
    torch.set_grad_enabled(False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch_device = torch.device(f"cuda:{device_id}")
    torch.cuda.set_device(torch_device)
    t5 = load_t5(torch_device, max_length=512)
    clip = load_clip(torch_device)
    model = load_flow_model(MODEL_NAME, device=torch_device)
    ae = load_ae(MODEL_NAME, device=torch_device)
    denoise = SAMPLERS[args.sampler]

    for item in tqdm(items, desc=f"GPU {device_id}", position=int(device_id)):
        source_prompt = item["original_prompt"].replace("[", "").replace("]", "")
        target_prompt = item["editing_prompt"].replace("[", "").replace("]", "")
        image_path = os.path.join(args.data_path, "annotation_images", item["image_path"])

        image = np.array(Image.open(image_path).convert("RGB"))
        image = image[: image.shape[0] - image.shape[0] % 16, : image.shape[1] - image.shape[1] % 16]
        height, width = image.shape[:2]
        init_image = encode(image, torch_device, ae)

        for config in configs:
            output_dir = os.path.join(args.output_dir, f"{args.sampler}_seed_{config['seed']}")
            if config["use_mask"]:
                output_dir += "__use_attn_mask"
            if config["use_KV_injection"]:
                output_dir += "__use_KV_injection"
            output_dir += f"__sb_{get_continuous_ranges(config['selected_single_blocks'])}"
            output_dir += f"__db_{get_continuous_ranges(config['selected_double_blocks'])}"
            if config["global_single_blocks"]:
                output_dir += f"__gsb_{get_continuous_ranges(config['global_single_blocks'])}"
            if config["global_double_blocks"]:
                output_dir += f"__gdb_{get_continuous_ranges(config['global_double_blocks'])}"
            if config["global_single_blocks"] or config["global_double_blocks"]:
                output_dir += f"__gts_{config['global_timestep_start']}-->{config['global_timestep_end']}"
            save_path = os.path.join(output_dir, "annotation_images", item["image_path"])
            if os.path.exists(save_path) and not args.rerun_exist_images:
                continue
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            torch.manual_seed(config["seed"])
            torch.cuda.empty_cache()
            info = {
                "feature": {},
                "inject_step": args.inject,
                "use_KV_injection": config["use_KV_injection"],
                "use_mask": config["use_mask"],
                "selected_single_blocks": config["selected_single_blocks"],
                "selected_double_blocks": config["selected_double_blocks"],
                "global_single_blocks": config["global_single_blocks"],
                "global_double_blocks": config["global_double_blocks"],
                "global_timestep_start": config["global_timestep_start"],
                "global_timestep_end": config["global_timestep_end"],
            }

            inp = prepare(t5, clip, init_image, prompt=source_prompt)
            inp_target = prepare(t5, clip, init_image, prompt=target_prompt)
            inp_target["src_img"] = inp["img"].clone().detach()
            inp_target["src_txt"] = inp["txt"]
            inp_target["src_txt_ids"] = inp["txt_ids"]
            inp_target["src_vec"] = inp["vec"]
            timesteps = get_schedule(args.num_steps, inp["img"].shape[1])

            if config["use_mask"]:
                info["mask_setting"] = config["mask_setting"]
                info["mask_setting_num_steps"] = config["mask_setting_num_steps"]
                info["masking_timesteps"] = get_schedule(config["mask_setting_num_steps"], inp["img"].shape[1])
            # Tokens added to (new) or removed from (deleted) the prompt; they locate the editing mask.
            source_tokens, target_tokens = t5_tokens(t5, source_prompt), t5_tokens(t5, target_prompt)
            info["new_token_indices"] = find_new_and_changed_tokens(source_tokens, target_tokens)
            info["deleted_token_indices"] = find_new_and_changed_tokens(target_tokens, source_tokens)
            info["orig_latent_size"] = init_image.shape[2:]

            # Inversion under the source prompt (guidance 1) caches the source K/V.
            z, info = denoise(model, **inp, timesteps=timesteps, guidance=1, inverse=True, info=info)
            inp_target["img"] = z
            inp_target["init_noise"] = z.clone().detach()
            x, info = denoise(model, **inp_target, timesteps=timesteps, guidance=args.guidance, inverse=False,
                              info=info)

            x = unpack(x.float(), height, width)
            with torch.autocast(device_type=torch_device.type, dtype=torch.bfloat16):
                x = ae.decode(x)
            x = rearrange(x.clamp(-1, 1)[0], "c h w -> h w c")
            Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy()).save(save_path)

            if config["save_mask"]:
                mask = F.interpolate(info["total_attn_map"][None, None], size=(height, width), mode="bilinear",
                                     align_corners=False).squeeze(0)
                save_image(mask, save_path.replace(".jpg", "_attn_map.jpg"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LayerVerse editing on PIE-Bench (FLUX.1-dev)")
    parser.add_argument("--data_path", default=os.path.join(REPO_ROOT, "data", "PIE-Bench_v1"), help="PIE-Bench root")
    parser.add_argument("--output_dir", default=os.path.join(REPO_ROOT, "results"))
    parser.add_argument("--i2i_config", default=os.path.join(SRC_DIR, "configs", "benchmark_config_FLUX_editing.yaml"))
    parser.add_argument("--sampler", default="fireflow", choices=list(SAMPLERS))
    parser.add_argument("--num_steps", type=int, default=15)
    parser.add_argument("--inject", type=int, default=15, help="number of steps with KV-injection")
    parser.add_argument("--guidance", type=float, default=2.0, help="guidance of the editing pass")
    parser.add_argument("--devices", nargs="+", default=["0"], help="GPU ids; images are split across them")
    parser.add_argument("--edit_category_list", nargs="+", default=[str(i) for i in range(9)],
                        help="PIE-Bench editing types (9, style transfer, is excluded by default)")
    parser.add_argument("--rerun_exist_images", action="store_true")
    args = parser.parse_args()

    with open(os.path.join(args.data_path, "mapping_file.json")) as f:
        items = [item for item in json.load(f).values() if item["editing_type_id"] in args.edit_category_list]
    configs = get_config(args.i2i_config).all_configs
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
