# Modified for LayerVerse from UniEdit-Flow (https://github.com/DSL-Lab/UniEdit-Flow, Apache-2.0):
# checkpoints default to ../ckpts/FLUX.1-dev; FLUX.1-schnell, Hub download and watermarking removed.
import os
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file as load_sft

from flux.model import Flux, FluxParams
from flux.modules.autoencoder import AutoEncoder, AutoEncoderParams
from flux.modules.conditioner import HFEmbedder


@dataclass
class ModelSpec:
    params: FluxParams
    ae_params: AutoEncoderParams
    ckpt_path: str
    ae_path: str

# flux -> src -> LayerVerse_FLUX -> directory holding ckpts/
DEFAULT_CKPT_DIR = Path(__file__).resolve().parents[3] / "ckpts" / "FLUX.1-dev"

configs = {
    "flux-dev": ModelSpec(
        ckpt_path=os.getenv("FLUX_DEV", str(DEFAULT_CKPT_DIR / "flux1-dev.safetensors")),
        params=FluxParams(
            in_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=True,
        ),
        ae_path=os.getenv("AE", str(DEFAULT_CKPT_DIR / "ae.safetensors")),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
}


def print_load_warning(missing: list[str], unexpected: list[str]) -> None:
    if len(missing) > 0 and len(unexpected) > 0:
        print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
        print("\n" + "-" * 79 + "\n")
        print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))
    elif len(missing) > 0:
        print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
    elif len(unexpected) > 0:
        print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))


def load_flow_model(name: str, device: str | torch.device = "cuda"):
    # Loading Flux
    print("Init model")

    with torch.device("meta"):
        model = Flux(configs[name].params).to(torch.bfloat16)

    print("Loading checkpoint")
    # load_sft doesn't support torch.device
    sd = load_sft(configs[name].ckpt_path, device=str(device))
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    print_load_warning(missing, unexpected)
    return model


def load_t5(device: str | torch.device = "cuda", max_length: int = 512) -> HFEmbedder:
    # max length 64, 128, 256 and 512 should work (if your sequence is short enough)
    return HFEmbedder("google/t5-v1_1-xxl", max_length=max_length, is_clip=False, torch_dtype=torch.bfloat16).to(device)


def load_clip(device: str | torch.device = "cuda") -> HFEmbedder:
    return HFEmbedder("openai/clip-vit-large-patch14", max_length=77, is_clip=True, torch_dtype=torch.bfloat16).to(device)


def load_ae(name: str, device: str | torch.device = "cuda") -> AutoEncoder:
    # Loading the autoencoder
    print("Init AE")
    with torch.device("meta"):
        ae = AutoEncoder(configs[name].ae_params)

    sd = load_sft(configs[name].ae_path, device=str(device))
    missing, unexpected = ae.load_state_dict(sd, strict=False, assign=True)
    print_load_warning(missing, unexpected)
    return ae
