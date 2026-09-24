# LayerVerse · FLUX.1-[dev]

*LayerVerse* image editing on the PIE-Bench benchmark with FLUX.1-[dev]. Environment, checkpoints, and data are set up as in the [main README](../README.md).

## Run

From this folder:

```shell
bash src/scripts/run_fireflow.sh    # FireFlow + LayerVerse
bash src/scripts/run_rfsolver.sh    # RF-Solver + LayerVerse
bash src/scripts/run_euler.sh       # Euler + LayerVerse (used for calibration)
```

The scripts use the paper settings: **15 steps**, guidance **1** for inversion and **2** for editing. Further arguments are passed to `src/edit_run_benchmark.py`:

| Argument | Default | Description |
| --- | --- | --- |
| `--devices` | `0` | GPU ids; images are split across them |
| `--data_path` | `../data/PIE-Bench_v1` | PIE-Bench root |
| `--output_dir` | `../results` | directory where edits are written |
| `--edit_category_list` | `0 ... 8` | PIE-Bench editing categories; `9` (style transfer) is excluded |
| `--rerun_exist_images` | off | rerun edits that already exist; otherwise they are skipped |

Weights are read from `../ckpts/FLUX.1-dev/flux1-dev.safetensors` and `../ckpts/FLUX.1-dev/ae.safetensors`; the `FLUX_DEV` and `AE` environment variables override these paths. A GPU worker holds FLUX.1-[dev], T5-XXL, and CLIP-L in bf16 and peaks at about **35 GB** of GPU memory.

Edits are saved under `results/<solver>_seed_1234__use_attn_mask__use_KV_injection__sb_32_37__db_1_2__gdb_2__gts_0-->14/annotation_images/` with the same relative paths as the PIE-Bench images.

## Configuration

- `src/configs/selected_combinations_config.yaml`: block allocation. Masked injection is applied to Double block `1` and Single blocks `32–37`; global injection is applied to Double block `2`.
- `src/configs/benchmark_config_FLUX_editing.yaml`: automatic-mask settings, global-injection timesteps, seed, and `save_mask`.

For the default release configuration, the automatic mask is extracted at step `11` of a 15-step schedule from blocks `[14, 17, 18, 23]`, counted over the 19 Double blocks followed by the 38 Single blocks.
