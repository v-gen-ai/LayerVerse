# LayerVerse · Stable Diffusion 3.5-medium

*LayerVerse* image editing on the PIE-Bench benchmark with SD3.5-medium. Environment, checkpoints, and data are set up as in the [main README](../README.md).

## Run

From this folder:

```shell
bash src/scripts/run_fireflow.sh    # FireFlow + LayerVerse
bash src/scripts/run_rfsolver.sh    # RF-Solver + LayerVerse
bash src/scripts/run_euler.sh       # Euler + LayerVerse (used for calibration)
```

The scripts use the paper settings: **28 steps**, CFG **1** for inversion and **7.5** for editing. Further arguments are passed to `src/edit_run_benchmark.py`:

| Argument | Default | Description |
| --- | --- | --- |
| `--devices` | `0` | GPU ids; images are split across them |
| `--data_path` | `../data/PIE-Bench_v1` | PIE-Bench root |
| `--output_dir` | `../results` | directory where edits are written |
| `--edit_category_list` | `0 ... 8` | PIE-Bench editing categories; `9` (style transfer) is excluded |
| `--model`, `--model_folder` | `../ckpts/SD3.5-medium/` | model checkpoint and folder with `clip_l`, `clip_g`, and `t5xxl` |
| `--seed` | `1234` | seed of the VAE posterior sample, reset for every image |
| `--rerun_exist_images` | off | rerun edits that already exist; otherwise they are skipped |

As in the Stability AI reference implementation, the text encoders run on the CPU.

Edits are saved under `results/<solver>_seed_1234__use_attn_mask__use_KV_injection__blocks_0_1--3_4--12--14--17_23__gb_20_23/annotation_images/` with the same relative paths as the PIE-Bench images.

## Configuration

- `src/configs/selected_combinations_config.yaml`: block allocation. Of the 24 joint blocks, `[0, 1, 3, 4, 12, 14, 17, 18, 19]` receive masked injection and `[20, 21, 22, 23]` receive global injection.
- `src/configs/benchmark_config_SD3_editing.yaml`: automatic-mask settings. In the default release configuration, the mask is read at step `7` of a 15-step schedule from blocks `[7, 9, 13, 23]`.

`sd3_infer.py` and `dit_embedder.py` are the unmodified Stability AI files; `mmditx.py`, `sd3_impls.py`, and `other_impls.py` add the KV-injection, masking, and solver code on top of them.
