# LayerVerse: block-role calibration

This module chooses which transformer blocks receive **masked** KV-injection, which receive **global** KV-injection, and which stay inactive. `surrogate_calibration.py` fits the error surrogates from measured metrics; `solve_blocks.py` solves the MILP and writes the block allocation used by the editing runners.

**The main-paper experiments use only the bottom-up Model A.** It estimates individual block effects and pairwise interactions from no-injection, single-block, and block-pair measurements. All launch examples below explicitly use `--interp model_A_only`, which is also the code default.

**Model B and the fuller A/B interpolation are introduced in Supplementary Section S1 as theoretical extensions.** The code supports these optional modes, but they are not used for the main-paper results. Their input requirements are listed separately below.

The supplied FLUX.1-[dev] and SD3.5-medium allocations are already calibrated; this step is optional for editing with those configurations.

## Input: two CSV tables

The paper's calibration measurements use **115 held-out images from Microsoft COCO (MS-COCO)**, their **ground-truth masks**, and an **Euler solver**. Use ground-truth masks when collecting the masked-injection measurements and measuring background preservation. Average each metric over the calibration images.

This module reads precomputed measurements. The CSVs and the script for collecting them are not included in the release.

| Flag | Measured with | Provides |
| --- | --- | --- |
| `--csv_masked` | Masked KV-injection into the listed blocks | Background metric (`--metric_bg`) |
| `--csv_structural` | Global KV-injection into the listed blocks | Structure metric (`--metric_str`) |

Each CSV row describes one block subset. The first column contains its key (the header is ignored); the remaining columns contain the averaged metrics. These excerpts illustrate the format, not complete inputs or reported results:

```csv
blocks_key,psnr_unedit_part,lpips_unedit_part
blocks_[],14.21,0.3312
blocks_[0],14.87,0.3190
"blocks_[0, 1]",15.93,0.2871
```

```csv
blocks_key,structure_distance
blocks_[],0.1021
blocks_[0],0.0934
"blocks_[0, 1]",0.0718
```

### Subset keys

Block indices are **0-based** and written explicitly.

- **SD3.5-medium:** `blocks_[i, j, ...]`, with indices `0`–`23`.
- **FLUX.1-[dev]:** `double_[...]_single_[...]` or `single_[...]_double_[...]`, with Double indices `0`–`18` and Single indices `0`–`37`.
- Empty subsets: `blocks_[]` or `double_[]_single_[]`.

Text after the lists is ignored. Quote CSV fields containing commas, as in the examples above.

### Required rows for Model A (main paper)

Let `P` be the total number of blocks. **Model A fits its coefficients from subsets of sizes `0`, `1`, and `2`.** The implementation also requires the full-set row (size `P`) in each CSV to scale the two objectives. This row is used for normalization; it does not fit Model B.

| Backbone | `P` | Required subset sizes | Rows per CSV |
| --- | ---: | --- | ---: |
| FLUX.1-[dev] | 57 | `0`, `1`, `2`, `P` | 1,655 |
| SD3.5-medium | 24 | `0`, `1`, `2`, `P` | 302 |

Include every subset of these sizes, once per CSV. Model A does **not** require the all-but-one or all-but-two measurements. With `--opt_mode linear_solver`, the pairwise rows (size `2`) are omitted; use the default `quadratic_solver` for the paper's pairwise model. Missing required measurements raise an error; rows with empty metric cells are skipped.

### Metric columns

For the background objective, column names containing `psnr` or `ssim` mean higher is better; names containing `structure`, `mse`, or `lpips` mean lower is better. Other names are rejected. The structure objective is always minimized; use a lower-is-better metric such as `structure_distance`.

`--log_bg` and `--log_str` optionally log-transform the corresponding metric before fitting. They are off by default in the CLI; enabled transforms require strictly positive values.

## Run with Model A (main paper)

Use the environment from the [main README](../README.md). From the repository root:

```shell
cd calibration_with_MILP
```

Prepare the two CSVs for the chosen backbone at the paths below. Both examples use **Model A only**, with the active-block and global-anchor budgets reported in the main paper.

**FLUX.1-[dev]: 8 active blocks, including 1 global anchor**

```shell
python solve_blocks.py --arch flux --K 8 --K_str 1 \
    --csv_masked calibration_data/flux_masked.csv --csv_structural calibration_data/flux_structural.csv \
    --metric_bg psnr_unedit_part --metric_str structure_distance --interp model_A_only \
    --out_yaml selected_combinations_config.yaml
```

**SD3.5-medium: 13 active blocks, including 4 global anchors**

```shell
python solve_blocks.py --arch sd3 --K 13 --K_str 4 \
    --csv_masked calibration_data/sd35_masked.csv --csv_structural calibration_data/sd35_structural.csv \
    --metric_bg psnr_unedit_part --metric_str structure_distance --interp model_A_only \
    --out_yaml selected_combinations_config.yaml
```

## Options and objective

| Option | Meaning |
| --- | --- |
| `--K`, `--K_str` | Exact total number of active blocks and, within that total, global anchors. |
| `--arch` | `flux` uses `P = 57`; `sd3` uses `--num_blocks` or infers `P` from `--csv_masked`. |
| `--interp` | **`model_A_only` (default; main paper)**. Other modes are supplementary extensions; see below. |
| `--opt_mode` | `quadratic_solver` (default) keeps pairwise terms; `linear_solver` keeps only individual effects. |
| `--lam_user_str` | Additional multiplier for the structure objective; default `1.0`. |
| `--log_bg`, `--log_str` | Log-transform the corresponding metric; off by default in the CLI. |
| `--milp_time_limit` | MILP time limit in seconds; default `3600`. |
| `--out_yaml` | Output YAML path; omit to print the allocation without saving it. |

The objectives are scaled by their dynamic ranges:

<p align="center">
  <img src="../assets/objective_lambda.svg" alt="Structure weight: user multiplier times the ratio of background and structure metric ranges." width="420">
</p>

The implementation adds `1e-12` to the denominator for numerical stability. The ranges are computed after any log transformation and sign change.

The solver minimizes background error over **all active blocks** and structure error over **global anchors only**, with exactly `K` active blocks, exactly `K_str` global anchors, and at most one role per block. CBC is configured with a 1% relative-gap stopping tolerance.

## Optional modes from the supplementary extension

The main paper uses **Model A only**. Supplementary Section S1 introduces **Model B**, calibrated by removing blocks from the full set, and a fuller landscape that interpolates between A and B. The following modes are supported in code but were **not used for the main-paper experiments**:

| `--interp` | Role | Required subset sizes | Rows per CSV: FLUX / SD3.5 |
| --- | --- | --- | --- |
| `model_B_only` | Top-down Model B from Supplementary S1. | `0`, `P-2`, `P-1`, `P` | 1,655 / 302 |
| `cubic` | A/B interpolation with the cubic weight described in Supplementary S1. | `0`, `1`, `2`, `P-2`, `P-1`, `P` | 3,308 / 602 |
| `linear` | Additional linear A/B interpolation option in the implementation. | `0`, `1`, `2`, `P-2`, `P-1`, `P` | 3,308 / 602 |

Counts assume `quadratic_solver`. For these optional modes, `linear_solver` omits the pairwise subset sizes `2` and `P-2` where applicable. The empty-set and full-set rows are always required for normalization. Interpolation uses the active-block budget `K` for the background objective and the global-anchor budget `K_str` for the structure objective.

## Output

The solver prints the selected masked and global blocks, predicted metrics, the objective value, and $\lambda$. With `--out_yaml`, it also saves the allocation. The selected block identities depend on the input measurements.

For SD3.5-medium, the supplied allocation illustrates the YAML format:

```yaml
selected_blocks_list:
  - blocks: [0, 1, 3, 4, 12, 14, 17, 18, 19, 20, 21, 22, 23]
    global_blocks: [20, 21, 22, 23]
```

`blocks` lists all active blocks; `global_blocks` identifies the global anchors. The other active blocks receive masked injection. FLUX uses `double_blocks` / `single_blocks` and `global_double_blocks` / `global_single_blocks` for the same convention.

To use the result, replace `src/configs/selected_combinations_config.yaml` in the corresponding backbone directory with the generated YAML.
