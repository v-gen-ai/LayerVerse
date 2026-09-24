"""Calibration of the second-order editing-error surrogate from a CSV table.

LayerVerse models the editing error of a block subset S with a quadratic
surrogate

    L(x) = c0 + sum_i w_i x_i + sum_{i<j} W_ij x_i x_j,     x_i = [i in S]

and fits its coefficients in closed form from a small set of measurements:

  Model A (from below)  uses the no-injection baseline plus every single block
                        and every block pair.  This is the default.
  Model B (from above)  optionally uses the all-blocks measurement plus every
                        "all but one" and "all but two" subset.  It is more
                        accurate for large |S|.

The two can also be blended by a budget-dependent weight (see
``get_interpolation_alpha``).

See README.md for the exact CSV rows this requires.
"""
import re
from io import StringIO

import numpy as np
import pandas as pd

# =============================================================================
# 1. Parsing and interpolation helpers
# =============================================================================

def parse_index_string(index_str):
    """Parse an index list, including ranges (e.g. '0, 5, 10,...,18')."""
    indices = set()
    # Normalise 'start,...,end' to 'start-end'
    normalized_str = re.sub(r'(\d+)\s*,\.\.\.,\s*(\d+)', r'\1-\2', index_str)

    for part in normalized_str.split(','):
        part = part.strip()
        if not part: continue
        try:
            if '-' in part:
                start, end = map(int, part.split('-'))
                indices.update(range(start, end + 1))
            else:
                indices.add(int(part))
        except ValueError:
            raise ValueError(f"Malformed index: '{part}' in '{index_str}'")
    return sorted(list(indices))

def parse_blocks_key(key_string, N_DOUBLE, N_SINGLE):
    """Parse a blocks_key string into a set of global block indices.

    Two formats are supported:
    - FLUX:   'double_[...]_single_[...]' or 'single_[...]_double_[...]'
    - SD3.5:  'blocks_[...]'
    """
    # SD3.5 format: blocks_[...]
    sd3_match = re.search(r"blocks_\[(.*?)\]", key_string)
    if sd3_match and 'single_' not in key_string and 'double_' not in key_string:
        block_str = sd3_match.group(1)
        indices = parse_index_string(block_str) if block_str else []
        P = N_DOUBLE + N_SINGLE  # For SD3.5, N_DOUBLE=P, N_SINGLE=0
        return set(idx for idx in indices if 0 <= idx < P)

    # FLUX format: double_[...]_single_[...]
    match = re.search(r"double_\[(.*?)\]_single_\[(.*?)\]", key_string)
    if not match:
        match = re.search(r"single_\[(.*?)\]_double_\[(.*?)\]", key_string)
        if match:
            single_str, double_str = match.groups()
        else:
            return None
    else:
        double_str, single_str = match.groups()

    double_indices = parse_index_string(double_str) if double_str else []
    single_indices = parse_index_string(single_str) if single_str else []

    # Convert to global indices (double blocks come first)
    global_indices = set()
    for idx in double_indices:
        if 0 <= idx < N_DOUBLE:
            global_indices.add(idx)
    for idx in single_indices:
        if 0 <= idx < N_SINGLE:
            global_indices.add(idx + N_DOUBLE)

    return global_indices

def get_interpolation_alpha(P, K, method="linear"):
    """Blending weight alpha for model A.

    alpha=1 at K=0 (pure model A), alpha=0 at K=P (pure model B).
    """
    if P == 0: return 1.0

    if method == "model_A_only":
        return 1.0
    elif method == "model_B_only":
        return 0.0

    t = K / P  # normalised budget (0 to 1)

    if method == "linear":
        # alpha = 1 - t
        return 1.0 - t
    elif method == "cubic":
        # Smoothstep. Weight for B: h(t) = 3t^2 - 2t^3; weight for A: 1 - h(t).
        return 1.0 - (3.0 * t**2 - 2.0 * t**3)
    else:
        raise ValueError(f"Unknown interpolation method: {method}")

# =============================================================================
# 2. Measurement extraction and model calibration
# =============================================================================

def extract_data(df, metric_name, P, N_DOUBLE, N_SINGLE, log_transform=False, flip_sign=False):
    """Extract L_empty, L_full and the measurement matrices A and B.

    Transform order: raw -> log (if log_transform) -> negate (if flip_sign).
    flip_sign=True turns a maximisation metric (PSNR, SSIM) into a minimisation
    one.

    A[i, i] holds the measurement for the single block {i} and A[i, j] the one
    for the pair {i, j}; B[i, i] holds "all blocks except i" and B[i, j] "all
    blocks except i and j".
    """
    if metric_name not in df.columns:
        raise ValueError(f"Metric '{metric_name}' not found.")

    L_empty, L_full = None, None
    A = np.full((P, P), np.nan)
    B = np.full((P, P), np.nan)
    U = set(range(P))
    key_col_name = df.columns[0]

    for index, row in df.iterrows():
        if pd.isna(row[metric_name]): continue

        if log_transform:
            assert float(row[metric_name]) > 0, f"Metric value in row {index} must be positive."

        try:
            value = float(row[metric_name]) if not log_transform else np.log(float(row[metric_name]))
        except (ValueError, TypeError):
            raise ValueError(f"Malformed metric value in row {index}.")

        if flip_sign:
            value = -value

        S = parse_blocks_key(row[key_col_name], N_DOUBLE, N_SINGLE)
        if S is None: continue

        size = len(S)

        try:
            if size == 0:
                L_empty = value
            elif size == P:
                L_full = value
            elif size == 1:
                i = list(S)[0]; A[i, i] = value
            elif size == 2:
                i, j = sorted(list(S)); A[i, j] = A[j, i] = value
            elif size == P - 1:
                i = list(U - S)[0]; B[i, i] = value
            elif size == P - 2:
                i, j = sorted(list(U - S)); B[i, j] = B[j, i] = value
        except IndexError:
            # Ignore indices outside [0, P)
            continue

    if L_empty is None or L_full is None:
        raise ValueError("Missing the empty-set (L_empty) or full-set (L_full) row.")

    return L_empty, L_full, A, B

def calibrate_model_A(P, L_empty, A, mode="quadratic_solver"):
    """Calibrate model A (from below). Returns wA, WA, c0A."""
    wA = np.zeros(P)
    WA = np.zeros((P, P))
    c0A = L_empty

    for i in range(P):
        if np.isnan(A[i, i]):
            raise ValueError(f"Incomplete data: A[{i}, {i}] is missing.")
        wA[i] = A[i, i] - L_empty

        if mode == "quadratic_solver":

            for j in range(i + 1, P):
                if np.isnan(A[i, j]):
                    raise ValueError(f"Incomplete data: A[{i}, {j}] is missing.")
                # W_ij^A = A_ij - A_ii - A_jj + L_empty
                WA[i, j] = A[i, j] - A[i, i] - A[j, j] + L_empty
                WA[j, i] = WA[i, j]
    return wA, WA, c0A

def calibrate_and_transform_model_B(P, L_full, B, mode="quadratic_solver"):
    """Calibrate model B (from above) and rewrite it in the inclusion basis.

    Returns wB_new, WB_new, c0B_new.
    """
    # Step 1: calibrate in the exclusion basis y -> (vB, VB)
    vB = np.zeros(P)
    VB = np.zeros((P, P))
    for i in range(P):
        if np.isnan(B[i, i]):
            raise ValueError(f"Incomplete data: B[{i}, {i}] is missing.")
        vB[i] = B[i, i] - L_full

        if mode == "quadratic_solver":

            for j in range(i + 1, P):
                if np.isnan(B[i, j]):
                    raise ValueError(f"Incomplete data: B[{i}, {j}] is missing.")
                # V_ij^B = B_ij - B_ii - B_jj + L_full
                VB[i, j] = B[i, j] - B[i, i] - B[j, j] + L_full
                VB[j, i] = VB[i, j]

    # Step 2: transform into the inclusion basis x

    # W_ij^{B_new} = V_ij^B
    WB_new = VB.copy()

    # w_i^{B_new} = -v_i^B - sum_{j!=i} V_ij^B
    sum_VB_rows = np.sum(VB, axis=1)
    wB_new = -vB - sum_VB_rows

    # Constant c_0^{B_new} = L_full + sum v_i^B + sum_{i<j} V_ij^B
    sum_vB = np.sum(vB)
    # Upper triangle only (i<j)
    sum_V_ij_upper = np.sum(np.triu(VB, k=1))
    c0B_new = L_full + sum_vB + sum_V_ij_upper

    return wB_new, WB_new, c0B_new

def interpolate_models(P, K, paramsA, paramsB, method="model_A_only"):
    """Blend the coefficients of models A and B; single-model methods need only that model."""
    if method == "model_A_only":
        return paramsA
    if method == "model_B_only":
        return paramsB
    wA, WA, c0A = paramsA
    wB_new, WB_new, c0B_new = paramsB

    alpha = get_interpolation_alpha(P, K, method)

    w_comb = alpha * wA + (1 - alpha) * wB_new
    W_comb = alpha * WA + (1 - alpha) * WB_new
    c0_comb = alpha * c0A + (1 - alpha) * c0B_new

    return w_comb, W_comb, c0_comb

# =============================================================================
# 3. Metric bookkeeping and I/O
# =============================================================================

def sense_for_metric(metric):
    """Infer whether a metric column is higher-is-better from its name."""
    m = metric.lower()
    if 'psnr' in m or 'ssim' in m:
        return 'maximize'
    if 'structure' in m or 'mse' in m or 'lpips' in m:
        return 'minimize'
    raise ValueError(
        f"Cannot infer the direction of metric '{metric}'. Column names "
        "containing psnr/ssim are treated as higher-is-better and those "
        "containing structure/mse/lpips as lower-is-better; rename the column "
        "or extend sense_for_metric()."
    )

def format_output(indices, N_DOUBLE, N_SINGLE=None):
    """Format global indices back into per-stream block notation.

    For SD3.5 (N_SINGLE=0) only 'Blocks' is meaningful; for FLUX the indices
    are split into 'Double Blocks' and 'Single Blocks'.
    """
    if N_SINGLE == 0:
        # SD3.5 mode: joint blocks only
        return {
            "Global Indices": sorted(indices),
            "Blocks": sorted(indices),
            "Double Blocks": [],
            "Single Blocks": []
        }

    # FLUX mode
    double_blocks, single_blocks = [], []
    for idx in indices:
        if idx < N_DOUBLE:
            double_blocks.append(idx)
        else:
            single_blocks.append(idx - N_DOUBLE)
    return {
        "Global Indices": sorted(indices),
        "Double Blocks": sorted(double_blocks),
        "Single Blocks": sorted(single_blocks)
    }

def detect_num_blocks(csv_path):
    """Auto-detect P (total block count) from an SD3-format CSV.

    Scans all ``blocks_[...]`` entries and returns max_index + 1.
    """
    max_idx = -1
    with open(csv_path) as f:
        for line in f:
            m = re.search(r"blocks_\[(.*?)\]", line)
            if m:
                idxs = [int(x.strip()) for x in m.group(1).split(',') if x.strip()]
                if idxs:
                    max_idx = max(max_idx, max(idxs))
    if max_idx < 0:
        raise ValueError(f"Could not detect block count from {csv_path}")
    return max_idx + 1


def load_data(file_path_or_data):
    """Load a table from a path, a raw CSV string, or a DataFrame."""
    if isinstance(file_path_or_data, pd.DataFrame):
        return file_path_or_data
    elif isinstance(file_path_or_data, str):
        # Heuristic: a short .csv-suffixed string without newlines is a path
        is_file = (file_path_or_data.endswith('.csv') and '\n' not in file_path_or_data and len(file_path_or_data) <= 255)
        if is_file:
            return pd.read_csv(file_path_or_data)
        else:
            return pd.read_csv(StringIO(file_path_or_data))
    else:
        raise TypeError("Unsupported input type.")
