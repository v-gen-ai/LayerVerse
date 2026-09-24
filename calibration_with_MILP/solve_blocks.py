#!/usr/bin/env python3
"""LayerVerse block-role selection: bi-objective MILP over KV-injection roles.

Every transformer block is assigned one of three roles by a single MILP:

  m_i = 1  masked injection   -- source K/V outside the editing mask only,
                                 which protects the background
  s_i = 1  global injection   -- source K/V everywhere, which anchors the
                                 structure of the edited object
  m_i = s_i = 0               -- inactive, the block is left untouched

Two calibrated surrogates are minimised jointly (both in "lower is better"
form):

  J_bg   background fidelity, a function of v_i = m_i + s_i (any injection
         helps the background)
  J_str  structure preservation, a function of s_i only (only global
         injection anchors structure)

Constraints:

  sum(m_i + s_i) = K          total budget of active blocks
  sum(s_i)       = K_str      exact number of global anchors (equality)
  m_i + s_i     <= 1          a block has at most one role

The two objectives live on different scales, so J_str is rescaled by the ratio
of the two landscapes' dynamic ranges before being added (see
``run_block_selection``).

Surrogate coefficients come from a CSV of calibration measurements; see
README.md for the required format, and surrogate_calibration.py for the fit.
"""
import argparse
import itertools
import sys
import numpy as np

try:
    import pulp
except ImportError:
    pulp = None

from surrogate_calibration import (
    extract_data,
    calibrate_model_A,
    calibrate_and_transform_model_B,
    interpolate_models,
    load_data,
    sense_for_metric,
    format_output,
    detect_num_blocks,
)


# =============================================================================
# 1. Calibration Helper
# =============================================================================

def _calibrate_landscape(csv_data, metric_col, P, N_DOUBLE, N_SINGLE,
                         K_target, log_transform, flip_sign,
                         interp_method, opt_mode):
    """Full pipeline for one objective: load -> extract -> calibrate -> interpolate.

    Returns
    -------
    coefs   : tuple (w, W, c0)
    delta   : float  -- |L_full - L_empty| in transformed space
    L_empty : float  -- baseline value (no injections)
    """
    df = load_data(csv_data)
    L_e, L_f, A, B = extract_data(
        df, metric_col, P, N_DOUBLE, N_SINGLE,
        log_transform=log_transform, flip_sign=flip_sign,
    )
    pA = calibrate_model_A(P, L_e, A, mode=opt_mode) if interp_method != 'model_B_only' else None
    pB = calibrate_and_transform_model_B(P, L_f, B, mode=opt_mode) if interp_method != 'model_A_only' else None
    coefs = interpolate_models(P, K_target, pA, pB, method=interp_method)
    delta = abs(L_f - L_e)
    return coefs, delta, L_e


# =============================================================================
# 2. Surrogate Evaluation
# =============================================================================

def _eval_surrogate(x_vec, w, W, c0, use_quad):
    """Exact surrogate value for a binary vector x (including constant c0)."""
    val = c0 + np.dot(w, x_vec)
    if use_quad:
        x_outer = np.outer(x_vec, x_vec)
        val += np.sum(np.triu(x_outer * W, k=1))
    return val


# =============================================================================
# 3. Bi-objective MILP
# =============================================================================

def solve_milp(P, K, K_str,
               coefs_bg, coefs_str,
               lambda_str,
               opt_mode='quadratic_solver', time_limit=3600):
    """Solve the bi-objective role-assignment MILP.

    Objective:
      min  J_bg + lambda_str * J_str
    where
      J_bg  = C_bg  + sum w_bg_i * (m_i + s_i) + sum W_bg_ij * z_bg_ij
      J_str = C_str + sum w_str_i * s_i         + sum W_str_ij * z_str_ij

    Constraints:
      m_i + s_i <= 1           (one role per block)
      sum(m_i + s_i) = K       (total budget)
      sum(s_i)       = K_str   (exact number of global anchors)

    The quadratic terms are linearised with McCormick envelopes: z_bg_ij
    stands for v_i*v_j and z_str_ij for s_i*s_j. Both products are of binary
    variables, so the envelopes are exact at any integral solution.

    Returns
    -------
    (masked_indices, structural_indices, objective_value) or None.
    """
    if pulp is None:
        raise RuntimeError("PuLP not installed. pip install pulp")

    w_bg, W_bg, c0_bg = coefs_bg
    w_str, W_str, c0_str = coefs_str
    use_quad = (opt_mode == 'quadratic_solver')

    model = pulp.LpProblem("LayerVerse_BlockRoles", pulp.LpMinimize)

    # --- Binary variables ---
    m = pulp.LpVariable.dicts("m", range(P), cat='Binary')
    s = pulp.LpVariable.dicts("s", range(P), cat='Binary')

    pairs = list(itertools.combinations(range(P), 2))
    if use_quad:
        z_bg  = pulp.LpVariable.dicts("z_bg",  pairs, lowBound=0, upBound=1, cat='Continuous')
        z_str = pulp.LpVariable.dicts("z_str", pairs, lowBound=0, upBound=1, cat='Continuous')

    # --- Objective: J_bg + lambda * J_str ---
    c0_total = c0_bg + lambda_str * c0_str
    obj = pulp.LpAffineExpression(c0_total)

    for i in range(P):
        obj += w_bg[i] * m[i]
        obj += (w_bg[i] + lambda_str * w_str[i]) * s[i]

    if use_quad:
        for i, j in pairs:
            obj += W_bg[i, j] * z_bg[(i, j)]
            obj += (lambda_str * W_str[i, j]) * z_str[(i, j)]

    model += obj

    # --- Constraints ---
    for i in range(P):
        model += m[i] + s[i] <= 1

    model += pulp.lpSum([m[i] + s[i] for i in range(P)]) == K
    model += pulp.lpSum([s[i] for i in range(P)]) == K_str

    if use_quad:
        for i, j in pairs:
            ui, uj = m[i] + s[i], m[j] + s[j]
            model += z_bg[(i, j)] <= ui
            model += z_bg[(i, j)] <= uj
            model += z_bg[(i, j)] >= ui + uj - 1

            model += z_str[(i, j)] <= s[i]
            model += z_str[(i, j)] <= s[j]
            model += z_str[(i, j)] >= s[i] + s[j] - 1

    # --- Solve ---
    model.solve(pulp.PULP_CBC_CMD(
        msg=0, timeLimit=time_limit,
        gapRel=0.01, timeMode="elapsed",
    ))
    status = pulp.LpStatus[model.status]

    if status != 'Optimal':
        print(f"  MILP status: {status}", file=sys.stderr)
        return None

    res_m = sorted(i for i in range(P) if pulp.value(m[i]) > 0.5)
    res_s = sorted(i for i in range(P) if pulp.value(s[i]) > 0.5)
    return res_m, res_s, pulp.value(model.objective)


# =============================================================================
# 4. Public API
# =============================================================================

def run_block_selection(
    csv_masked, csv_structural,
    metric_bg, metric_str,
    K, K_str,
    N_DOUBLE, N_SINGLE,
    lambda_user_str=1.0,
    log_transform_bg=True,
    log_transform_str=True,
    interpolation_method='model_A_only',
    optimization_mode='quadratic_solver',
    milp_time_limit=3600,
):
    """Calibrate both surrogates and solve for the block-role assignment.

    Parameters
    ----------
    csv_masked       : str or DataFrame -- measurements from masked-injection runs.
    csv_structural   : str or DataFrame -- measurements from global-injection runs.
    metric_bg        : str -- background metric column  (e.g. 'psnr_unedit_part').
    metric_str       : str -- structure metric column   (e.g. 'structure_distance').
    K                : int -- total budget of active blocks.
    K_str            : int -- exact number of global anchor blocks.
    N_DOUBLE, N_SINGLE : int -- architecture layout (FLUX: 19,38; SD3.5: P,0).
    lambda_user_str  : float -- extra multiplier on the structure objective.
    log_transform_bg : bool -- log-transform background metric (used for MSE/LPIPS).
    log_transform_str: bool -- log-transform structure metric.
    interpolation_method : str -- 'model_A_only' (default), 'model_B_only', or a blend of
                                  the two: 'linear', 'cubic'.
    optimization_mode    : str -- 'quadratic_solver' keeps the pairwise terms,
                                  'linear_solver' drops them.
    milp_time_limit      : int -- solver time limit in seconds.

    Returns
    -------
    dict with block indices, predicted metrics, and the lambda used, or None if
    the solver did not reach an optimal solution.
    """
    P = N_DOUBLE + N_SINGLE
    bg_sense = sense_for_metric(metric_bg)
    use_quad = (optimization_mode == 'quadratic_solver')
    flip_bg = (bg_sense == 'maximize')

    assert 0 <= K <= P,       f"K={K} not in [0, {P}]"
    assert 0 <= K_str <= K,   f"K_str={K_str} not in [0, K={K}]"

    print(f"\nLayerVerse | P={P} K={K} K_str={K_str} "
          f"| BG={metric_bg}({bg_sense}) STR={metric_str} "
          f"| interp={interpolation_method} mode={optimization_mode} "
          f"log_bg={log_transform_bg} log_str={log_transform_str}")

    if K == 0:
        return _build_result([], [], N_DOUBLE, N_SINGLE, {}, 0.0, 0.0)

    # ------------------------------------------------------------------
    # 1. Calibrate two landscapes
    # ------------------------------------------------------------------
    coefs_bg, delta_bg, _ = _calibrate_landscape(
        csv_masked, metric_bg, P, N_DOUBLE, N_SINGLE,
        K_target=K, log_transform=log_transform_bg, flip_sign=flip_bg,
        interp_method=interpolation_method, opt_mode=optimization_mode,
    )
    print(f"  BG calibrated  | delta={delta_bg:.6f}")

    coefs_str, delta_str, _ = _calibrate_landscape(
        csv_structural, metric_str, P, N_DOUBLE, N_SINGLE,
        K_target=K_str, log_transform=log_transform_str, flip_sign=False,
        interp_method=interpolation_method, opt_mode=optimization_mode,
    )
    print(f"  STR calibrated | delta={delta_str:.6f}")

    # ------------------------------------------------------------------
    # 2. Put the two objectives on a comparable scale
    # ------------------------------------------------------------------
    eps = 1e-12
    lam_str = lambda_user_str * (delta_bg / (delta_str + eps))
    print(f"  Lambda_Str = {lambda_user_str} * {delta_bg:.4e}/{delta_str:.4e} = {lam_str:.4f}")

    # ------------------------------------------------------------------
    # 3. Solve MILP
    # ------------------------------------------------------------------
    result = solve_milp(
        P, K, K_str,
        coefs_bg, coefs_str,
        lam_str,
        opt_mode=optimization_mode, time_limit=milp_time_limit,
    )
    if result is None:
        print("  Optimization FAILED.", file=sys.stderr)
        return None

    masked_idx, struct_idx, milp_obj = result

    # ------------------------------------------------------------------
    # 4. Exact surrogate predictions for individual metrics
    # ------------------------------------------------------------------
    m_vec = np.zeros(P); m_vec[masked_idx] = 1
    s_vec = np.zeros(P); s_vec[struct_idx] = 1
    v_vec = m_vec + s_vec

    w_bg, W_bg, c0_bg = coefs_bg
    w_str, W_str, c0_str = coefs_str

    pred_bg  = _eval_surrogate(v_vec, w_bg,  W_bg,  c0_bg,  use_quad)
    pred_str = _eval_surrogate(s_vec, w_str, W_str, c0_str, use_quad)

    # De-transform to original metric scale
    if flip_bg:
        final_bg = np.exp(-pred_bg) if log_transform_bg else -pred_bg
    else:
        final_bg = np.exp(pred_bg) if log_transform_bg else pred_bg
    final_str = np.exp(pred_str) if log_transform_str else pred_str

    predictions = {
        'BG_Loss': final_bg,
        'Structure_Distance': final_str,
    }

    # ------------------------------------------------------------------
    # 5. Summary
    # ------------------------------------------------------------------
    nm, ns = len(masked_idx), len(struct_idx)
    print(f"  => Total {nm + ns}/{K} | Structural {ns}/{K_str} "
          f"| Obj={milp_obj:.6f} | lam_str={lam_str:.4f}")
    print(f"  Predicted BG Loss        : {final_bg:.6f}")
    print(f"  Predicted Structure Dist. : {final_str:.6f}")

    return _build_result(masked_idx, struct_idx, N_DOUBLE, N_SINGLE,
                         predictions, milp_obj, lam_str)


# =============================================================================
# 5. Output Formatting
# =============================================================================

def _build_result(masked_idx, struct_idx, N_DOUBLE, N_SINGLE,
                  predictions, milp_obj, lam_str):
    """Package solver output into a result dict."""
    fmt_m = format_output(masked_idx, N_DOUBLE, N_SINGLE)
    fmt_s = format_output(struct_idx, N_DOUBLE, N_SINGLE)
    out = {
        'Masked Global Indices':      masked_idx,
        'Masked Double Blocks':       fmt_m['Double Blocks'],
        'Masked Single Blocks':       fmt_m['Single Blocks'],
        'Structural Global Indices':  struct_idx,
        'Structural Double Blocks':   fmt_s['Double Blocks'],
        'Structural Single Blocks':   fmt_s['Single Blocks'],
        'MILP_Objective':             milp_obj,
        'Lambda_Str':                 lam_str,
    }
    out.update(predictions)
    if N_SINGLE == 0:
        out['Masked Blocks']     = fmt_m.get('Blocks', [])
        out['Structural Blocks'] = fmt_s.get('Blocks', [])
    return out


def _j(indices):
    return ', '.join(str(i) for i in sorted(indices))


def format_inference_yaml(res, arch, K, K_str):
    """Render the solution as the ``selected_combinations_config.yaml`` block
    that the LayerVerse_FLUX / LayerVerse_SD3.5 inference code reads.

    The inference configs list *every* active block plus the subset that gets
    global injection; the masked blocks are the difference of the two.
    """
    header = (f"# LayerVerse block allocation: N = {K} active blocks, of which "
              f"N_str = {K_str} are global anchors.\n")
    if arch == 'flux':
        single = sorted(set(res['Masked Single Blocks']) | set(res['Structural Single Blocks']))
        double = sorted(set(res['Masked Double Blocks']) | set(res['Structural Double Blocks']))
        return header + (
            "selected_blocks_list: [\n"
            "  {\n"
            f'    "single_blocks": [{_j(single)}],\n'
            f'    "double_blocks": [{_j(double)}],\n'
            f'    "global_single_blocks": [{_j(res["Structural Single Blocks"])}],\n'
            f'    "global_double_blocks": [{_j(res["Structural Double Blocks"])}]\n'
            "  },\n"
            "]\n"
        )
    blocks = sorted(set(res['Masked Blocks']) | set(res['Structural Blocks']))
    return header + (
        "selected_blocks_list:\n"
        f"  - blocks: [{_j(blocks)}]\n"
        f"    global_blocks: [{_j(res['Structural Blocks'])}]\n"
    )


# =============================================================================
# 6. CLI
# =============================================================================

def main():
    p = argparse.ArgumentParser(
        description="LayerVerse block-role selection (bi-objective MILP)")

    p.add_argument('--csv_masked',   required=True,
                   help="CSV of masked-injection measurements (background objective)")
    p.add_argument('--csv_structural', required=True,
                   help="CSV of global-injection measurements (structure objective)")
    p.add_argument('--metric_bg',    default='psnr_unedit_part',
                   help="Background metric column (default: psnr_unedit_part)")
    p.add_argument('--metric_str',   default='structure_distance',
                   help="Structure metric column (default: structure_distance)")
    p.add_argument('--K',            type=int, default=10,
                   help="Total budget of active blocks")
    p.add_argument('--K_str',        type=int, default=2,
                   help="Exact number of global anchor blocks")
    p.add_argument('--lam_user_str', type=float, default=1.0,
                   help="User multiplier for structure weight (default 1.0)")
    p.add_argument('--arch',         default='flux', choices=['flux', 'sd3'],
                   help="Architecture (flux: D=19,S=38; sd3: auto-detect or --num_blocks)")
    p.add_argument('--num_blocks',   type=int, default=None,
                   help="Total block count P for sd3 (auto-detected from CSV if omitted)")
    p.add_argument('--interp',       default='model_A_only',
                   choices=['model_A_only', 'model_B_only', 'linear', 'cubic'],
                   help="Surrogate: model A (default), model B, or a budget-dependent blend")
    p.add_argument('--opt_mode',     default='quadratic_solver',
                   choices=['quadratic_solver', 'linear_solver'])
    p.add_argument('--log_bg',       action='store_true',
                   help="Log-transform background metric (recommended for MSE/LPIPS)")
    p.add_argument('--log_str',      action='store_true',
                   help="Log-transform structure metric")
    p.add_argument('--milp_time_limit', type=int, default=3600)
    p.add_argument('--out_yaml', default=None,
                   help="Write the result as a selected_combinations_config.yaml "
                        "for the inference subfolders")
    args = p.parse_args()

    if args.arch == 'flux':
        N_D, N_S = 19, 38
    else:
        detected = args.num_blocks or detect_num_blocks(args.csv_masked)
        N_D, N_S = detected, 0
        print(f"SD3 mode: P={detected} blocks")

    res = run_block_selection(
        csv_masked=args.csv_masked,
        csv_structural=args.csv_structural,
        metric_bg=args.metric_bg,
        metric_str=args.metric_str,
        K=args.K,
        K_str=args.K_str,
        N_DOUBLE=N_D,
        N_SINGLE=N_S,
        lambda_user_str=args.lam_user_str,
        log_transform_bg=args.log_bg,
        log_transform_str=args.log_str,
        interpolation_method=args.interp,
        optimization_mode=args.opt_mode,
        milp_time_limit=args.milp_time_limit,
    )

    if res:
        print(f"\n{'='*60}")
        print("BLOCK ROLE ASSIGNMENT")
        print(f"{'='*60}")
        if N_S > 0:
            print(f"STRUCTURAL (anchor): Double={res['Structural Double Blocks']}, "
                  f"Single={res['Structural Single Blocks']}")
            print(f"MASKED (background): Double={res['Masked Double Blocks']}, "
                  f"Single={res['Masked Single Blocks']}")
        else:
            print(f"STRUCTURAL: {res['Structural Blocks']}")
            print(f"MASKED:     {res['Masked Blocks']}")
        nm = len(res['Masked Global Indices'])
        ns = len(res['Structural Global Indices'])
        print(f"\nTotal: {nm + ns}/{args.K} | Structural: {ns}/{args.K_str}")
        print(f"BG Loss:      {res.get('BG_Loss', float('nan')):.6f}")
        print(f"Str Dist:     {res.get('Structure_Distance', float('nan')):.6f}")
        print(f"MILP Obj:     {res['MILP_Objective']:.6f}")
        print(f"Lambda_Str:   {res['Lambda_Str']:.4f}")

        yaml_text = format_inference_yaml(res, args.arch, args.K, args.K_str)
        print(f"\n{'='*60}")
        print("selected_combinations_config.yaml")
        print(f"{'='*60}")
        print(yaml_text, end='')
        if args.out_yaml:
            with open(args.out_yaml, 'w') as f:
                f.write(yaml_text)
            print(f"Written to {args.out_yaml}")
    else:
        print("Optimization failed.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
