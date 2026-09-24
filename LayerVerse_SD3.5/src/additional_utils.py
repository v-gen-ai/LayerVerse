"""Config loading and prompt-diff helpers for LayerVerse editing with SD3.5."""

import os

import yaml
from easydict import EasyDict


def find_new_and_changed_tokens(source_tokens, target_tokens):
    """Indices of target tokens that are not aligned to the source by the longest common subsequence."""

    def lcs_alignment(seq1, seq2):
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i - 1] == seq2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        alignment = []
        i, j = m, n
        while i > 0 and j > 0:
            if seq1[i - 1] == seq2[j - 1]:
                alignment.append((i - 1, j - 1))
                i -= 1
                j -= 1
            elif dp[i - 1][j] > dp[i][j - 1]:
                i -= 1
            else:
                j -= 1
        return list(reversed(alignment))

    aligned = {target_idx for _, target_idx in lcs_alignment(source_tokens, target_tokens)}
    return [i for i in range(len(target_tokens)) if i not in aligned]


def get_continuous_ranges(blocks):
    """Compress block indices into a folder-name string, e.g. [0, 1, 3, 4, 12] -> '0_1--3_4--12'."""
    if not blocks:
        return ""
    blocks = sorted(blocks)
    ranges = []
    start = end = blocks[0]
    for b in blocks[1:]:
        if b == end + 1:
            end = b
        else:
            ranges.append(str(start) if start == end else f"{start}_{end}")
            start = end = b
    ranges.append(str(start) if start == end else f"{start}_{end}")
    return "--".join(ranges)


def get_benchmark_config(config_path):
    """Load an editing config and expand it into one dict per block allocation.

    SD3.5 has a single stream of joint blocks: `blocks` lists every active block
    and `global_blocks` the subset with global injection. `selected_blocks_url`
    is resolved relative to the config file.
    """
    params = yaml.safe_load(open(config_path))
    blocks_url = os.path.join(os.path.dirname(os.path.abspath(config_path)), params["selected_blocks_url"])
    allocations = yaml.safe_load(open(blocks_url))["selected_blocks_list"]

    config = EasyDict(params)
    config.all_configs = [
        {
            "use_KV_injection": params["use_KV_injection"],
            "use_mask": params["use_mask"],
            "selected_blocks": entry["blocks"],
            "global_blocks": entry.get("global_blocks", []),
            "mask_blocks": params["mask_blocks"],
            "mask_sigma_index": params["mask_sigma_index"],
            "mask_setting_num_steps": params["mask_setting_num_steps"],
        }
        for entry in allocations
    ]
    return config
