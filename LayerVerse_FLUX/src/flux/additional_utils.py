import os

import yaml
from easydict import EasyDict


def get_config(config_path, seeds=None):

    config_editing = EasyDict(__name__='Img2Img Editing')

    params_yaml = yaml.safe_load(open(config_path, 'r'))

    # Selected blocks, resolved relative to the config that references them
    config_editing.selected_blocks_url = os.path.join(os.path.dirname(os.path.abspath(config_path)),
                                                      params_yaml["selected_blocks_url"])
    config_editing.selected_blocks_list = yaml.safe_load(open(config_editing.selected_blocks_url, 'r'))["selected_blocks_list"]

    # KV injection
    config_editing.use_KV_injection = params_yaml["use_KV_injection"]

    # Automatic editing mask
    config_editing.use_mask = params_yaml["use_mask"]
    config_editing.save_mask = params_yaml["save_mask"]
    config_editing.mask_setting = params_yaml["mask_setting"]
    config_editing.mask_setting_num_steps = params_yaml["mask_setting_num_steps"]

    # Global KV injection timestep range
    config_editing.global_timestep_starts = params_yaml["global_timestep_starts"]
    config_editing.global_timestep_ends = params_yaml["global_timestep_ends"]

    # seeds
    if seeds is not None:
        config_editing.seeds = seeds
    elif params_yaml["seeds"] is not None:
        config_editing.seeds = params_yaml["seeds"]
    else:
        raise ValueError("seeds or seeds_list must be provided")

    assert len(config_editing.global_timestep_starts) == len(config_editing.global_timestep_ends)

    config_editing.all_configs = []

    for s in range(len(config_editing.seeds)):
        for block_ind in range(len(config_editing.selected_blocks_list)):
            for gts_ind in range(len(config_editing.global_timestep_starts)):

                selected_blocks = config_editing.selected_blocks_list[block_ind]

                i2i_config = {
                    "use_KV_injection": config_editing.use_KV_injection,
                    "selected_single_blocks": selected_blocks["single_blocks"],
                    "selected_double_blocks": selected_blocks["double_blocks"],
                    "global_single_blocks": selected_blocks["global_single_blocks"],
                    "global_double_blocks": selected_blocks["global_double_blocks"],
                    "global_timestep_start": config_editing.global_timestep_starts[gts_ind],
                    "global_timestep_end": config_editing.global_timestep_ends[gts_ind],
                    "use_mask": config_editing.use_mask,
                    "save_mask": config_editing.save_mask,
                    "mask_setting": config_editing.mask_setting,
                    "mask_setting_num_steps": config_editing.mask_setting_num_steps,
                    "seed": config_editing.seeds[s],
                }

                config_editing.all_configs.append(i2i_config)

    for i2i_config in config_editing.all_configs:
        assert i2i_config["seed"] >= 0
        assert i2i_config["global_timestep_start"] <= i2i_config["global_timestep_end"]

    return config_editing

def find_new_and_changed_tokens(source_tokens, target_tokens):

    def lcs_alignment(seq1, seq2):
        m, n = len(seq1), len(seq2)

        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])

        alignment = []
        i, j = m, n
        
        while i > 0 and j > 0:
            if seq1[i-1] == seq2[j-1]:
                alignment.append((i-1, j-1))
                i -= 1
                j -= 1
            elif dp[i-1][j] > dp[i][j-1]:
                i -= 1
            else:
                j -= 1
        
        return list(reversed(alignment))
    
    alignment = lcs_alignment(source_tokens, target_tokens)

    aligned_target_positions = set(target_idx for _, target_idx in alignment)

    new_token_indices = []

    for i in range(len(target_tokens)):
        if i not in aligned_target_positions:
            new_token_indices.append(i)

    return new_token_indices

def get_continuous_ranges(blocks):
    if not blocks:
        return ""
    
    blocks = sorted(blocks)
    ranges = []
    start = blocks[0]
    end = blocks[0]
    
    for i in range(1, len(blocks)):
        if blocks[i] == end + 1:
            end = blocks[i]
        else:
            if start == end:
                ranges.append(str(start))
            else:
                ranges.append(f"{start}_{end}")
            start = blocks[i]
            end = blocks[i]

    if start == end:
        ranges.append(str(start))
    else:
        ranges.append(f"{start}_{end}")
    
    return "--".join(ranges)