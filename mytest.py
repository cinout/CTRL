from collections import Counter
import numpy as np
import torch
import random


uncleansed_asr = [
    # BYOL
    2.1,
    4.1,
    44.0,
    41.9,
    83.0,
    88.0,
    0.0,
    2.0,
    50.6,
    62.7,
    89.9,
    67.3,
    # MoCo
    5.5,
    82.0,
    40.3,
    44.9,
    1.9,
    0.5,
    17.0,
    65.5,
    22.5,
    23.8,
    4.1,
    1.4,
    # SimCLR
    38.4,
    84.2,
    79.1,
    35.1,
    81.8,
    70.6,
    43.8,
    66.0,
    68.9,
    14.0,
    86.7,
    53.2,
]


baseline_asr = [
    # BYOL
    1.9,
    2.1,
    2.2,
    2.1,
    6.5,
    13.5,
    1.6,
    5.7,
    3.1,
    2.4,
    26.0,
    6.2,
    # MoCo
    1.8,
    23.9,
    20.1,
    2.2,
    2.4,
    0.6,
    2.6,
    15.8,
    20.6,
    1.8,
    3.7,
    1.0,
    # SimCLR
    1.8,
    23.9,
    20.1,
    2.2,
    2.4,
    0.6,
    2.6,
    15.8,
    20.6,
    1.8,
    3.7,
    1.0,
]


ours_sample16_asr = [
    # BYOL
    0.4,
    4.5,
    0.0,
    0.8,
    37.8,
    0.1,
    0.3,
    7.6,
    0.0,
    0.7,
    14.3,
    0.3,
    # MoCo
    2.4,
    53.7,
    3.5,
    0.8,
    1.1,
    0.1,
    0.3,
    56.7,
    1.2,
    0.7,
    2.1,
    0.3,
    # SimCLR
    1.3,
    0.8,
    0.0,
    0.4,
    14.6,
    0.0,
    0.0,
    0.1,
    0.0,
    0.0,
    3.1,
    12.0,
]

uncleansed_asr = np.array(uncleansed_asr)
baseline_asr = np.array(baseline_asr)
ours_sample16_asr = np.array(ours_sample16_asr)

"""
Absolute Diff
"""

diff = baseline_asr - ours_sample16_asr

baseline_wins = np.sum(diff < 0)
our_wins = np.sum(diff > 0)
ties = np.sum(diff == 0)

print(f"baseline_wins: {baseline_wins}; our_wins: {our_wins}; ties: {ties}")


"""
percent diff with respect to uncleansed
"""
# only consider uncleansed ASR>1.0
uncleansed_asr_gt_1 = uncleansed_asr > 1

uncleansed_asr = uncleansed_asr[uncleansed_asr_gt_1]
baseline_asr = baseline_asr[uncleansed_asr_gt_1]
ours_sample16_asr = ours_sample16_asr[uncleansed_asr_gt_1]


baseline_percent_wrt_uncleansed = (baseline_asr - uncleansed_asr) / uncleansed_asr
ours_sample16_percent_wrt_uncleansed = (
    ours_sample16_asr - uncleansed_asr
) / uncleansed_asr


print(
    f"baseline reduction rate:{baseline_percent_wrt_uncleansed.mean()}, our reduction rate: {ours_sample16_percent_wrt_uncleansed.mean()}"
)
