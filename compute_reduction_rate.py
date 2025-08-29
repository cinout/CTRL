import numpy as np

"""
ACC
"""
uncleansed_acc = [
    # BYOL
    42.7,
    86.9,
    55.9,
    42.8,
    88,
    56.6,
    22.2,
    76.7,
    34.9,
    22.3,
    77.3,
    36.2,
    # MoCo
    42.4,
    78.1,
    45.5,
    42.5,
    79.3,
    45.5,
    26,
    69.5,
    24.5,
    26.7,
    70.6,
    27.2,
    # SimCLR
    38.1,
    84.9,
    48.8,
    38.3,
    85,
    49.7,
    21.6,
    73,
    25,
    24.6,
    75.7,
    24.8,
]

# FIXME: update
baseline_acc = [
    # BYOL
    40.8,
    86.9,
    54.0,
    40.8,
    86.5,
    54.2,
    20.3,
    82.4,
    37.8,
    21.0,
    77.8,
    36.4,
    # MoCo
    41.0,
    77.7,
    44.9,
    41.1,
    78.1,
    44.1,
    22.4,
    67.4,
    21.8,
    24.5,
    68.8,
    24.3,
    # SimCLR
    36.4,
    80.7,
    47.1,
    36.2,
    79.5,
    47.7,
    18.9,
    64.6,
    21.8,
    22.1,
    69.3,
    24.5,
]
uncleansed_acc = np.array(uncleansed_acc)
baseline_acc = np.array(baseline_acc)
baseline_percent_wrt_uncleansed = (baseline_acc - uncleansed_acc) / uncleansed_acc
print(f"ACC reduction rate:{baseline_percent_wrt_uncleansed.mean()*100:.1f}")

"""
ASR
"""
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

# FIXME: update
baseline_asr = [
    # BYOL
    0.3,
    2.8,
    0.1,
    0.4,
    33.2,
    0.1,
    0.3,
    2.5,
    0.0,
    0.6,
    11.9,
    0.2,
    # MoCo
    1.8,
    47.5,
    1.8,
    0.8,
    1.3,
    0.0,
    0.4,
    51.2,
    0.2,
    0.7,
    2.9,
    0.2,
    # SimCLR
    1.0,
    1.0,
    0.1,
    0.2,
    16.9,
    0.0,
    0.1,
    0.1,
    0.0,
    0.1,
    4.5,
    8.8,
]


# ours_sample16_asr = [
#     #
# ]

uncleansed_asr = np.array(uncleansed_asr)
baseline_asr = np.array(baseline_asr)
# ours_sample16_asr = np.array(ours_sample16_asr)

# """
# Absolute Diff
# """

# diff = baseline_asr - ours_sample16_asr

# baseline_wins = np.sum(diff < 0)
# our_wins = np.sum(diff > 0)
# ties = np.sum(diff == 0)

# print(f"baseline_wins: {baseline_wins}; our_wins: {our_wins}; ties: {ties}")


"""
percent diff with respect to uncleansed
"""
# FIXME: update
filter = np.array([1, 10, 1] * 12)

uncleansed_asr_gt_1 = uncleansed_asr > filter

uncleansed_asr = uncleansed_asr[uncleansed_asr_gt_1]
baseline_asr = baseline_asr[uncleansed_asr_gt_1]
# ours_sample16_asr = ours_sample16_asr[uncleansed_asr_gt_1]


baseline_percent_wrt_uncleansed = (baseline_asr - uncleansed_asr) / uncleansed_asr
# ours_sample16_percent_wrt_uncleansed = (
#     ours_sample16_asr - uncleansed_asr
# ) / uncleansed_asr


print(f"ASR reduction rate:{baseline_percent_wrt_uncleansed.mean()*100:.1f}")
