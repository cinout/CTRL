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
    42.0,
    87.2,
    55.0,
    41.9,
    87.1,
    55.4,
    20.5,
    82.2,
    37.9,
    21.6,
    77.9,
    36.3,
    # MoCo
    42.0,
    78.3,
    46.2,
    42.2,
    79.1,
    45.5,
    23.5,
    68.2,
    22.4,
    25.1,
    70.0,
    24.9,
    # SimCLR
    37.5,
    82.3,
    48.6,
    37.8,
    80.8,
    48.9,
    19.8,
    67.5,
    22.9,
    22.8,
    70.0,
    23.6,
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
    0.5,
    4.2,
    0.0,
    0.7,
    37.7,
    0.2,
    0.8,
    6.4,
    0.0,
    0.8,
    15.9,
    0.5,
    # MoCo
    2.6,
    54.7,
    2.4,
    1.1,
    1.4,
    0.1,
    0.9,
    58.1,
    0.6,
    1.1,
    3.0,
    0.3,
    # SimCLR
    1.1,
    1.8,
    0.0,
    0.4,
    24.3,
    0.0,
    0.0,
    0.2,
    0.0,
    0.2,
    10.1,
    10.9,
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
