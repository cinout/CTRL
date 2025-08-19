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
    41.8,
    87.2,
    54.9,
    41.7,
    86.8,
    55.3,
    20.6,
    82.3,
    37.9,
    21.4,
    77.3,
    35.5,
    # MoCo
    42.1,
    78.4,
    46.2,
    41.9,
    79.0,
    45.4,
    23.3,
    68.2,
    22.2,
    24.9,
    69.7,
    24.7,
    # SimCLR
    37.5,
    81.9,
    48.4,
    37.7,
    79.9,
    49.0,
    19.9,
    66.7,
    22.7,
    22.8,
    69.6,
    22.8,
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
# only consider uncleansed ASR>1.0
uncleansed_asr_gt_1 = uncleansed_asr > 1

uncleansed_asr = uncleansed_asr[uncleansed_asr_gt_1]
baseline_asr = baseline_asr[uncleansed_asr_gt_1]
# ours_sample16_asr = ours_sample16_asr[uncleansed_asr_gt_1]


baseline_percent_wrt_uncleansed = (baseline_asr - uncleansed_asr) / uncleansed_asr
# ours_sample16_percent_wrt_uncleansed = (
#     ours_sample16_asr - uncleansed_asr
# ) / uncleansed_asr


print(f"ASR reduction rate:{baseline_percent_wrt_uncleansed.mean()*100:.1f}")
