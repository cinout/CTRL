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
    42.4,
    86.7,
    54.7,
    42.1,
    87.7,
    55.4,
    21.5,
    81.9,
    39.7,
    21.5,
    81.3,
    36.9,
    # MoCo
    42.3,
    78.4,
    45.8,
    42.4,
    79.5,
    45.8,
    24.7,
    68.0,
    24.0,
    25.6,
    68.9,
    26.2,
    # SimCLR
    37.5,
    82.0,
    48.0,
    37.9,
    82.6,
    48.6,
    20.7,
    70.5,
    24.9,
    22.9,
    72.3,
    24.0,
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
    1.7,
    6.6,
    42.5,
    47.2,
    87.8,
    89.6,
    0.0,
    10.5,
    52.5,
    66.6,
    91.5,
    68.7,
    # MoCo
    15.6,
    84.4,
    38.3,
    39.8,
    2.2,
    0.4,
    15.9,
    71.3,
    18.5,
    23.4,
    5.6,
    1.5,
    # SimCLR
    51.8,
    88.2,
    79.1,
    35.4,
    86.5,
    69.6,
    26.9,
    75.6,
    65.7,
    14.8,
    91.3,
    35.2,
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
