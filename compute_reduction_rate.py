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
    86.9,
    55.2,
    42.1,
    87.7,
    55.1,
    20.9,
    80.6,
    37.2,
    21.4,
    80.7,
    35.9,
    # MoCo
    42.4,
    78.5,
    46.2,
    42.3,
    79.5,
    45.8,
    23.8,
    68.4,
    23.0,
    25.3,
    69.6,
    25.8,
    # SimCLR
    37.4,
    83.2,
    48.4,
    37.6,
    82.9,
    48.9,
    19.9,
    70.6,
    22.9,
    23.0,
    72.2,
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
    2.8,
    6.1,
    0.1,
    15.8,
    55.1,
    15.9,
    2.0,
    3.9,
    0.0,
    25.0,
    35.6,
    4.6,
    # MoCo
    6.3,
    64.5,
    10.6,
    6.7,
    1.9,
    0.2,
    3.6,
    52.7,
    5.0,
    5.3,
    4.5,
    0.7,
    # SimCLR
    2.4,
    16.6,
    0.1,
    8.7,
    56.4,
    9.8,
    0.3,
    12.1,
    0.0,
    3.1,
    53.0,
    16.9,
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
