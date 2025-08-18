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
    85.0,
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
baseline_acc = [
    # BYOL
    24.5,
    86.7,
    56.0,
    23.4,
    87.2,
    56.0,
    17.4,
    82.5,
    44.2,
    16.2,
    83.6,
    44.3,
    # MoCo
    31.4,
    76.8,
    43.8,
    30.5,
    77.7,
    43.6,
    23.6,
    71.3,
    28.0,
    22.4,
    72.8,
    29.1,
    # SimCLR
    30.4,
    87.2,
    53.9,
    32.0,
    87.1,
    53.4,
    23.4,
    84.7,
    42.1,
    24.5,
    83.6,
    42.4,
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
    67.7,
    44.0,
    41.9,
    83.0,
    88.0,
    0.0,
    94.6,
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
    1.5,
    2.2,
    1.4,
    1.4,
    8.3,
    14.5,
    1.6,
    19.8,
    1.8,
    1.7,
    17.2,
    6.6,
    # MoCo
    2.7,
    23.5,
    22.0,
    1.8,
    2.3,
    0.6,
    4.9,
    24.4,
    17.1,
    1.4,
    2.9,
    0.8,
    # SimCLR
    4.1,
    47.4,
    29.0,
    2.4,
    12.0,
    16.7,
    1.7,
    45.5,
    24.7,
    2.4,
    11.9,
    11.9,
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
