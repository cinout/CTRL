from collections import Counter
import numpy as np
import torch
import random

# ACC
acc_byol_avg = [42.4, 87.2, 55.7, 41.8, 87.6, 56.3, 20.2, 84.9, 33.5, 21.1, 75.5, 34.3]
acc_byol_std = [0.6, 0.1, 0.2, 0.5, 0.2, 0.1, 0.6, 0.2, 0.4, 0.3, 2.0, 0.4]


acc_mocov2_avg = [
    42.5,
    78.9,
    45.9,
    42.5,
    79.6,
    46.5,
    24.9,
    68.3,
    24.0,
    25.3,
    69.2,
    26.5,
]
acc_mocov2_std = [0.9, 0.3, 0.2, 0.2, 0.1, 0.2, 0.4, 1.1, 0.2, 0.7, 1.0, 0.2]


acc_simclr_avg = [
    38.4,
    85.1,
    49.9,
    38.8,
    85.4,
    50.4,
    20.4,
    70.8,
    24.7,
    23.6,
    74.9,
    24.0,
]
acc_simclr_std = [0.3, 0.1, 0.2, 0.2, 0.3, 0.3, 0.2, 0.6, 0.9, 0.3, 0.6, 0.8]

# ASR
asr_byol_avg = [2.1, 66.8, 38.7, 44.9, 86.5, 88.0, 0.0, 78.8, 46.1, 64.6, 92.0, 67.0]
asr_byol_std = [1.0, 21.3, 21.9, 2.1, 0.4, 0.8, 0.0, 31.4, 11.9, 1.0, 1.6, 10.9]


asr_mocov2_avg = [2.0, 84.7, 43.2, 41.8, 1.8, 0.3, 6.6, 61.9, 19.6, 22.2, 3.8, 1.1]
asr_mocov2_std = [0.6, 0.2, 2.7, 2.6, 0.1, 0.1, 6.7, 8.9, 2.3, 7.1, 0.3, 0.2]


asr_simclr_avg = [
    48.6,
    85.7,
    80.3,
    28.3,
    82.3,
    72.5,
    48.8,
    66.9,
    71.0,
    14.1,
    86.6,
    47.2,
]
asr_simclr_std = [12.5, 0.5, 0.6, 4.8, 2.2, 0.9, 22.9, 4.2, 5.9, 5.3, 3.0, 18.5]


acc_byol = "\t".join([f"{v}±{e}" for v, e in zip(acc_byol_avg, acc_byol_std)]) + "\n"
acc_mocov2 = (
    "\t".join([f"{v}±{e}" for v, e in zip(acc_mocov2_avg, acc_mocov2_std)]) + "\n"
)
acc_simclr = (
    "\t".join([f"{v}±{e}" for v, e in zip(acc_simclr_avg, acc_simclr_std)]) + "\n"
)
asr_byol = "\t".join([f"{v}±{e}" for v, e in zip(asr_byol_avg, asr_byol_std)]) + "\n"
asr_mocov2 = (
    "\t".join([f"{v}±{e}" for v, e in zip(asr_mocov2_avg, asr_mocov2_std)]) + "\n"
)
asr_simclr = (
    "\t".join([f"{v}±{e}" for v, e in zip(asr_simclr_avg, asr_simclr_std)]) + "\n"
)


output_file = open("combined.txt", "w", encoding="utf-8")


output_file.write(acc_byol)
output_file.write(acc_mocov2)
output_file.write(acc_simclr)
output_file.write(asr_byol)
output_file.write(asr_mocov2)
output_file.write(asr_simclr)
