from collections import Counter
import numpy as np
import torch
import random

# ACC
acc_byol_avg = [24.5, 86.7, 56.0, 23.4, 87.2, 56.0, 17.4, 82.5, 44.2, 16.2, 83.6, 44.3]
acc_byol_std = [0.2, 0.2, 0.6, 0.8, 0.3, 0.7, 0.3, 0.6, 1.1, 0.6, 0.3, 1.5]

acc_mocov2_avg = [
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
]
acc_mocov2_std = [0.6, 0.2, 0.4, 0.7, 0.3, 0.2, 0.5, 0.3, 1.1, 0.0, 0.5, 1.0]

acc_simclr_avg = [
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
acc_simclr_std = [0.6, 0.1, 0.2, 0.1, 0.4, 0.3, 0.6, 0.5, 1.9, 0.5, 0.8, 1.6]

# ASR
asr_byol_avg = [1.5, 2.2, 1.4, 1.4, 8.3, 14.5, 1.6, 19.8, 1.8, 1.7, 17.2, 6.6]
asr_byol_std = [0.4, 0.2, 0.7, 0.6, 4.1, 8.1, 0.3, 3.1, 1.2, 0.6, 9.6, 4.3]

asr_mocov2_avg = [2.7, 23.5, 22.0, 1.8, 2.3, 0.6, 4.9, 24.4, 17.1, 1.4, 2.9, 0.8]
asr_mocov2_std = [0.9, 2.6, 6.8, 0.4, 0.2, 0.0, 2.8, 7.5, 13.8, 0.4, 0.8, 0.3]

asr_simclr_avg = [4.1, 47.4, 29.0, 2.4, 12.0, 16.7, 1.7, 45.5, 24.7, 2.4, 11.9, 11.9]
asr_simclr_std = [2.5, 2.6, 1.9, 0.3, 2.3, 3.6, 1.0, 1.5, 4.8, 0.6, 0.4, 5.9]


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
