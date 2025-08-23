from collections import Counter
import numpy as np
import torch
import random

# ACC
acc_byol_avg = [5.5, 87.1, 56.7, 8.1, 88.0, 56.2, 6.6, 83.6, 42.8, 7.7, 82.8, 41.8]
acc_byol_std = [2.2, 0.3, 0.4, 2.8, 0.3, 1.2, 2.0, 0.9, 0.3, 1.9, 0.5, 1.4]

acc_mocov2_avg = [
    33.8,
    71.5,
    36.1,
    34.7,
    72.3,
    36.6,
    27.1,
    68.0,
    23.7,
    26.6,
    67.9,
    24.8,
]
acc_mocov2_std = [0.6, 1.0, 1.2, 2.7, 1.4, 1.4, 0.8, 0.6, 0.5, 1.4, 0.2, 0.4]

acc_simclr_avg = [
    35.0,
    85.9,
    51.8,
    35.9,
    86.2,
    51.4,
    27.2,
    83.5,
    38.5,
    26.7,
    82.5,
    38.9,
]
acc_simclr_std = [1.0, 0.2, 0.1, 1.0, 0.1, 0.2, 0.4, 0.1, 0.1, 0.9, 0.1, 0.3]

# ASR
asr_byol_avg = [0.7, 8.7, 7.4, 0.5, 45.4, 45.7, 1.0, 65.7, 10.4, 3.2, 51.4, 31.8]
asr_byol_std = [0.2, 2.7, 4.3, 0.3, 11.9, 6.5, 0.3, 12.0, 3.4, 0.2, 45.1, 4.8]

asr_mocov2_avg = [14.3, 38.6, 26.9, 12.6, 3.2, 1.0, 24.0, 20.7, 17.1, 14.5, 3.5, 0.9]
asr_mocov2_std = [2.9, 10.3, 11.4, 8.0, 0.6, 0.3, 2.6, 16.1, 6.0, 10.8, 0.3, 0.1]

asr_simclr_avg = [4.1, 69.0, 36.2, 0.9, 58.9, 51.4, 0.9, 58.5, 31.3, 1.2, 64.1, 31.9]
asr_simclr_std = [1.9, 2.8, 3.0, 0.5, 3.5, 1.6, 1.2, 1.9, 2.4, 0.6, 4.2, 1.4]


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
