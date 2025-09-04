from collections import Counter
import numpy as np
import torch
import random

# ACC
acc_byol_avg = [39.2, 87.1, 53.8, 35.8, 87.5, 50.1, 18.7, 78.4, 34.8, 15.2, 77.8, 30.3]
acc_byol_std = [5.1, 1.3, 9.9, 7.0, 1.9, 6.0, 5.8, 4.2, 13.9, 8.7, 3.7, 8.4]

acc_mocov2_avg = [10.8, 48.0, 29.5, 26.8, 51.3, 37.0, 1.3, 17.4, 8.5, 10.3, 23.3, 16.4]
acc_mocov2_std = [0.9, 1.7, 5.2, 11.1, 6.8, 2.6, 0.3, 0.9, 3.3, 9.3, 9.5, 2.5]

acc_simclr_avg = [11.0, 55.4, 31.8, 11.2, 51.5, 41.7, 0.9, 27.1, 11.9, 1.5, 26.1, 22.0]
acc_simclr_std = [0.8, 16.4, 14.8, 1.6, 21.9, 4.8, 0.2, 21.6, 15.1, 0.3, 27.8, 3.6]

# ASR
asr_byol_avg = [0.8, 3.0, 35.7, 24.9, 57.0, 61.6, 0.0, 4.6, 36.0, 36.9, 60.7, 38.7]
asr_byol_std = [1.2, 2.3, 30.2, 34.0, 34.0, 52.2, 0.0, 5.4, 31.0, 41.4, 40.6, 34.9]

asr_mocov2_avg = [0.8, 11.9, 1.0, 10.8, 3.0, 0.4, 0.0, 1.8, 23.3, 6.9, 22.3, 0.5]
asr_mocov2_std = [0.4, 8.8, 0.8, 16.7, 0.6, 0.1, 0.0, 3.0, 33.3, 11.8, 28.4, 0.4]

asr_simclr_avg = [0.9, 4.3, 28.7, 0.3, 5.8, 56.8, 0.0, 2.3, 25.9, 0.0, 15.9, 54.2]
asr_simclr_std = [0.7, 1.9, 46.7, 0.2, 4.3, 47.7, 0.0, 3.2, 42.1, 0.0, 27.6, 46.3]


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
