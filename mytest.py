from collections import Counter
import numpy as np
import torch
import random

a = [87.1, 86.9, 86.9]
b = [2.0, 5.7, 2.9]


c = [75.0, 74.6, 78.3]
d = [1.1, 1.8, 0.5]

print(f"{np.mean(a):.1f}±{np.std(a):.1f}")
print(f"{np.mean(b):.1f}±{np.std(b):.1f}")
print(f"{np.mean(c):.1f}±{np.std(c):.1f}")
print(f"{np.mean(d):.1f}±{np.std(d):.1f}")
