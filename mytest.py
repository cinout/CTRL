import torchvision.datasets as datasets
from PIL import Image
import torchvision.transforms as transforms
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import copy
from collections import Counter
import random

a = [12, 3, 4, 3, 5, 11, 12, 6, 7]

x = Counter(a)

res = [key for key, count in x.items() if count == 2]
print(res)
