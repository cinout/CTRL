import torchvision.datasets as datasets
from PIL import Image
import torchvision.transforms as transforms
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import copy

is_poisoned = np.array([0, 1, 1, 0, 1, 1])

res = np.nonzero(is_poisoned == 1)
print(type(res[0]))
