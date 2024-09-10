import torchvision.datasets as datasets
from PIL import Image
import torchvision.transforms as transforms
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import copy

hey1 = torch.tensor([2, 4, 5])
hey2 = torch.tensor([2, 3, 5])
now = []
now.append(hey1)
now.append(hey2)

now = torch.cat(now)
print(now)
