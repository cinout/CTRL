import torchvision.datasets as datasets
from PIL import Image
import torchvision.transforms as transforms
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import copy
from collections import Counter

a = torch.tensor([0, 0, 0, 0, 0])
print(1 in a)
print(torch.nonzero(a == 1).flatten())
