import torchvision.datasets as datasets
from PIL import Image
import torchvision.transforms as transforms
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


topk_channel = 2
hey = np.random.randint(0, 20, size=(3, 8))
print(hey)
rest = np.argsort(hey, axis=1)
print(rest)
rest = rest[:, -topk_channel:]
print(rest)

print(rest.flatten())
