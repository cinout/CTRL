import torchvision.datasets as datasets
from PIL import Image
import torchvision.transforms as transforms
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

x_memory_tensor = torch.randn(size=(10, 20))
for x in x_memory_tensor:
    print(x)
