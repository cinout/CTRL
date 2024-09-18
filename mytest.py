import torchvision.datasets as datasets
from PIL import Image
import torchvision.transforms as transforms
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import copy
from collections import Counter
import random

device = "cuda" if torch.cuda.is_available() else "cpu"
haha = torch.randint(0, 10, (10, 2))
haha = -1 * haha[:, 1]
haha = haha.detach().cpu().tolist()
print(haha)
