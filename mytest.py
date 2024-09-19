import torchvision.datasets as datasets
from PIL import Image
import torchvision.transforms as transforms
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import copy
from collections import Counter
import random

aa = [0, 2, 3]
print(aa[-2:])
print(aa[-2:-1])
print(aa[-2:0])
