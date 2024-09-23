import torchvision.datasets as datasets
from PIL import Image
import torchvision.transforms as transforms
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import copy
from collections import Counter
import random

a = ["red", "blue"]
a.remove("red")
a.append("green")
a.append("orange")
print(a)
