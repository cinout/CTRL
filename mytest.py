import torchvision.datasets as datasets
from PIL import Image
import torchvision.transforms as transforms
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import copy

fool = dict()
fool["1"] = "one"

lish_1 = fool
lish_2 = copy.deepcopy(fool)

fool["2"] = "two"

print(lish_1)
print(lish_2)
