import torchvision.datasets as datasets
from PIL import Image
import torchvision.transforms as transforms
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import copy
from collections import Counter
import random

arr_1 = np.array([9, 21, 2, 10, 4, 3, 11, 20, 7])
arr_2 = np.array([7, 2, 21, 20])


arr_1 = np.array([index for index in arr_1 if index in arr_2])

print(arr_1.shape[0])
