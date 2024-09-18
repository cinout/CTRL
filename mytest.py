import torchvision.datasets as datasets
from PIL import Image
import torchvision.transforms as transforms
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import copy
from collections import Counter
import random

labels = (np.concatenate((np.zeros(5), np.ones(5)), axis=0)).astype(np.uint)
print(labels)

idx = np.arange(10)
random.shuffle(idx)
print(idx)
labels = labels[idx]
print(labels)
