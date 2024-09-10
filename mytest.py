import torchvision.datasets as datasets
from PIL import Image
import torchvision.transforms as transforms
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import copy

corrs = np.random.random((10, 4))
ss_scores = np.max(corrs, axis=0).tolist()  # [bs]
print(type(ss_scores))
