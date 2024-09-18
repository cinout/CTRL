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
labels = np.concatenate(
    (
        np.zeros(10),
        np.ones(10),
    ),
    axis=0,
)

labels = torch.tensor(labels, device=device, dtype=torch.long)
