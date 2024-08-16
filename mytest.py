import torchvision.datasets as datasets
from PIL import Image
import torchvision.transforms as transforms
import torch
import numpy as np

y_memory_tensor = torch.tensor(np.array([3, 6, 9]), dtype=torch.long)
for i, label in enumerate(y_memory_tensor):
    print(i)
    print(label)
    print("========")
