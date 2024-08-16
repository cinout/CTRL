import torchvision.datasets as datasets
from PIL import Image
import torchvision.transforms as transforms
import torch

target_class = 4
bs = 10
data = torch.randint(0, 20, (bs, 5))
print(data)
labels = torch.tensor([11] * bs)


condition = labels != target_class

needed = data[condition]
print(needed)
