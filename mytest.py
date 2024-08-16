import torchvision.datasets as datasets
from PIL import Image
import torchvision.transforms as transforms
import torch

target_class = 4
bs = 10
data = torch.randint(0, 20, (bs, 5))
print(data)
data = data[torch.tensor([3, 1, 5])]
print(data)
