import torchvision.datasets as datasets
from PIL import Image
import torchvision.transforms as transforms
import torch

bs = 8
k = 3
classes = 10

sim_labels = torch.randint(low=0, high=classes - 1, size=(bs, k))

one_hot_label = torch.zeros(bs * k, classes)
one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
one_hot_label = one_hot_label.view(bs, -1, classes)
print(sim_labels)
print(one_hot_label)
