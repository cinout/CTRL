import torchvision.datasets as datasets
from PIL import Image
import torchvision.transforms as transforms
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


def add_value(item):
    item["new_value"] = 3
    return item


our_obj = dict()
our_obj["first"] = 1

print(our_obj)

our_obj = add_value(our_obj)
print(our_obj)
