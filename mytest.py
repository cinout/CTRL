import torchvision.datasets as datasets
from PIL import Image
import torchvision.transforms as transforms
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from methods.MoCoV2.mocov2 import MoCo
import torchvision.models as models

model = MoCo(
    models.__dict__["resnet18"],  # args.arch == "resnet18"
    dim=128,
    K=65536,
    m=0.999,
    contr_tau=0.2,
    align_alpha=2,
    unif_t=3,
    unif_intra_batch=True,
    mlp=True,
)


print(model.encoder_q)
