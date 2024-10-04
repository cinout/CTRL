import glob
import PIL
import torchvision.datasets as datasets
from PIL import Image
import torchvision.transforms as transforms
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import copy
from collections import Counter
import random, math
import torch.nn.functional as F
import cv2

import imgaug.augmenters as iaa

from frequency_detector import (
    CutPasteNormal,
    addnoise,
    confetti_noise,
    confetti_poisoning,
    defocus,
    draem_augment,
    pixel_dropout,
    posterize,
    rand_rain,
    rand_sunflare,
    randshadow,
    smooth_noise,
    spatter_mud,
    spatter_rain,
)

image_size = 32
img = Image.open(
    "/Users/haitianh/Downloads/Code/_datasets/Imagenet100/val/n02087046/ILSVRC2012_val_00014912.jpg"
).convert("RGB")
img = img.resize((image_size, image_size))
img = np.asarray(img).astype(np.float32) / 255.0
img = torch.tensor(img)
img = np.array(
    img.cpu(), dtype=np.float32
)  # shape: [32, 32, 3]; value range: [0, 1], channels in RGB order


"""
Guassian Noise
"""
augmented = addnoise(img)
augmented = augmented * 255.0
augmented = np.clip(augmented, 0, 255)
augmented = np.array(augmented, dtype=np.uint8)
augmented = PIL.Image.fromarray(augmented)
augmented.save("test_gaussiannoise.png", "PNG")
exit()

"""
Shadow
"""
augmented = randshadow(img, image_size)
augmented = augmented * 255.0
augmented = np.clip(augmented, 0, 255)
augmented = np.array(augmented, dtype=np.uint8)
augmented = PIL.Image.fromarray(augmented)
augmented.save("test_shadow.png", "PNG")
exit()

"""
CUTPASTE
"""
cutpaste_normal = CutPasteNormal()
augmented = cutpaste_normal(img)
augmented = augmented * 255.0
augmented = np.clip(augmented, 0, 255)
augmented = np.array(augmented, dtype=np.uint8)
augmented = PIL.Image.fromarray(augmented)
augmented.save("test_cutpaste.png", "PNG")
exit()

"""
DRAEM
"""
augmented = draem_augment(img, image_size)
augmented = augmented * 255.0
augmented = np.clip(augmented, 0, 255)
augmented = np.array(augmented, dtype=np.uint8)
augmented = PIL.Image.fromarray(augmented)
augmented.save("test_draem.png", "PNG")
exit()

"""
CONFETTI
"""
augmented = confetti_poisoning(img, image_size)
augmented = augmented * 255.0
augmented = np.clip(augmented, 0, 255)
augmented = np.array(augmented, dtype=np.uint8)
augmented = PIL.Image.fromarray(augmented)
augmented.save("test_confetti.png", "PNG")
exit()

"""
rain
"""
augmented = rand_rain(img, image_size)
augmented = augmented * 255.0
augmented = np.clip(augmented, 0, 255)
augmented = np.array(augmented, dtype=np.uint8)
augmented = PIL.Image.fromarray(augmented)
augmented.save("test_randrain.png", "PNG")
exit()

"""
rand_sunflare
"""
augmented = rand_sunflare(img, image_size)
augmented = augmented * 255.0
augmented = np.clip(augmented, 0, 255)
augmented = np.array(augmented, dtype=np.uint8)
augmented = PIL.Image.fromarray(augmented)
augmented.save("test_randsunflare.png", "PNG")
exit()

"""
Spatter Mud
"""
augmented = spatter_mud(img)
augmented = augmented * 255.0
augmented = np.clip(augmented, 0, 255)
augmented = np.array(augmented, dtype=np.uint8)
augmented = PIL.Image.fromarray(augmented)
augmented.save("test_spattermud.png", "PNG")
exit()


"""
posterize
"""
augmented = posterize(img)
augmented = augmented * 255.0
augmented = np.clip(augmented, 0, 255)
augmented = np.array(augmented, dtype=np.uint8)
augmented = PIL.Image.fromarray(augmented)
augmented.save("test_posterize.png", "PNG")
exit()


"""
PIXEL DROPOUT
"""
augmented = pixel_dropout(img)
augmented = augmented * 255.0
augmented = np.clip(augmented, 0, 255)
augmented = np.array(augmented, dtype=np.uint8)
augmented = PIL.Image.fromarray(augmented)
augmented.save("test_pixeldropout.png", "PNG")
exit()


"""
Spatter Rain
"""
augmented = spatter_rain(img)
augmented = augmented * 255.0
augmented = np.clip(augmented, 0, 255)
augmented = np.array(augmented, dtype=np.uint8)
augmented = PIL.Image.fromarray(augmented)
augmented.save("test_spatterrain.png", "PNG")
exit()


"""
Defocus
"""
augmented = defocus(img)
augmented = augmented * 255.0
augmented = np.clip(augmented, 0, 255)
augmented = np.array(augmented, dtype=np.uint8)
augmented = PIL.Image.fromarray(augmented)
augmented.save("test_defocus.png", "PNG")
exit()
