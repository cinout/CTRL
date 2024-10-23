import glob
import os
import PIL
import torchvision.datasets as datasets
from PIL import Image
import torchvision.transforms as transforms
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Subset
import copy
from collections import Counter
import random, math
import torch.nn.functional as F
import cv2
from sklearn.cluster import KMeans
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
from methods.base import get_pairwise_distance

device = "cuda" if torch.cuda.is_available() else "cpu"

triggers = torch.load(
    os.path.join("trigger_estimation_DEBUG", f"0.pth"), map_location=device
)
print(triggers["mask"].detach())


exit()


image_size = 64
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
BLEND
"""
blend_img_1 = Image.open(
    "/Users/haitianh/Downloads/Code/_datasets/Imagenet100/val/n03259280/ILSVRC2012_val_00009837.jpg"
).convert("RGB")
blend_img_1 = blend_img_1.resize((image_size, image_size))
blend_img_1 = np.asarray(blend_img_1).astype(np.float32) / 255.0
blend_img_1 = torch.tensor(blend_img_1)
blend_img_1 = np.array(blend_img_1.cpu(), dtype=np.float32)

blend_img_2 = Image.open(
    "/Users/haitianh/Downloads/Code/_datasets/Imagenet100/val/n03594734/ILSVRC2012_val_00019687.jpg"
).convert("RGB")
blend_img_2 = blend_img_2.resize((image_size, image_size))
blend_img_2 = np.asarray(blend_img_2).astype(np.float32) / 255.0
blend_img_2 = torch.tensor(blend_img_2)
blend_img_2 = np.array(blend_img_2.cpu(), dtype=np.float32)

blend_1 = np.copy(img) + 0.3 * blend_img_1
blend_1[blend_1 > 1] = 1
blend_1 = blend_1 * 255.0
blend_1 = np.clip(blend_1, 0, 255)
blend_1 = np.array(blend_1, dtype=np.uint8)
blend_1 = PIL.Image.fromarray(blend_1)
blend_1.save("test_blend1.png", "PNG")

blend_2 = np.copy(img) + 0.3 * blend_img_2
blend_2[blend_2 > 1] = 1
blend_2 = blend_2 * 255.0
blend_2 = np.clip(blend_2, 0, 255)
blend_2 = np.array(blend_2, dtype=np.uint8)
blend_2 = PIL.Image.fromarray(blend_2)
blend_2.save("test_blend2.png", "PNG")
exit()

"""
White Box
"""
pat_size_x = np.random.randint(2, 8)
pat_size_y = np.random.randint(2, 8)
block = np.random.rand(pat_size_x, pat_size_y, 3)
# block = np.ones((pat_size_x, pat_size_y, 3))
margin = np.random.randint(0, 6)
rand_loc = np.random.randint(0, 4)

if rand_loc == 0:
    img[margin : margin + pat_size_x, margin : margin + pat_size_y, :] = (
        block  # upper left
    )
elif rand_loc == 1:
    img[
        margin : margin + pat_size_x,
        image_size - margin - pat_size_y : image_size - margin,
        :,
    ] = block
elif rand_loc == 2:
    img[
        image_size - margin - pat_size_x : image_size - margin,
        margin : margin + pat_size_y,
        :,
    ] = block
elif rand_loc == 3:
    img[
        image_size - margin - pat_size_x : image_size - margin,
        image_size - margin - pat_size_y : image_size - margin,
        :,
    ] = block  # right bottom
img = img * 255.0
img = np.clip(img, 0, 255)
img = np.array(img, dtype=np.uint8)
img = PIL.Image.fromarray(img)
img.save("test_colorbox.png", "PNG")
exit()

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
