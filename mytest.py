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
    confetti_noise,
    confetti_poisoning,
    smooth_noise,
)

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
CUTPASTE
"""
cutpaste_normal = CutPasteNormal()
cutpaste_result = cutpaste_normal(img)
cutpaste_result = cutpaste_result * 255.0
cutpaste_result = np.clip(cutpaste_result, 0, 255)
cutpaste_result = np.array(cutpaste_result, dtype=np.uint8)
cutpaste_result = PIL.Image.fromarray(cutpaste_result)
cutpaste_result.save("test_cutpaste.png", "PNG")
exit()

"""
CONFETTI
"""
confetti_poisoned_image = confetti_poisoning(img, image_size)
confetti_poisoned_image = confetti_poisoned_image.squeeze(0)
confetti_poisoned_image = torch.permute(confetti_poisoned_image, (1, 2, 0))

augmented_image = PIL.Image.fromarray(confetti_poisoned_image.numpy())
augmented_image.save("test_confetti.png", "PNG")


"""
DRAEM
"""


def lerp_np(x, y, w):
    fin_out = (y - x) * w + x
    return fin_out


def rand_perlin_2d_np(shape, res, fade=lambda t: 6 * t**5 - 15 * t**4 + 10 * t**3):
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0 : res[0] : delta[0], 0 : res[1] : delta[1]].transpose(1, 2, 0) % 1

    angles = 2 * math.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
    tt = np.repeat(np.repeat(gradients, d[0], axis=0), d[1], axis=1)

    tile_grads = lambda slice1, slice2: np.repeat(
        np.repeat(
            gradients[slice1[0] : slice1[1], slice2[0] : slice2[1]], d[0], axis=0
        ),
        d[1],
        axis=1,
    )
    dot = lambda grad, shift: (
        np.stack(
            (
                grid[: shape[0], : shape[1], 0] + shift[0],
                grid[: shape[0], : shape[1], 1] + shift[1],
            ),
            axis=-1,
        )
        * grad[: shape[0], : shape[1]]
    ).sum(axis=-1)

    n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0])
    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
    n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])
    t = fade(grid[: shape[0], : shape[1]])
    return math.sqrt(2) * lerp_np(
        lerp_np(n00, n10, t[..., 0]), lerp_np(n01, n11, t[..., 0]), t[..., 1]
    )


augmenters = [
    iaa.GammaContrast((0.5, 2.0), per_channel=True),
    iaa.MultiplyAndAddToBrightness(mul=(0.8, 1.2), add=(-30, 30)),
    iaa.pillike.EnhanceSharpness(),
    iaa.AddToHueAndSaturation((-50, 50), per_channel=True),
    iaa.Solarize(0.5, threshold=(32, 128)),
    iaa.Posterize(),
    iaa.Invert(),
    iaa.pillike.Autocontrast(),
    iaa.pillike.Equalize(),
    iaa.Affine(rotate=(-45, 45)),
]
aug_ind = np.random.choice(np.arange(len(augmenters)), 3, replace=False)
aug = iaa.Sequential(
    [
        augmenters[aug_ind[0]],
        augmenters[aug_ind[1]],
        augmenters[aug_ind[2]],
    ]
)

anomaly_source_paths = sorted(glob.glob("./datasets/dtd/images/*/*.jpg"))
# print(anomaly_source_paths)

perlin_scale = 6
min_perlin_scale = 0

anomaly_source_idx = torch.randint(0, len(anomaly_source_paths), (1,)).item()
anomaly_source_path = anomaly_source_paths[anomaly_source_idx]

anomaly_source_img = cv2.imread(anomaly_source_path)
anomaly_source_img = cv2.cvtColor(anomaly_source_img, cv2.COLOR_BGR2RGB)
anomaly_source_img = cv2.resize(anomaly_source_img, dsize=(image_size, image_size))
anomaly_img_augmented = aug(image=anomaly_source_img)
perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
perlin_noise = rand_perlin_2d_np(
    (image_size, image_size), (perlin_scalex, perlin_scaley)
)


perlin_noise = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])(image=perlin_noise)

threshold = 0.5
perlin_thr = np.where(
    perlin_noise > threshold,
    np.ones_like(perlin_noise),
    np.zeros_like(perlin_noise),
)
perlin_thr = np.expand_dims(perlin_thr, axis=2)
img_thr = (
    anomaly_img_augmented.astype(np.float32) * perlin_thr / 255.0
)  # range [0, 1], shape: [32, 32, 3], channels in BGR order
beta = torch.rand(1).numpy()[0] * 0.8


augmented_image = (
    img * (1 - perlin_thr) + (1 - beta) * img_thr + beta * img * (perlin_thr)
)
augmented_image = augmented_image.astype(np.float32)
msk = (perlin_thr).astype(np.float32)
augmented_image = msk * augmented_image + (1 - msk) * img  # shape: [64, 64, 3]

print(augmented_image.shape)

augmented_image = augmented_image * 255.0
augmented_image = np.clip(augmented_image, 0, 255)
augmented_image = np.array(augmented_image, dtype=np.uint8)
augmented_image = PIL.Image.fromarray(augmented_image)
augmented_image.save("test_draem.png", "PNG")
