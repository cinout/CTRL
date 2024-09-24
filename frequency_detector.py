import torch
import torch.nn as nn
import albumentations
from scipy.fftpack import dct
import numpy as np
import cv2


def dct2(block):
    return dct(dct(block.T, norm="ortho").T, norm="ortho")


def addnoise(img):
    aug = albumentations.GaussNoise(p=1, mean=25, var_limit=(10, 70))
    augmented = aug(image=(img * 255).astype(np.uint8))
    auged = augmented["image"] / 255
    return auged


def randshadow(img, image_size):
    aug = albumentations.RandomShadow(p=1)
    test = (img * 255).astype(np.uint8)
    augmented = aug(image=cv2.resize(test, (image_size, image_size)))
    auged = augmented["image"] / 255
    return auged


def patching_train(
    clean_sample, x_train, image_size, ensemble_id, frequency_train_trigger_size
):
    """
    this code conducts a patching procedure with random white blocks or random noise block
    """
    # FIXME: hack code
    if ensemble_id == 0:
        if frequency_train_trigger_size == 2:
            attack = np.random.choice([0, 1], 1)[0]
        elif frequency_train_trigger_size == 3:
            attack = np.random.choice([0, 1, 2], 1)[0]
        elif frequency_train_trigger_size == 4:
            attack = np.random.choice([0, 1, 2, 3], 1)[0]
        elif frequency_train_trigger_size == 5:
            attack = np.random.choice([0, 1, 2, 3, 4], 1)[0]
    elif ensemble_id == 1:
        if frequency_train_trigger_size == 2:
            attack = np.random.choice([2, 3], 1)[0]
        elif frequency_train_trigger_size == 3:
            attack = np.random.choice([2, 3, 4], 1)[0]
        elif frequency_train_trigger_size == 4:
            attack = np.random.choice([1, 2, 3, 4], 1)[0]
        elif frequency_train_trigger_size == 5:
            attack = np.random.choice([0, 1, 2, 3, 4], 1)[0]
    elif ensemble_id == 2:
        if frequency_train_trigger_size == 2:
            attack = np.random.choice([1, 4], 1)[0]
        elif frequency_train_trigger_size == 3:
            attack = np.random.choice([0, 3, 4], 1)[0]
        elif frequency_train_trigger_size == 4:
            attack = np.random.choice([1, 2, 4, 5], 1)[0]
        elif frequency_train_trigger_size == 5:
            attack = np.random.choice([0, 1, 2, 3, 4], 1)[0]

    pat_size_x = np.random.randint(2, 8)
    pat_size_y = np.random.randint(2, 8)
    output = np.copy(clean_sample)
    # TODO: remove later
    print(f"attack is {attack}")
    if attack == 0:
        block = np.ones((pat_size_x, pat_size_y, 3))
    elif attack == 1:
        block = np.random.rand(pat_size_x, pat_size_y, 3)
    elif attack == 2:
        return addnoise(output)
    elif attack == 3:
        return randshadow(output, image_size)
    if attack == 4:
        randind = np.random.randint(x_train.shape[0])
        tri = x_train[randind]
        mid = output + 0.3 * tri
        mid[mid > 1] = 1
        return mid
    # TODO: remove later
    print(f"block.shape is {block.shape}")

    margin = np.random.randint(0, 6)
    rand_loc = np.random.randint(0, 4)

    print(f"rand_loc is {rand_loc}")
    print("==================")
    if rand_loc == 0:
        output[margin : margin + pat_size_x, margin : margin + pat_size_y, :] = (
            block  # upper left
        )
    elif rand_loc == 1:
        output[
            margin : margin + pat_size_x,
            image_size - margin - pat_size_y : image_size - margin,
            :,
        ] = block
    elif rand_loc == 2:
        output[
            image_size - margin - pat_size_x : image_size - margin,
            margin : margin + pat_size_y,
            :,
        ] = block
    elif rand_loc == 3:
        output[
            image_size - margin - pat_size_x : image_size - margin,
            image_size - margin - pat_size_y : image_size - margin,
            :,
        ] = block  # right bottom

    output[output > 1] = 1
    return output


class FrequencyDetector(nn.Module):
    def __init__(self, height, width):
        super(FrequencyDetector, self).__init__()

        self.num_classes = 2  # poisoned or clean
        self.height = height
        self.width = width
        self.softmax = nn.Softmax(dim=1)

        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ELU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ELU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ELU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ELU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.3),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ELU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ELU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.4),
            nn.Flatten(),
            nn.Linear(128 * (self.height // 8) * (self.width // 8), self.num_classes),
        )

    def forward(self, x):
        x = self.model(x)
        x = self.softmax(x)
        return x
