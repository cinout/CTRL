from torch.utils.data import DataLoader, TensorDataset
import random
import numpy as np
from PIL import Image
from torch import Tensor
from typing import Callable
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from kornia import augmentation as aug


device = "cuda" if torch.cuda.is_available() else "cpu"


def get_data_and_label(paths, size):
    images = []
    targets = []
    for i, item in enumerate(paths):
        if i % 1000 == 0:
            print(f"transform data to {i}/{len(paths)}")
        img = Image.open(item.split()[0]).convert("RGB")
        img = img.resize((size, size))

        img = np.asarray(img).astype(np.float32) / 255.0
        img = torch.tensor(img)
        img = torch.permute(img, (2, 0, 1))  # shape: [c=3, h, w], value: [0, 1]
        images.append(img)

        target = int(item.split()[1])
        target = torch.tensor(target, dtype=torch.long)
        targets.append(target)

    return images, targets


class NCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n):
        self.base_transform = base_transform
        self.n = n

    def __call__(self, x):
        aug_image = []

        for _ in range(self.n):
            aug_image.append(self.base_transform(x))

        return aug_image


class PoisonAgent:
    def __init__(
        self,
        args,
        fre_agent,
        trainset,
        validset,
        memory_loader,
        magnitude_train,
        magnitude_val,
    ):
        self.args = args
        self.trainset = (
            trainset  # third 3rd element returned by set_aug_diff(), train_dataset
        )
        self.validset = (
            validset  # seventh 7th element returned by set_aug_diff(), test_dataset
        )
        self.memory_loader = (
            memory_loader  # eighth 8th element returned by set_aug_diff()
        )
        self.poison_num = int(
            len(trainset) * self.args.poison_ratio
        )  #  determine how many to be poisoned
        self.fre_poison_agent = fre_agent  # who does the poisoning work

        self.magnitude_train = magnitude_train
        self.magnitude_val = magnitude_val

        if self.args.detect_trigger_channels:
            ss_views_aug = [
                transforms.RandomResizedCrop(
                    self.args.image_size,
                    scale=(self.args.rrc_scale_min, self.args.rrc_scale_max),
                    ratio=(0.2, 5),
                ),
                transforms.RandomPerspective(p=0.5),
            ]
            self.ss_transform = NCropsTransform(
                transforms.Compose(ss_views_aug), self.args.num_views
            )

        print(
            f"Initializing Poison data (chosen images, examples, sources, labels) with random seed {self.args.seed}"
        )

        (
            self.train_pos_loader,
            self.test_loader,
            self.test_pos_loader,
            self.memory_loader,
            self.train_probe_loader,
        ) = self.choose_poisons_randomly()

    def generate_view_tensors(self, input):
        # input.shape: [total, 3, 32, 32]; value range: [0, 1]
        input = torch.permute(input, (0, 2, 3, 1))
        input = input * 255.0
        input = torch.clamp(input, 0, 255)
        input = np.array(
            input.cpu(), dtype=np.uint8
        )  # shape: [total, 32, 32, 3]; value range: [0, 255]

        view_tensors = []
        for img in input:
            img = Image.fromarray(img)  # in PIL format now
            views = self.ss_transform(
                img
            )  # a list of args.num_views elements, each one is a PIL image

            tensors_of_an_image = []
            for view in views:
                view = np.asarray(view).astype(np.float32) / 255.0
                view = torch.tensor(view)
                view = torch.permute(
                    view, (2, 0, 1)
                )  # shape: [c=3, h, w], value: [0, 1]
                tensors_of_an_image.append(view)
            tensors_of_an_image = torch.stack(
                tensors_of_an_image, dim=0
            )  # [num_views, c, h, w]
            view_tensors.append(tensors_of_an_image)

        view_tensors = torch.stack(view_tensors, dim=0)  # [total, num_views, c, h, w]

        return view_tensors

    def choose_poisons_randomly(self):

        # construct class prototype for each class

        """
        basic data manipulation
        """
        if self.args.dataset == "imagenet100":
            train_paths = self.trainset
            val_paths = self.validset

            print("transform training data")

            if self.args.load_cached_tensors:
                with open(f"x_train_tensor_{self.args.dataset}.t", "rb") as f:
                    x_train_tensor = torch.load(f, map_location=device)
                with open(f"y_train_tensor_{self.args.dataset}.t", "rb") as f:
                    y_train_tensor = torch.load(f, map_location=device)
            else:
                x_train_tensor, y_train_tensor = get_data_and_label(
                    train_paths, self.args.image_size
                )
                x_train_tensor = torch.stack(x_train_tensor)
                y_train_tensor = torch.stack(y_train_tensor)
                with open(f"x_train_tensor_{self.args.dataset}.t", "wb") as f:
                    torch.save(x_train_tensor, f)
                with open(f"y_train_tensor_{self.args.dataset}.t", "wb") as f:
                    torch.save(y_train_tensor, f)

            print("transform validation data")
            if self.args.load_cached_tensors:
                with open(f"x_test_tensor_{self.args.dataset}.t", "rb") as f:
                    x_test_tensor = torch.load(f, map_location=device)
                with open(f"y_test_tensor_{self.args.dataset}.t", "rb") as f:
                    y_test_tensor = torch.load(f, map_location=device)
            else:
                x_test_tensor, y_test_tensor = get_data_and_label(
                    val_paths, self.args.image_size
                )
                x_test_tensor = torch.stack(x_test_tensor)
                y_test_tensor = torch.stack(y_test_tensor)
                with open(f"x_test_tensor_{self.args.dataset}.t", "wb") as f:
                    torch.save(x_test_tensor, f)
                with open(f"y_test_tensor_{self.args.dataset}.t", "wb") as f:
                    torch.save(y_test_tensor, f)

            # memory
            x_memory_tensor = x_train_tensor.clone().detach()
            y_memory_tensor = y_train_tensor.clone().detach()

        else:
            # CIFAR-10/100

            # get image data
            x_train_np, x_test_np = (
                self.trainset.data.astype(np.float32)
                / 255.0,  # .data returns numpy array; value range: 0-254; shape: [50000, 32, 32, 3]
                self.validset.data.astype(np.float32) / 255.0,
            )
            x_memory_np = self.memory_loader.dataset.data.astype(np.float32) / 255.0

            # get labels
            y_train_np, y_test_np = np.array(self.trainset.targets), np.array(
                self.validset.targets
            )
            y_memory_np = np.array(self.memory_loader.dataset.targets)

            # turn from np to torch tensor [keep all y]
            x_train_tensor, y_train_tensor = torch.tensor(x_train_np), torch.tensor(
                y_train_np, dtype=torch.long
            )
            x_test_tensor, y_test_tensor = torch.tensor(x_test_np), torch.tensor(
                y_test_np, dtype=torch.long
            )
            x_memory_tensor = torch.tensor(x_memory_np)
            y_memory_tensor = torch.tensor(y_memory_np, dtype=torch.long)

            # shift image data into [bs, c=3, h, w] shape, [update all x_]
            x_train_tensor = x_train_tensor.permute(
                0, 3, 1, 2
            )  # shape: [50000, 3, 32, 32]; value range: [0, 1]
            x_test_tensor = x_test_tensor.permute(0, 3, 1, 2)
            x_memory_tensor = x_memory_tensor.permute(0, 3, 1, 2)

        """
        # POISONed Validation Set
        """
        # test set (poison all images)
        x_test_pos_tensor, y_test_pos_tensor = (
            self.fre_poison_agent.Poison_Frequency_Diff(
                x_test_tensor.clone().detach(),
                y_test_tensor.clone().detach(),
                self.magnitude_val,
            )
        )
        # why? is it because above code does not assign correct label to poisoned images?
        # [YES], the Poison_Frequency_Diff() function only poisons image data, but does not pollute label.
        y_test_pos_tensor = (
            torch.ones_like(y_test_pos_tensor, dtype=torch.long)
            * self.args.target_class
        )

        """
        # POISONed Train Set (for stage 1 attack)
        """
        poison_index = torch.where(y_train_tensor == self.args.target_class)[0]
        poison_index = poison_index[: self.poison_num]

        # train set (poison only a portion of train images)
        x_train_tensor[poison_index], y_train_tensor[poison_index] = (
            self.fre_poison_agent.Poison_Frequency_Diff(
                x_train_tensor[poison_index],
                y_train_tensor[poison_index],
                self.magnitude_train,
            )
        )

        train_index = torch.tensor(list(range(len(self.trainset))), dtype=torch.long)
        test_index = torch.tensor(list(range(len(self.validset))), dtype=torch.long)
        memory_index = torch.tensor(list(range(len(x_memory_tensor))), dtype=torch.long)

        """
        Create dataloaders
        """
        # contain both CLEAN and a portion of POISONED images
        train_loader = DataLoader(
            TensorDataset(x_train_tensor, y_train_tensor, train_index),
            batch_size=self.args.batch_size,
            sampler=None,
            shuffle=True,
            drop_last=True,
        )

        if self.args.detect_trigger_channels:
            view_tensors = self.generate_view_tensors(x_test_tensor)

        # clean validation set (used in knn eval only, in base.py)
        test_loader = DataLoader(
            (
                TensorDataset(x_test_tensor, view_tensors, y_test_tensor, test_index)
                if self.args.detect_trigger_channels
                else TensorDataset(x_test_tensor, y_test_tensor, test_index)
            ),
            batch_size=(
                self.args.linear_probe_batch_size
                if self.args.use_linear_probing
                else self.args.eval_batch_size
            ),
            shuffle=False,
            drop_last=False,
        )

        if self.args.detect_trigger_channels:
            view_tensors = self.generate_view_tensors(x_test_pos_tensor)

        # poisoned validation set (used in knn eval only, in base.py)
        test_pos_loader = DataLoader(
            (
                TensorDataset(
                    x_test_pos_tensor,
                    view_tensors,
                    y_test_pos_tensor,
                    y_test_tensor,
                    test_index,
                )
                if self.args.detect_trigger_channels
                else TensorDataset(
                    x_test_pos_tensor, y_test_pos_tensor, y_test_tensor, test_index
                )
            ),  # y_test_tensor serves as the original label tensor (for correcting ASR)
            batch_size=(
                self.args.linear_probe_batch_size
                if self.args.use_linear_probing
                else self.args.eval_batch_size
            ),
            shuffle=False,
        )

        # memory set is never poisoned (used in knn eval only, in base.py)
        memory_loader = DataLoader(
            TensorDataset(x_memory_tensor, y_memory_tensor, memory_index),
            batch_size=(
                self.args.linear_probe_batch_size
                if self.args.use_linear_probing
                else self.args.eval_batch_size
            ),
            shuffle=False,
        )

        if self.args.use_linear_probing:
            # create 1% train set for classifier training
            percent = 0.01
            id_and_label = dict()
            for i, label in enumerate(y_memory_tensor.cpu().detach().numpy()):
                if label in id_and_label.keys():
                    id_and_label[label].append(i)
                else:
                    id_and_label[label] = [i]

            x_probe_tensor = []
            y_probe_tensor = []
            for label, indices in id_and_label.items():

                # for each label (class)
                random.shuffle(indices)
                indices = torch.tensor(indices[: int(len(indices) * percent)])

                x_probe_tensor.append(x_memory_tensor[indices])
                y_probe_tensor.append(y_memory_tensor[indices])
            x_probe_tensor = torch.cat(x_probe_tensor, dim=0)
            y_probe_tensor = torch.cat(y_probe_tensor, dim=0)
            probe_index = torch.tensor(
                list(range(len(x_probe_tensor))), dtype=torch.long
            )

            if self.args.detect_trigger_channels:
                view_tensors = self.generate_view_tensors(x_probe_tensor)
            train_probe_loader = DataLoader(
                (
                    TensorDataset(
                        x_probe_tensor, view_tensors, y_probe_tensor, probe_index
                    )
                    if self.args.detect_trigger_channels
                    else TensorDataset(x_probe_tensor, y_probe_tensor, probe_index)
                ),
                batch_size=self.args.linear_probe_batch_size,
                shuffle=True,
            )

            return (
                train_loader,
                test_loader,
                test_pos_loader,
                memory_loader,
                train_probe_loader,
            )
        else:
            return train_loader, test_loader, test_pos_loader, memory_loader, None


class RandomApply(nn.Module):
    def __init__(self, fn: Callable, p: float):
        super().__init__()
        self.fn = fn
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        return x if random.random() > self.p else self.fn(x)


def set_aug_diff(args):
    if args.dataset == "cifar10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        args.num_classes = 10
        args.save_freq = 100

    elif args.dataset == "cifar100":
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
        args.num_classes = 100
        args.save_freq = 100

    # used for imagenet100
    elif args.dataset == "imagenet100":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        args.save_freq = 100
        args.num_classes = 100
    else:
        raise ValueError(args.dataset)

    normalize = aug.Normalize(mean=mean, std=std)

    # this is applied during training, not during poison generation
    train_transform = nn.Sequential(
        aug.RandomResizedCrop(
            size=(args.image_size, args.image_size), scale=(0.2, 1.0)
        ),
        aug.RandomHorizontalFlip(),
        RandomApply(aug.ColorJitter(0.4, 0.4, 0.4, 0.1), p=0.8),
        aug.RandomGrayscale(p=0.2),
        normalize,
    )

    ####################### Define Datasets #######################
    if args.dataset == "cifar10":

        train_dataset = datasets.CIFAR10(
            root=args.data_path, train=True, transform=None, download=True
        )

        test_dataset = datasets.CIFAR10(
            root=args.data_path, train=False, transform=None, download=True
        )
        memory_dataset = datasets.CIFAR10(
            root=args.data_path, train=True, transform=None, download=False
        )

    elif args.dataset == "cifar100":
        train_dataset = datasets.CIFAR100(
            root=args.data_path, train=True, transform=None, download=True
        )

        test_dataset = datasets.CIFAR100(
            root=args.data_path, train=False, transform=None, download=True
        )
        memory_dataset = datasets.CIFAR100(
            root=args.data_path, train=True, transform=None, download=False
        )
    elif args.dataset == "imagenet100":
        train_file_path = "./datasets/imagenet100_train_clean_filelist.txt"
        val_file_path = "./datasets/imagenet100_val_clean_filelist.txt"
        with open(train_file_path, "r") as f:
            train_file_list = f.readlines()
            train_file_list = [row.rstrip() for row in train_file_list]
        f.close()
        with open(val_file_path, "r") as f:
            val_file_list = f.readlines()
            val_file_list = [row.rstrip() for row in val_file_list]
        f.close()

        train_dataset = train_file_list
        memory_dataset = train_file_list
        test_dataset = val_file_list
    else:
        raise NotImplementedError

    memory_loader = torch.utils.data.DataLoader(
        memory_dataset,
        args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    return (
        train_dataset,  # used as PoisonAgent's train_dataset
        test_dataset,  # used as PoisonAgent's val_dataset
        memory_loader,  # used as PoisonAgent's memory_loader
        train_transform,  # used in train_loader iteration, not in poisoning
    )
