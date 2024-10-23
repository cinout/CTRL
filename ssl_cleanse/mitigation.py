import os
import PIL
import numpy as np
from torch.optim.lr_scheduler import MultiStepLR
import torchvision.transforms as T
import random
from PIL import ImageFilter, Image
import torchvision.transforms.functional as F
import torch
from torch.utils.data import Dataset


# class AddTrigger:
#     def __init__(self, trigger_path, trigger_width, location_ratio):
#         self.trigger_path = trigger_path
#         self.trigger_width = trigger_width
#         self.location_ratio = location_ratio

#     def __call__(self, x):
#         base_image = x.clone()
#         # this is where error occurs

#         img_trigger = Image.open(self.trigger_path).convert("RGB")

#         width, height = base_image.size(1), base_image.size(2)
#         t_width, t_height = self.trigger_width, int(
#             img_trigger.size[0] * self.trigger_width / img_trigger.size[1]
#         )
#         img_trigger = F.to_tensor(F.resize(img_trigger, [t_width, t_height]))
#         location = (
#             int((width - t_width) * self.location_ratio),
#             int((height - t_height) * self.location_ratio),
#         )
#         base_image[
#             :,
#             location[1] : (location[1] + t_height),
#             location[0] : (location[0] + t_width),
#         ] = img_trigger
#         return base_image


# class TriggerT:
#     def __init__(
#         self,
#         base_transform,
#         mean,
#         std,
#         trigger_path=None,
#         trigger_width=None,
#         trigger_location=None,
#     ):
#         if (
#             trigger_path is not None
#             and trigger_width is not None
#             and trigger_location is not None
#         ):
#             self.trigger_transform = T.Compose(
#                 [
#                     base_transform,
#                     AddTrigger(trigger_path, trigger_width, trigger_location),
#                     T.Normalize(mean=mean, std=std),
#                 ]
#             )
#         else:
#             self.trigger_transform = T.Compose(
#                 [base_transform, T.Normalize(mean=mean, std=std)]
#             )

#     def __call__(self, x):
#         return self.trigger_transform(x)


def get_scheduler(args, optimizer):
    m = [args.mitigate_epoches - a for a in args.drop]
    return MultiStepLR(optimizer, milestones=m, gamma=args.drop_gamma)


def aug_transform(args):
    """augmentation transform generated from config"""
    return T.Compose(
        [
            T.RandomResizedCrop(
                args.image_size,
                scale=(args.crop_s0, args.crop_s1),
                ratio=(args.crop_r0, args.crop_r1),
                interpolation=T.InterpolationMode.BICUBIC,
            ),
            T.RandomApply(
                [T.ColorJitter(args.cj0, args.cj1, args.cj2, args.cj3)], p=args.cj_p
            ),
            T.RandomGrayscale(p=args.gs_p),
            T.RandomApply([RandomBlur(0.1, 2.0)], p=0.5),
            T.RandomHorizontalFlip(p=args.hf_p),
            T.ToTensor(),
        ]
    )


class RandomBlur:
    def __init__(self, r0, r1):
        self.r0, self.r1 = r0, r1

    def __call__(self, image):
        r = random.uniform(self.r0, self.r1)
        return image.filter(ImageFilter.GaussianBlur(radius=r))


class FileListDataset(Dataset):
    def __init__(
        self,
        args,
        trainset_data,  # a tuple of (x_untransformed, y), where y is cluster id, x_untransformed values are in [0,1]
    ):
        # aug_with_blur = aug_transform(args)

        self.basic_transform = T.Compose(
            [aug_transform(args), T.Normalize(mean=args.mean, std=args.std)]
        )
        self.num_clusters = args.num_clusters

        # self.triggers = []
        # for target in range(args.num_clusters):
        #     trigger_path = os.path.join(args.trigger_path, f"{target}.pth")
        #     trigger = torch.load(
        #         trigger_path, map_location=device
        #     )  # I guess it is {"mask": xxx, "delta": xxx}
        #     self.triggers.append(
        #         {"mask": trigger["mask"].detach(), "delta": trigger["delta"].detach()}
        #     )

        # self.t_0 = TriggerT(base_transform=aug_with_blur, mean=args.mean, std=args.std)
        # self.t_1 = TriggerT(
        #     base_transform=aug_with_blur,
        #     mean=args.mean,
        #     std=args.std,
        #     trigger_path=args.trigger_path,
        #     trigger_width=args.trigger_width,
        #     trigger_location=args.trigger_location,
        # )

        self.image_list = trainset_data[
            0
        ]  # [#total=1% trainset, 3, image_size, image_size], values in [0,1]

        self.cluster_list = trainset_data[1]  # [#total=1% trainset]

    def __getitem__(self, idx):

        image = self.image_list[idx]
        image = torch.permute(image, (1, 2, 0))
        image = image * 255.0
        image = torch.clamp(image, 0, 255)
        image = np.array(image.cpu(), dtype=np.uint8)
        image = PIL.Image.fromarray(image)  # PIL format
        clean_view_1 = self.basic_transform(image)  # tensor, [3, img_size, img_size]
        clean_view_2 = self.basic_transform(image)  # tensor, [3, img_size, img_size]
        clean_view_3 = self.basic_transform(image)  # tensor

        cluster_id = self.cluster_list[idx]
        valid_trigger_indices = [
            index for index in range(self.num_clusters) if index != cluster_id
        ]
        trigger_index = random.choice(valid_trigger_indices)

        # trigger = self.triggers[trigger_index]
        # mask, delta = trigger["mask"], trigger["delta"]

        # trigger_view = torch.mul(clean_view_3.unsqueeze(0), 1 - mask) + torch.mul(
        #     delta, mask
        # )  # [1, 3, img_size, img_size]
        # trigger_view = trigger_view.squeeze(0)  # [3, img_size, img_size]

        return clean_view_1, clean_view_2, clean_view_3, trigger_index

    def __len__(self):
        return self.cluster_list.shape[0]


def ds_train(args, trainset_data):

    return FileListDataset(args, trainset_data)
