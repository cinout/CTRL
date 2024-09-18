import torchvision.datasets as datasets
from PIL import Image
import torchvision.transforms as transforms
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import copy
from collections import Counter

device = "cuda" if torch.cuda.is_available() else "cpu"

folder_name = "analyze_data"
prefix_options = ["dataset_htba_imagenet100"]
# prefix_options = [
#     "dataset_cifar10",
#     "dataset_cifar100",
#     "dataset_imagenet100",
#     "dataset_htba_imagenet100",
# ]
group_id_options = [1]
# group_id_options = [1,2,3,4,]

for prefix in prefix_options:
    for group_id in group_id_options:
        with open(f"{folder_name}/{prefix}_clean_{group_id}.t", "rb") as f:
            clean_features = torch.load(f, map_location=device)  # [bs, views, 512]
        with open(f"{folder_name}/{prefix}_poi_{group_id}.t", "rb") as f:
            poi_features = torch.load(f, map_location=device)  # [bs, views, 512]
        with open(f"{folder_name}/{prefix}_poi_{group_id}_position.t", "rb") as f:
            poi_position = torch.load(f, map_location=device)  # [bs]

        bs, n_views, C = clean_features.shape

        clean_features = clean_features.reshape(-1, C).detach().cpu().numpy()
        poi_features = poi_features.reshape(-1, C).detach().cpu().numpy()

        _, _, v_clean = np.linalg.svd(
            clean_features - np.mean(clean_features, axis=0, keepdims=True),
            full_matrices=False,
        )
        eig_for_indexing = v_clean[0:1]  # [1, C]
        _, _, v_poi = np.linalg.svd(
            poi_features - np.mean(poi_features, axis=0, keepdims=True),
            full_matrices=False,
        )
        eig_for_indexing = v_poi[0:1]  # [1, C]

        # clean_indices = torch.nonzero(poi_position == 0).flatten()
        # poison_indices = torch.nonzero(poi_position == 1).flatten()

        # clean_at_110 = poi[clean_indices, :, 110]
        # poison_at_110 = poi[poison_indices, :, 110]
