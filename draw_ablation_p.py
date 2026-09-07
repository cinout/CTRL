[43.2, 43.1, 87.2, 88.2, 56.5, 57.4]
[41.4, 41.3, 86.9, 86.5, 54.3, 54.5]
[41.3, 41.3, 86.9, 86.9, 54.3, 54.7]
[41.4, 41.4, 86.8, 86.9, 54.4, 54.9]
[41.3, 41.4, 86.9, 87.3, 54.4, 54.8]
[41.2, 41.5, 86.9, 87.3, 54.3, 54.9]
[41.4, 41.6, 86.8, 87.6, 54.3, 54.8]
[41.2, 41.6, 86.6, 87.5, 54.5, 54.6]
[41.6, 41.7, 86.6, 87.7, 54.3, 55.0]
[41.8, 41.6, 86.4, 87.6, 54.0, 54.8]

[42.9, 42.6, 78.5, 79.6, 46.0, 46.2]
[41.6, 41.3, 77.9, 78.6, 45.7, 44.8]
[41.6, 41.5, 77.8, 78.6, 45.7, 44.8]
[41.7, 41.5, 77.8, 78.6, 45.7, 44.9]
[41.6, 41.6, 77.9, 78.9, 45.6, 44.9]
[41.6, 41.7, 77.8, 78.9, 45.6, 45.0]
[41.7, 41.9, 78.0, 79.0, 45.7, 45.0]
[42.0, 41.8, 78.1, 79.0, 45.6, 45.2]
[41.9, 42.0, 78.0, 79.0, 45.4, 45.3]
[41.7, 41.9, 78.0, 78.9, 45.2, 45.1]

[38.3, 39.1, 85.2, 85.3, 49.7, 50.5]
[36.9, 37.0, 81.3, 80.1, 47.3, 48.3]
[37.0, 37.1, 81.9, 80.2, 47.5, 48.3]
[37.2, 37.1, 81.9, 80.4, 47.5, 48.2]
[37.1, 37.1, 82.0, 80.9, 47.5, 48.2]
[37.0, 36.9, 81.9, 81.3, 47.5, 48.2]
[37.0, 37.0, 82.5, 81.4, 47.5, 48.0]
[36.7, 37.0, 82.8, 82.5, 47.5, 48.1]
[36.7, 37.2, 81.9, 82.5, 47.6, 47.9]
[37.0, 37.3, 81.6, 82.2, 47.0, 47.6]


# precision of 8 poisoned images P

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

#  [ImageNet100+HTBA, ImageNet100+FTrojan, CIFAR10+HTBA, CIFAR10+FTrojan, CIFAR100+HTBA, CIFAR100+FTrojan]

dataset_names = [
    "ImageNet100+HTBA",
    "ImageNet100+FTrojan",
    "CIFAR10+HTBA",
    "CIFAR10+FTrojan",
    "CIFAR100+HTBA",
    "CIFAR100+FTrojan",
]

x_labels = ["-", "8", "7", "6", "5", "4", "3", "2", "1", "0"]
x_values = list(range(len(x_labels)))

method_values = {
    "BYOL": {
        "ImageNet100+HTBA": [2.7, 0.4, 0.4, 0.4, 0.4, 0.5, 0.8, 2.7, 11.6, 1.1],
        "ImageNet100+FTrojan": [42.0, 0.6, 0.6, 0.7, 0.8, 1.1, 1.7, 15.8, 46.6, 48.2],
        "CIFAR10+HTBA": [4.3, 3.7, 3.8, 4.2, 3.5, 4.8, 5.8, 6.0, 8.4, 7.6],
        "CIFAR10+FTrojan": [83.2, 30.6, 35.0, 37.6, 41.9, 47.0, 49.2, 55.2, 64.6, 87.9],
        "CIFAR100+HTBA": [45.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 9.3, 41.3],
        "CIFAR100+FTrojan": [87.9, 0.0, 0.0, 0.0, 0.1, 0.2, 7.4, 15.7, 83.9, 89.7],
    },
    "MoCoV2": {
        "ImageNet100+HTBA": [6.4, 2.0, 2.0, 2.5, 3.1, 3.6, 3.9, 6.2, 10.3, 13.8],
        "ImageNet100+FTrojan": [43.5, 0.8, 0.8, 1.1, 1.3, 1.6, 2.3, 7.1, 36.7, 43.9],
        "CIFAR10+HTBA": [82.4, 52.3, 50.5, 54.0, 54.3, 57.3, 62.3, 64.3, 80.3, 83.5],
        "CIFAR10+FTrojan": [1.9, 1.2, 1.2, 1.4, 1.5, 1.6, 1.8, 1.9, 2.0, 2.2],
        "CIFAR100+HTBA": [41.3, 3.3, 2.3, 2.6, 3.0, 3.1, 7.8, 10.9, 20.2, 36.8],
        "CIFAR100+FTrojan": [0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.3, 0.5],
    },
    "SimCLR": {
        "ImageNet100+HTBA": [42.1, 1.1, 0.8, 1.0, 1.3, 1.7, 1.9, 2.3, 25.5, 47.1],
        "ImageNet100+FTrojan": [34.1, 0.2, 0.3, 0.3, 0.4, 0.5, 0.8, 9.2, 28.5, 38.4],
        "CIFAR10+HTBA": [84.5, 0.5, 0.8, 1.7, 2.0, 4.0, 13.8, 16.4, 53.5, 88.0],
        "CIFAR10+FTrojan": [81.8, 23.2, 20.7, 24.1, 28.3, 34.8, 37.3, 56.2, 77.3, 86.5],
        "CIFAR100+HTBA": [79.6, 0.0, 0.0, 0.0, 0.1, 0.1, 0.1, 0.1, 22.0, 78.9],
        "CIFAR100+FTrojan": [70.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.6, 27.7, 69.2],
    },
}

fig, axes = plt.subplots(3, 6, figsize=(24, 12), squeeze=False)
# fig.suptitle("ASR trend across poisoned sample precision P", fontsize=16, y=0.98)

for row_idx, method in enumerate(["BYOL", "MoCoV2", "SimCLR"]):
    for col_idx, dataset_name in enumerate(dataset_names):
        ax = axes[row_idx, col_idx]
        y = method_values[method][dataset_name]
        ax.plot(x_values, y, marker="o", linewidth=2, markersize=5, color="coral")
        ax.set_title(f"{method}+{dataset_name}", fontsize=18)
        ax.set_xticks(x_values)
        ax.set_xticklabels(x_labels, rotation=20, fontsize=14)
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.4)
        ax.set_ylim(bottom=0)

plt.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("ablation_p_asr_trend.png", dpi=300)
plt.close(fig)
