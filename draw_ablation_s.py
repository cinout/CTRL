[43.2, 43.1, 87.2, 88.2, 56.5, 57.4]
[42.1, 41.7, 86.8, 86.1, 54.8, 55.0]
[41.5, 41.3, 86.8, 87.1, 54.3, 55.0]
[41.4, 41.4, 87.0, 86.8, 54.5, 54.8]
[41.4, 41.3, 87.0, 87.0, 54.5, 54.7]
[41.4, 41.4, 87.0, 86.9, 54.5, 54.8]
[41.3, 41.3, 86.9, 87.0, 54.3, 54.7]

[42.9, 42.6, 78.5, 79.6, 46.0, 46.2]
[42.1, 41.8, 78.0, 78.8, 45.7, 45.0]
[41.8, 41.6, 77.8, 78.7, 45.6, 44.8]
[41.8, 41.5, 77.8, 78.6, 45.7, 44.9]
[41.7, 41.4, 77.9, 78.5, 45.7, 44.8]
[41.6, 41.5, 77.8, 78.5, 45.6, 44.8]
[41.6, 41.5, 77.9, 78.5, 45.7, 44.9]

[38.3, 39.1, 85.2, 85.3, 49.7, 50.5]
[37.6, 37.5, 83.7, 81.8, 48.2, 48.8]
[37.0, 37.1, 82.1, 81.3, 47.8, 48.5]
[37.1, 37.2, 81.6, 80.5, 47.5, 48.4]
[37.1, 37.1, 81.8, 80.8, 47.4, 48.4]
[37.0, 37.0, 81.9, 80.4, 47.3, 48.3]
[36.9, 36.9, 81.8, 80.5, 47.3, 48.4]

# number of augmented views S
#  [ImageNet100+HTBA, ImageNet100+FTrojan, CIFAR10+HTBA, CIFAR10+FTrojan, CIFAR100+HTBA, CIFAR100+FTrojan]

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

dataset_names = [
    "ImageNet100+HTBA",
    "ImageNet100+FTrojan",
    "CIFAR10+HTBA",
    "CIFAR10+FTrojan",
    "CIFAR100+HTBA",
    "CIFAR100+FTrojan",
]

x_labels = ["-", "1", "4", "16", "32", "64", "128"]
x_values = list(range(len(x_labels)))

method_values = {
    "BYOL": {
        "ImageNet100+HTBA": [2.7, 11.8, 0.7, 0.4, 0.4, 0.4, 0.4],
        "ImageNet100+FTrojan": [42.0, 5.4, 0.9, 0.7, 0.7, 0.6, 0.6],
        "CIFAR10+HTBA": [4.3, 1.9, 4.1, 4.1, 4.4, 3.7, 4.2],
        "CIFAR10+FTrojan": [83.2, 28.5, 48.5, 40.6, 37.5, 37.0, 38.6],
        "CIFAR100+HTBA": [45.4, 0.4, 0.1, 0.0, 0.0, 0.0, 0.0],
        "CIFAR100+FTrojan": [87.9, 1.0, 0.2, 0.1, 0.0, 0.0, 0.0],
    },
    "MoCoV2": {
        "ImageNet100+HTBA": [6.4, 8.2, 3.2, 2.9, 2.9, 2.6, 2.7],
        "ImageNet100+FTrojan": [43.5, 2.8, 1.6, 1.3, 1.2, 1.2, 1.2],
        "CIFAR10+HTBA": [82.4, 50.6, 53.8, 53.6, 53.8, 54.2, 54.1],
        "CIFAR10+FTrojan": [1.9, 1.2, 1.3, 1.3, 1.2, 1.2, 1.2],
        "CIFAR100+HTBA": [41.3, 11.7, 5.9, 4.3, 4.7, 3.9, 4.7],
        "CIFAR100+FTrojan": [0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    },
    "SimCLR": {
        "ImageNet100+HTBA": [42.1, 8.5, 1.2, 1.2, 1.2, 1.2, 1.2],
        "ImageNet100+FTrojan": [34.1, 1.5, 0.5, 0.4, 0.4, 0.4, 0.3],
        "CIFAR10+HTBA": [84.5, 21.1, 2.0, 1.0, 1.0, 1.3, 1.3],
        "CIFAR10+FTrojan": [81.8, 23.6, 29.0, 24.3, 25.6, 24.2, 27.2],
        "CIFAR100+HTBA": [79.6, 11.5, 0.0, 0.0, 0.0, 0.0, 0.0],
        "CIFAR100+FTrojan": [70.9, 11.6, 0.0, 0.0, 0.0, 0.0, 0.0],
    },
}

fig, axes = plt.subplots(3, 6, figsize=(24, 12), squeeze=False)
# fig.suptitle("ASR trend across augmented view count S", fontsize=16, y=0.98)

for row_idx, method in enumerate(["BYOL", "MoCoV2", "SimCLR"]):
    for col_idx, dataset_name in enumerate(dataset_names):
        ax = axes[row_idx, col_idx]
        y = method_values[method][dataset_name]
        ax.plot(
            x_values, y, marker="o", linewidth=2, markersize=5, color="mediumpurple"
        )
        ax.set_title(f"{method}+{dataset_name}", fontsize=18)
        ax.set_xticks(x_values)
        ax.set_xticklabels(x_labels, rotation=20, fontsize=14)
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.4)
        ax.set_ylim(bottom=0)

plt.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("ablation_s_asr_trend.png", dpi=300)
plt.close(fig)
