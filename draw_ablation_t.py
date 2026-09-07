[43.2, 43.1, 87.2, 88.2, 56.5, 57.4]
[42.1, 42.0, 86.6, 87.1, 55.3, 55.9]
[41.4, 41.4, 87.0, 86.9, 54.5, 54.8]
[40.6, 40.7, 86.8, 86.5, 53.8, 53.9]
[39.7, 39.8, 86.4, 86.3, 53.1, 53.1]

[42.9, 42.6, 78.5, 79.6, 46.0, 46.2]
[42.5, 42.1, 78.2, 79.0, 46.3, 45.7]
[41.6, 41.5, 77.8, 78.5, 45.6, 44.8]
[40.8, 40.8, 77.6, 77.9, 44.6, 43.9]
[39.7, 40.1, 77.0, 77.3, 43.6, 42.8]

[38.3, 39.1, 85.2, 85.3, 49.7, 50.5]
[37.8, 37.9, 83.3, 82.1, 48.7, 49.1]
[37.0, 37.0, 81.9, 80.4, 47.3, 48.3]
[36.2, 35.9, 80.5, 79.3, 46.6, 47.4]
[35.1, 34.7, 79.8, 78.5, 46.5, 46.6]


# model-level backdoor-channel estimation number t

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

#  [ImageNet100+HTBA, ImageNet100+FTrojan, CIFAR10+HTBA, CIFAR10+FTrojan, CIFAR100+HTBA, CIFAR100+FTrojan]
# The x-axis values correspond to: uncleanse, 40, 70, 100, 130


dataset_names = [
    "ImageNet100+HTBA",
    "ImageNet100+FTrojan",
    "CIFAR10+HTBA",
    "CIFAR10+FTrojan",
    "CIFAR100+HTBA",
    "CIFAR100+FTrojan",
]

method_values = {
    "BYOL": {
        "ImageNet100+HTBA": [2.7, 0.5, 0.4, 0.4, 0.3],
        "ImageNet100+FTrojan": [42.0, 0.8, 0.6, 0.6, 0.3],
        "CIFAR10+HTBA": [4.3, 3.3, 3.7, 3.3, 3.5],
        "CIFAR10+FTrojan": [83.2, 40.8, 37.0, 34.3, 32.6],
        "CIFAR100+HTBA": [45.4, 0.0, 0.0, 0.0, 0.0],
        "CIFAR100+FTrojan": [87.9, 0.1, 0.0, 0.0, 0.0],
    },
    "MoCoV2": {
        "ImageNet100+HTBA": [6.4, 3.8, 2.6, 2.1, 1.8],
        "ImageNet100+FTrojan": [43.5, 1.6, 1.2, 0.9, 0.7],
        "CIFAR10+HTBA": [82.4, 55.2, 54.2, 49.9, 45.0],
        "CIFAR10+FTrojan": [1.9, 1.5, 1.2, 1.0, 0.9],
        "CIFAR100+HTBA": [41.3, 7.4, 3.9, 2.6, 2.1],
        "CIFAR100+FTrojan": [0.5, 0.1, 0.1, 0.1, 0.1],
    },
    "SimCLR": {
        "ImageNet100+HTBA": [42.1, 1.3, 1.2, 1.0, 1.0],
        "ImageNet100+FTrojan": [34.1, 0.5, 0.4, 0.3, 0.2],
        "CIFAR10+HTBA": [84.5, 3.0, 1.3, 0.6, 0.6],
        "CIFAR10+FTrojan": [81.8, 33.4, 24.2, 19.8, 17.6],
        "CIFAR100+HTBA": [79.6, 0.4, 0.0, 0.0, 0.0],
        "CIFAR100+FTrojan": [70.9, 0.2, 0.0, 0.0, 0.0],
    },
}

x_labels = ["-", "40", "70", "100", "130"]
x_values = list(range(len(x_labels)))

fig, axes = plt.subplots(3, 6, figsize=(24, 12), squeeze=False)
# fig.suptitle("ASR trend across cleansing thresholds", fontsize=16, y=0.98)

for row_idx, method in enumerate(["BYOL", "MoCoV2", "SimCLR"]):
    for col_idx, dataset_name in enumerate(dataset_names):
        ax = axes[row_idx, col_idx]
        y = method_values[method][dataset_name]
        ax.plot(
            x_values, y, marker="o", linewidth=2, markersize=5, color="mediumseagreen"
        )
        ax.set_title(f"{method}+{dataset_name}", fontsize=18)
        ax.set_xticks(x_values)
        ax.set_xticklabels(x_labels, rotation=20, fontsize=14)
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.4)
        ax.set_ylim(bottom=0)

plt.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("ablation_t_asr_trend.png", dpi=300)
plt.close(fig)
